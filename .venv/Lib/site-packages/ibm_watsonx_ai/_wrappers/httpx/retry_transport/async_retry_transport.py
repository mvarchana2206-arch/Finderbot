#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import asyncio
from typing import Iterable

import httpx

from ibm_watsonx_ai._wrappers.httpx.retry_transport.exceptions import NoneResponseError
from ibm_watsonx_ai._wrappers.httpx.retry_transport.retry_transport_mixin import (
    RetryTransportMixin,
)
from ibm_watsonx_ai._wrappers.httpx.retry_transport.utils import raise_verify_error


class AsyncRetryTransport(RetryTransportMixin, httpx.AsyncHTTPTransport):
    def __init__(
        self,
        retries: int,
        backoff_factor: float,
        status_forcelist: Iterable[int],
        verify_initial: bool | str | None,
        allow_ssl_fallback: bool,
        limits: httpx.Limits = httpx._config.DEFAULT_LIMITS,
        proxy: httpx._types.ProxyTypes | None = None,
    ) -> None:
        RetryTransportMixin.__init__(
            self,
            retries,
            backoff_factor,
            status_forcelist,
            verify_initial,
            allow_ssl_fallback,
            limits,
            proxy,
        )

        verify_for_transport = True if verify_initial is None else verify_initial
        httpx.AsyncHTTPTransport.__init__(
            self,
            verify=verify_for_transport,
            limits=self.limits,
            proxy=self.proxy,
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response: httpx.Response | None = None
        exceptions: list[Exception] = []

        for attempt in range(self.retries + 1):
            if response is not None:
                await response.aclose()

            try:
                response = await super().handle_async_request(request)
                if response is None:
                    raise NoneResponseError
            except (
                httpx.ConnectError,
                httpx.RemoteProtocolError,
                httpx.ReadTimeout,
                httpx.ConnectTimeout,
                NoneResponseError,
            ) as e:
                exceptions.append(e)
                is_ssl_error = self._is_ssl_error(e)

                if self._is_retry_on_server_disconnect(e, is_ssl_error, attempt):
                    continue

                if self._is_ssl_fallback_attempt(is_ssl_error):
                    await self.aclose()

                    AsyncRetryTransport.__init__(
                        self,
                        retries=self.retries,
                        backoff_factor=self.backoff_factor,
                        status_forcelist=self.status_forcelist,
                        verify_initial=False,
                        allow_ssl_fallback=self.allow_ssl_fallback,
                        limits=self.limits,
                        proxy=self.proxy,
                    )

                    response = await super().handle_async_request(request)
                elif self._is_verify_error(is_ssl_error):
                    raise_verify_error(e)
                elif self._is_proxy_error(e):
                    raise httpx.ProxyError(str(e)) from e
                else:
                    raise

            sleep_time = self._get_sleep_time(response, attempt)
            if sleep_time is not None:
                await asyncio.sleep(sleep_time)
            else:
                break

        return self._handle_retry_loop_exit(response, exceptions)
