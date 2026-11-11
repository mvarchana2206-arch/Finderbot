#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from abc import ABC
from functools import cached_property
from typing import Iterable

import httpx

from ibm_watsonx_ai._wrappers.httpx.global_httpx_settings import GlobalHttpxSettings
from ibm_watsonx_ai._wrappers.httpx.retry_transport.exceptions import NoneResponseError


class RetryTransportMixin(ABC):
    """Mixin to httpx transport classes with retry functionality"""

    SSL_ERROR_KEYWORDS = [
        "CERTIFICATE_VERIFY_FAILED",
        "certificate verify failed",
        "SSL",
        "TLS",
        "self-signed certificate",
    ]

    def __init__(
        self,
        retries: int,
        backoff_factor: float,
        status_forcelist: Iterable[int],
        verify_initial: bool | str | None,
        allow_ssl_fallback: bool,
        limits: httpx.Limits,
        proxy: httpx._types.ProxyTypes | None,
    ) -> None:
        self.proxies = GlobalHttpxSettings.proxies
        self.retries = retries
        self.backoff_factor = backoff_factor
        self.status_forcelist = status_forcelist
        self.allow_ssl_fallback = allow_ssl_fallback
        self.original_verify = verify_initial
        self.limits = limits

        self.proxy: httpx._types.ProxyTypes | None
        if self.proxies and (
            proxy_url := self.proxies.get("https") or self.proxies.get("http")
        ):
            self.proxy = httpx.Proxy(proxy_url)
        elif proxy:
            self.proxy = proxy
        else:
            self.proxy = None

        self._ssl_fallback_attempted = False

    @cached_property
    def effective_verify(self) -> bool | str:
        """Get the effective verify value for the HTTP client."""
        return GlobalHttpxSettings.get_effective_verify()

    def _is_ssl_error(self, error: Exception) -> bool:
        if isinstance(error, NoneResponseError):
            # The `NoneResponseError` can be caused by certificate verification
            # failure, so because we are missing information on why the exception
            # was raised, we can assume that the underlying error is caused by it.
            # If our assumption is correct, we will no longer encounter this error,
            # otherwise it should keep appearing and we would raise an exception
            # in the next iteration.
            return True

        return any(ssl_keyword in str(error) for ssl_keyword in self.SSL_ERROR_KEYWORDS)

    def _handle_retry_loop_exit(
        self, response: httpx.Response | None, exceptions: list[Exception]
    ) -> httpx.Response:
        if response is not None:
            return response

        if exceptions:
            raise ExceptionGroup(
                f"Request could not be completed successfully in {self.retries + 1} attempts",
                exceptions,
            )

        # If any iteration has been performed, either `response` or `exceptions` must be truthy
        raise ValueError(f"Number of retries ({self.retries}) cannot be negative")

    def _is_retry_on_server_disconnect(
        self, error: Exception, is_ssl_error: bool, attempt: int
    ) -> bool:
        """Retry on server disconnect (stale keep-alive connection) - but not SSL errors"""
        if is_ssl_error:
            return False

        return isinstance(error, httpx.RemoteProtocolError) and attempt < self.retries

    def _is_ssl_fallback_attempt(self, is_ssl_error: bool) -> bool:
        is_ssl_fallback_attempt = (
            is_ssl_error
            and not self._ssl_fallback_attempted
            and self.original_verify is None
            and self.allow_ssl_fallback
        )

        if is_ssl_fallback_attempt:
            self._ssl_fallback_attempted = True
            GlobalHttpxSettings.verify = False

        return is_ssl_fallback_attempt

    def _is_verify_error(self, is_ssl_error: bool) -> bool:
        """When verify is explicitly set to `True` or a path to CA bundle"""
        if not is_ssl_error:
            return False

        return self.effective_verify is True or isinstance(self.effective_verify, str)

    def _is_proxy_error(self, error: Exception) -> bool:
        """
        If proxies are configured and we get a connection error,
        `httpx.ProxyError` should be raised
        """
        if not self.proxies:
            return False

        return isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout))

    def _get_sleep_time(
        self, response: httpx.Response | None, attempt: int
    ) -> float | None:
        is_retry_possible = (
            response is not None
            and response.status_code in self.status_forcelist
            and attempt != self.retries
        )

        return (
            min(self.backoff_factor * (2**attempt), self.retries)
            if is_retry_possible
            else None
        )
