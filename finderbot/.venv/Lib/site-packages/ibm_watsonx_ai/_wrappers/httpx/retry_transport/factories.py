#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import httpx

from ibm_watsonx_ai._wrappers.httpx.global_httpx_settings import GlobalHttpxSettings
from ibm_watsonx_ai._wrappers.httpx.retry_transport.async_retry_transport import (
    AsyncRetryTransport,
)
from ibm_watsonx_ai._wrappers.httpx.retry_transport.retry_transport import (
    RetryTransport,
)
from ibm_watsonx_ai._wrappers.httpx.retry_transport.utils import raise_verify_error

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient

TRANSPORT_RETRIES = 3
TRANSPORT_BACKOFF_FACTOR = 0.3
TRANSPORT_STATUS_FORCELIST = (401, 500, 502, 503, 504, 520, 521, 524)


@overload
def retry_transport_factory(
    is_async: Literal[True],
    api_client: APIClient,
    limits: httpx.Limits = GlobalHttpxSettings.HTTPX_DEFAULT_LIMIT,
    proxy: str | None = None,
) -> AsyncRetryTransport: ...


@overload
def retry_transport_factory(
    is_async: Literal[False],
    api_client: APIClient,
    limits: httpx.Limits = GlobalHttpxSettings.HTTPX_DEFAULT_LIMIT,
    proxy: str | None = None,
) -> RetryTransport: ...


@overload
def retry_transport_factory(
    is_async: bool,
    api_client: APIClient,
    limits: httpx.Limits = GlobalHttpxSettings.HTTPX_DEFAULT_LIMIT,
    proxy: str | None = None,
) -> RetryTransport | AsyncRetryTransport: ...


def retry_transport_factory(
    is_async: bool,
    api_client: APIClient,
    limits: httpx.Limits = GlobalHttpxSettings.HTTPX_DEFAULT_LIMIT,
    proxy: str | None = None,
) -> RetryTransport | AsyncRetryTransport:
    """
    Create a retry transport class instance based on provided arguments, environment
    variables and global state. Depending on whether the transport should be
    asynchronous, returns either RetryTransport or AsyncRetryTransport.
    """

    verify_initial = (
        api_client.credentials.verify
        if api_client.credentials.verify is not None
        else GlobalHttpxSettings.get_verify_from_environment()
    )

    # Allow SSL fallback only if all verify sources (credentials, env var, global verify) are None
    allow_ssl_fallback = verify_initial is None

    transport_cls = AsyncRetryTransport if is_async else RetryTransport

    try:
        return transport_cls(
            retries=TRANSPORT_RETRIES,
            backoff_factor=TRANSPORT_BACKOFF_FACTOR,
            status_forcelist=TRANSPORT_STATUS_FORCELIST,
            verify_initial=verify_initial,
            allow_ssl_fallback=allow_ssl_fallback,
            limits=limits,
            proxy=proxy,
        )
    except FileNotFoundError as e:
        # When verify is a string path that doesn't exist
        if isinstance(verify_initial, str):
            raise_verify_error(e)
        raise
