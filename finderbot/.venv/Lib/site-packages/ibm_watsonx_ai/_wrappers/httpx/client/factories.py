#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Mapping, cast, overload

import httpx
from httpx._utils import get_environment_proxies

from ibm_watsonx_ai._wrappers.httpx.client.httpx_async_client import HTTPXAsyncClient
from ibm_watsonx_ai._wrappers.httpx.client.httpx_client import HTTPXClient
from ibm_watsonx_ai._wrappers.httpx.global_httpx_settings import GlobalHttpxSettings
from ibm_watsonx_ai._wrappers.httpx.retry_transport import (
    AsyncRetryTransport,
    RetryTransport,
    retry_transport_factory,
)

if TYPE_CHECKING:
    from ibm_watsonx_ai.client import APIClient


def _get_proxy_key(key: str) -> str:
    if key in ["http", "https"]:
        return key + "://"

    return key


@overload
def httpx_client_factory(
    *,
    is_async: Literal[True],
    api_client: APIClient,
    limits: httpx.Limits = GlobalHttpxSettings.HTTPX_DEFAULT_LIMIT,
    timeout: httpx.Timeout = GlobalHttpxSettings.HTTPX_DEFAULT_TIMEOUT,
    proxies: Mapping[str, str | None] | None = None,
) -> HTTPXAsyncClient: ...


@overload
def httpx_client_factory(
    *,
    is_async: Literal[False],
    api_client: APIClient,
    limits: httpx.Limits = GlobalHttpxSettings.HTTPX_DEFAULT_LIMIT,
    timeout: httpx.Timeout = GlobalHttpxSettings.HTTPX_DEFAULT_TIMEOUT,
    proxies: Mapping[str, str | None] | None = None,
) -> HTTPXClient: ...


@overload
def httpx_client_factory(
    *,
    is_async: bool,
    api_client: APIClient,
    limits: httpx.Limits = GlobalHttpxSettings.HTTPX_DEFAULT_LIMIT,
    timeout: httpx.Timeout = GlobalHttpxSettings.HTTPX_DEFAULT_TIMEOUT,
    proxies: Mapping[str, str | None] | None = None,
) -> HTTPXClient | HTTPXAsyncClient: ...


def httpx_client_factory(
    *,
    api_client: APIClient,
    is_async: bool,
    limits: httpx.Limits = GlobalHttpxSettings.HTTPX_DEFAULT_LIMIT,
    timeout: httpx.Timeout = GlobalHttpxSettings.HTTPX_DEFAULT_TIMEOUT,
    proxies: Mapping[str, str | None] | None = None,
) -> HTTPXClient | HTTPXAsyncClient:
    """
    Create a client class instance based on provided arguments, environment variables
    and global state. Depending on whether the transport should be asynchronous, returns
    either HTTPXClient or HTTPXAsyncClient.
    """

    # Passed proxies > Proxies from global state (set by credentials) > Environment proxies
    proxies = proxies or GlobalHttpxSettings.proxies or get_environment_proxies()

    if proxies:
        transport = None
        mounts = {
            _get_proxy_key(key): retry_transport_factory(
                is_async, api_client, limits, proxy
            )
            for key, proxy in proxies.items()
        }
        verify = next(iter(mounts.values())).effective_verify
    else:
        transport = retry_transport_factory(is_async, api_client, limits)
        mounts = None
        verify = transport.effective_verify

    if is_async:
        return HTTPXAsyncClient(
            transport=cast(AsyncRetryTransport | None, transport),
            mounts=cast(dict[str, AsyncRetryTransport] | None, mounts),
            timeout=timeout,
            verify=verify,
            limits=limits,
        )

    return HTTPXClient(
        transport=cast(RetryTransport | None, transport),
        mounts=cast(dict[str, RetryTransport] | None, mounts),
        timeout=timeout,
        verify=verify,
        limits=limits,
    )
