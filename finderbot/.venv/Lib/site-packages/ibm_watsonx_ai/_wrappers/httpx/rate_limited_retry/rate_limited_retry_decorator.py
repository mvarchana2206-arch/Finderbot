#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import os
import random
import time
from contextlib import asynccontextmanager, contextmanager
from functools import wraps
from typing import (
    TYPE_CHECKING,
    AsyncContextManager,
    AsyncIterator,
    Awaitable,
    Callable,
    ContextManager,
    Iterator,
    ParamSpec,
    Sequence,
    TypeVar,
)

import httpx

from ibm_watsonx_ai._wrappers.httpx.rate_limited_retry.token_bucket import TokenBucket

if TYPE_CHECKING:
    from ibm_watsonx_ai.client import APIClient


P = ParamSpec("P")
T = TypeVar("T", int, float)


class RateLimitedRetryDecorator:
    DEFAULT_RETRY_STATUS_CODES = (429, 503, 504, 520)
    MAX_RETRY_DELAY = 8
    DEFAULT_DELAY = 0.5

    MAX_RETRIES = 10  # number of retries after the first failure
    REMAINING_LIMIT_HEADER = "x-requests-limit-remaining"

    def __init__(
        self,
        *,
        api_client: APIClient,
        rate_limiter: TokenBucket,
        retry_status_codes: Sequence[int] | None = None,
        max_retries: int | None = None,
        delay_time: float | None = None,
    ) -> None:
        self._api_client = api_client
        self.rate_limiter = rate_limiter
        self.retry_status_codes = retry_status_codes
        self.max_retries = max_retries
        self.delay_time = delay_time

    def get_remaining_rate_limit(
        self, response: httpx.Response, default: int | None = None
    ) -> int:
        remaining_limit_header = response.headers.get(self.REMAINING_LIMIT_HEADER)
        if remaining_limit_header is not None:
            return int(remaining_limit_header)

        return default if default is not None else self.rate_limiter.capacity

    def _get_backoff_timeout(self, delay_time: float, attempt: int) -> float:
        jitter = 1 + 0.25 * random.random()
        exponent = 1 << attempt  # 2 ** attempt, but returns int

        sleep_seconds = min(delay_time * exponent, self.MAX_RETRY_DELAY)
        return sleep_seconds * jitter

    def _get_config_value(
        self,
        instance_value: T | None,
        env_var_name: str,
        argument_value: T | None,
        default_value: T,
        converter: Callable[[str], T],
    ) -> T:
        """
        Get the configuration value with the following priorities:
        1. instance attribute
        2. environment variable
        3. argument value
        4. default value
        """

        if instance_value is not None:
            return instance_value

        if (env_value := os.environ.get(env_var_name)) is not None:
            return converter(env_value)

        if argument_value is not None:
            return argument_value

        return default_value

    def _get_max_retries(self, argument_value: int | None) -> int:
        return self._get_config_value(
            self.max_retries,
            "WATSONX_MAX_RETRIES",
            argument_value,
            self.MAX_RETRIES,
            int,
        )

    def _get_delay_time(self, argument_value: float | None) -> float:
        return self._get_config_value(
            self.delay_time,
            "WATSONX_DELAY_TIME",
            argument_value,
            self.DEFAULT_DELAY,
            float,
        )

    def _get_retry_status_codes(
        self, argument_value: Sequence[int] | None
    ) -> Sequence[int]:
        """
        Get the retry status codes with the following priorities:
        1. instance attribute
        2. environment variable
        3. argument value
        4. default value
        """

        if self.retry_status_codes is not None:
            return self.retry_status_codes

        if (env_value := os.environ.get("WATSONX_RETRY_STATUS_CODES")) is not None:
            env_retry_status_codes = [
                int(code.strip())
                for code in env_value.strip("[]").split(",")
                if code.strip().isdigit()
            ]
        else:
            env_retry_status_codes = []

        if env_retry_status_codes:
            return env_retry_status_codes

        if argument_value:
            return argument_value

        return self.DEFAULT_RETRY_STATUS_CODES

    def _handle_backoff(
        self, response: httpx.Response, delay_time: float, attempt: int
    ) -> None:
        backoff_timeout = self._get_backoff_timeout(delay_time, attempt)

        if self._api_client.ICP_PLATFORM_SPACES:
            time.sleep(backoff_timeout)
            return

        if self.get_remaining_rate_limit(response) == 0:
            self.rate_limiter.adjust_tokens(0)
        else:
            time.sleep(backoff_timeout)

        self.rate_limiter.acquire()

    async def _handle_async_backoff(
        self, response: httpx.Response, delay_time: float, attempt: int
    ) -> None:
        backoff_timeout = self._get_backoff_timeout(delay_time, attempt)

        if self._api_client.ICP_PLATFORM_SPACES:
            await asyncio.sleep(backoff_timeout)
            return

        if self.get_remaining_rate_limit(response) == 0:
            await self.rate_limiter.async_adjust_tokens(0)
        else:
            await asyncio.sleep(backoff_timeout)

        await self.rate_limiter.acquire_async()

    def _is_backoff_required(
        self,
        response: httpx.Response,
        attempt: int,
        retry_status_codes: Sequence[int],
        max_retries: int,
    ) -> bool:
        return (
            response is not None
            and (response.status_code in retry_status_codes)
            and attempt != max_retries
        )

    def _handle_retry_loop_exit(
        self, response: httpx.Response | None, max_retries: int
    ) -> httpx.Response:
        if response is not None:
            return response

        raise ValueError(f"Number of retries ({max_retries}) cannot be negative")

    def rate_limited_retry(
        self,
        request_function: Callable[P, httpx.Response],
        *,
        max_retries: int | None = None,
        delay_time: float | None = None,
        retry_status_codes: Sequence[int] | None = None,
    ) -> Callable[P, httpx.Response]:
        @wraps(request_function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> httpx.Response:
            response: httpx.Response | None = None

            actual_max_retries = self._get_max_retries(max_retries)
            actual_delay_time = self._get_delay_time(delay_time)
            actual_retry_status_codes = self._get_retry_status_codes(retry_status_codes)

            for attempt in range(actual_max_retries + 1):
                if response is not None:
                    response.close()

                response = request_function(*args, **kwargs)

                if self._is_backoff_required(
                    response, attempt, actual_retry_status_codes, actual_max_retries
                ):
                    self._handle_backoff(response, actual_delay_time, attempt)
                else:
                    break

            return self._handle_retry_loop_exit(response, actual_max_retries)

        return wrapper

    def rate_limited_async_retry(
        self,
        request_function: Callable[P, Awaitable[httpx.Response]],
        *,
        max_retries: int | None = None,
        delay_time: float | None = None,
        retry_status_codes: Sequence[int] | None = None,
    ) -> Callable[P, Awaitable[httpx.Response]]:
        @wraps(request_function)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> httpx.Response:
            response: httpx.Response | None = None

            actual_max_retries = self._get_max_retries(max_retries)
            actual_delay_time = self._get_delay_time(delay_time)
            actual_retry_status_codes = self._get_retry_status_codes(retry_status_codes)

            for attempt in range(actual_max_retries + 1):
                if response is not None:
                    response.close()

                response = await request_function(*args, **kwargs)

                if self._is_backoff_required(
                    response, attempt, actual_retry_status_codes, actual_max_retries
                ):
                    await self._handle_async_backoff(
                        response, actual_delay_time, attempt
                    )
                else:
                    break

            return self._handle_retry_loop_exit(response, actual_max_retries)

        return wrapper

    def rate_limited_retry_stream(
        self,
        request_function: Callable[P, ContextManager[httpx.Response]],
        *,
        max_retries: int | None = None,
        delay_time: float | None = None,
        retry_status_codes: Sequence[int] | None = None,
    ) -> Callable[P, ContextManager[httpx.Response]]:
        @wraps(request_function)
        @contextmanager
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Iterator[httpx.Response]:
            actual_max_retries = self._get_max_retries(max_retries)
            actual_delay_time = self._get_delay_time(delay_time)
            actual_retry_status_codes = self._get_retry_status_codes(retry_status_codes)

            for attempt in range(actual_max_retries + 1):
                with request_function(*args, **kwargs) as response:
                    if self._is_backoff_required(
                        response, attempt, actual_retry_status_codes, actual_max_retries
                    ):
                        self._handle_backoff(response, actual_delay_time, attempt)
                        continue

                    yield response
                    break

        return wrapper

    def rate_limited_async_retry_stream(
        self,
        request_function: Callable[P, AsyncContextManager[httpx.Response]],
        *,
        max_retries: int | None = None,
        delay_time: float | None = None,
        retry_status_codes: Sequence[int] | None = None,
    ) -> Callable[P, AsyncContextManager[httpx.Response]]:
        @wraps(request_function)
        @asynccontextmanager
        async def wrapper(
            *args: P.args, **kwargs: P.kwargs
        ) -> AsyncIterator[httpx.Response]:
            actual_max_retries = self._get_max_retries(max_retries)
            actual_delay_time = self._get_delay_time(delay_time)
            actual_retry_status_codes = self._get_retry_status_codes(retry_status_codes)

            for attempt in range(actual_max_retries + 1):
                async with request_function(*args, **kwargs) as response:
                    if self._is_backoff_required(
                        response, attempt, actual_retry_status_codes, actual_max_retries
                    ):
                        await self._handle_async_backoff(
                            response, actual_delay_time, attempt
                        )
                        continue

                    yield response

                return

        return wrapper
