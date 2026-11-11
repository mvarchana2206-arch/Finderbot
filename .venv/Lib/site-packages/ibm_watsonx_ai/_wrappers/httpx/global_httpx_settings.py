#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import os
from functools import wraps
from typing import Callable, ParamSpec, TypeVar

import httpx

P = ParamSpec("P")
R = TypeVar("R")


class GlobalHttpxSettings:
    """Holds global state for all requests made using the `httpx` library."""

    HTTPX_KEEPALIVE_EXPIRY = 5

    HTTPX_DEFAULT_TIMEOUT = httpx.Timeout(timeout=30 * 60, connect=10)

    HTTPX_DEFAULT_LIMIT = httpx.Limits(
        max_connections=10,
        max_keepalive_connections=10,
        keepalive_expiry=HTTPX_KEEPALIVE_EXPIRY,
    )

    verify: bool | str | None = None
    proxies: dict[str, str] | None = None

    @classmethod
    def get_verify_from_environment(cls) -> bool | str | None:
        """
        Get the verify value implied by environment variables and global state.
        Prioritizes environment variable over global state.
        """

        env_val = os.environ.get("WX_CLIENT_VERIFY_REQUESTS")

        if env_val is None:
            return cls.verify
        if env_val == "" or env_val.lower() == "true":
            # Empty string means True (default verification)
            return True
        if env_val.lower() == "false":
            return False
        return env_val

    @classmethod
    def get_effective_verify(cls) -> bool | str:
        """
        Get the effective verify value from environment variable and global state.
        Prioritizes environment variable over global state.
        Defaults to True if none are set.
        Returns the verify value to use for SSL verification.
        """

        if (verify := cls.get_verify_from_environment()) is not None:
            return verify

        return True

    @classmethod
    def set_default_verify(cls, func: Callable[P, R]) -> Callable[P, R]:
        """
        This decorator sets the default value of the `verify` argument passed to
        the provided function. The default value is equal to `get_effective_verify()`.
        """

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if "verify" not in kwargs:
                kwargs["verify"] = cls.get_effective_verify()

            return func(*args, **kwargs)

        return wrapper
