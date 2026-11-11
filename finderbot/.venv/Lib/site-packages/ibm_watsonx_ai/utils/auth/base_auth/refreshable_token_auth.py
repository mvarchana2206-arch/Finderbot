#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Callable

from ibm_watsonx_ai.utils.auth.base_auth.base_auth import BaseAuth
from ibm_watsonx_ai.utils.auth.base_auth.models import TokenInfo
from ibm_watsonx_ai.utils.auth.base_auth.utils import get_token_payload

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient


class RefreshableTokenAuth(BaseAuth, ABC):
    """Abstract base class of all authentication methods which are using token generation and refresh.

    :param api_client: initialized APIClient object with set project or space ID
    :type api_client: APIClient

    :param on_token_creation: callback which allows to notify about token creation
    :type on_token_creation: function which takes no params and returns nothing

    :param on_token_refresh: callback which allows to notify about token refresh
    :type on_token_refresh: function which takes no params and returns nothing

    :param refreshing_timedelta: time to expiration below which the token will be refreshed before use
    :type refreshing_timedelta: timedelta, optional
    """

    def __init__(
        self,
        api_client: APIClient,
        on_token_creation: Callable[[], None] | None,
        on_token_refresh: Callable[[], None] | None,
        refreshing_timedelta: timedelta | None = None,
    ) -> None:
        self._api_client = api_client
        self._on_token_creation = on_token_creation
        self._on_token_refresh = on_token_refresh
        self._refreshing_timedelta = refreshing_timedelta
        self._lock = threading.Lock()
        self._async_lock = asyncio.Lock()

        self._token = ""
        self._hardcoded_expiration_datetime: datetime | None = None

    def get_token(self) -> str:
        """Returns the token. If the token will be about to expire, it will be refreshed.

        :returns: token to be used with service
        :rtype: str
        """
        # serve token if it is ready and not refreshing without lock
        if self._token and not self._is_refresh_needed():
            return self._token

        with self._lock:
            if not self._token:
                self._save_token_data(self._generate_token())
                self._set_refreshing_timedelta_if_needed()

                if self._on_token_creation:
                    self._on_token_creation()

                return self._token

            if self._is_refresh_needed():
                self._save_token_data(self._refresh_token())

                if self._on_token_refresh:
                    self._on_token_refresh()

            return self._token

    async def aget_token(self) -> str:
        """Returns the token asynchronously. If the token will be about to expire, it will be refreshed.

        :returns: token to be used with service
        :rtype: str
        """
        # serve token if it is ready and not refreshing without lock
        if self._token and not self._is_refresh_needed():
            return self._token

        async with self._async_lock:
            if not self._token:
                self._save_token_data(await self._agenerate_token())
                self._set_refreshing_timedelta_if_needed()

                if self._on_token_creation:
                    self._on_token_creation()

                return self._token

            if self._is_refresh_needed():
                self._save_token_data(await self._arefresh_token())

                if self._on_token_refresh:
                    self._on_token_refresh()

            return self._token

    def set_token(self, token: str) -> None:
        self._token = token

    def _set_refreshing_timedelta_if_needed(self):
        """Set refreshing timedelta basing on expiration time if no refreshing timedelta was passed in constructor."""
        time_to_expiration = self._get_expiration_datetime() - datetime.now()

        if self._refreshing_timedelta is None:
            if time_to_expiration > timedelta(minutes=30):
                self._refreshing_timedelta = timedelta(minutes=15)
            elif time_to_expiration > timedelta(minutes=3):
                # for minimal cloud token expiration = 15 min, the refreshing timedelta will be 5 min
                self._refreshing_timedelta = (time_to_expiration) / 3
            else:
                # for token expiration time < 3 min, the refreshing time is always 1 min,
                # which sometimes triggers refresh always (for expiration time < 1 min)
                self._refreshing_timedelta = timedelta(minutes=1)

    @abstractmethod
    def _generate_token(self) -> TokenInfo:
        """Generate token from scratch using user provided credentials.

        :returns: token info to be used by auth method
        :rtype: TokenInfo
        """

    @abstractmethod
    async def _agenerate_token(self) -> TokenInfo:
        """Generate token from scratch using user provided credentials.

        :returns: token info to be used by auth method
        :rtype: TokenInfo
        """

    def _refresh_token(
        self,
    ) -> TokenInfo:
        """Refresh token.

        :returns: token info to be used by auth method
        :rtype: TokenInfo
        """
        # if not provided implementation, refresh is handled as generation from creds
        return self._generate_token()

    async def _arefresh_token(
        self,
    ) -> TokenInfo:
        """Refresh token.

        :returns: token info to be used by auth method
        :rtype: TokenInfo
        """
        # if not provided implementation, refresh is handled as generation from creds
        return await self._agenerate_token()

    def _is_refresh_needed(self) -> bool:
        """Check if the time of expiration is below minimal expiration timedelta.

        :returns: result of check
        :rtype: bool
        """
        exp_datetime = self._get_expiration_datetime()

        return (
            self._refreshing_timedelta is None
            or exp_datetime - datetime.now() < self._refreshing_timedelta
        )

    def _get_expiration_datetime(self) -> datetime:
        """Return expiration datetime. Implementation for JWT token.

        :returns: datetime of token expiration
        :rtype: datetime
        """
        if self._hardcoded_expiration_datetime is not None:
            return self._hardcoded_expiration_datetime

        token_info = get_token_payload(self._token)
        token_expire = token_info["exp"]

        return datetime.fromtimestamp(token_expire)

    def _save_token_data(self, token_info: TokenInfo) -> None:
        """Write data from TokenInfo into authentication method fields for its mechanism to work properly.

        :param token_info: data of token returned after generation or refresh of token
        :type token_info: TokenInfo
        """
        self._token = token_info.token
        self._hardcoded_expiration_datetime = token_info.expiration_datetime
