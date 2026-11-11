#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from abc import ABC, abstractmethod


class BaseAuth(ABC):
    """Base class for any authentication method used in the APIClient."""

    @abstractmethod
    def get_token(self) -> str:
        """Returns the token.

        :returns: token to be used with service
        :rtype: str
        """

    @abstractmethod
    async def aget_token(self) -> str:
        """Returns the token.

        :returns: token to be used with service
        :rtype: str
        """

    @abstractmethod
    def set_token(self, token: str) -> None:
        """Set new token.

        :param token: token to be used with service
        :type token: str
        """
