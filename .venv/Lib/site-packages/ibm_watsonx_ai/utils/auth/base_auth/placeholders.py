#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from ibm_watsonx_ai.utils.auth.base_auth.base_auth import BaseAuth
from ibm_watsonx_ai.utils.autoai.errors import TokenRemovedDuringClientCopy


class TokenRemovedDuringClientCopyPlaceholder(BaseAuth):
    """Placeholder which indicates that no auth is currently available until `APIClient.set_token(token)` is used."""

    def get_token(self) -> str:
        """Raise an error when `get_token()` is called."""
        raise TokenRemovedDuringClientCopy()

    async def aget_token(self) -> str:
        """Raise an error when `aget_token()` is called."""
        raise TokenRemovedDuringClientCopy()

    def set_token(self, token: str) -> None:
        """Raise an error when `set_token()` is called."""
        raise TokenRemovedDuringClientCopy()
