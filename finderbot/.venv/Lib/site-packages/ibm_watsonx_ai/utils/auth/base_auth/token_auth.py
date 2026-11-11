#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------


from typing import Callable

from ibm_watsonx_ai.utils.auth.base_auth.base_auth import BaseAuth
from ibm_watsonx_ai.wml_resource import WMLResource


class TokenAuth(BaseAuth):
    """Basic authentication method, the object is keeping existing token and return it when asked.
    Token cannot be refreshed.

    :param token: token to be used with service
    :type token: str

    :param on_token_set: callback which allows to notify about token set
    :type on_token_set: function which takes no params and returns nothing, optional
    """

    def __init__(
        self,
        token: str,
        on_token_set: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self._on_token_set = on_token_set
        self.set_token(token)

    def get_token(self) -> str:
        """Returns the token. The token will not be refreshed.

        :returns: token to be used with service
        :rtype: str
        """
        return self._token

    async def aget_token(self) -> str:
        """Returns the token. The token will not be refreshed.

        :returns: token to be used with service
        :rtype: str
        """
        return self._token

    def set_token(self, token: str) -> None:
        """Set new token.

        :param token: token to be used with service
        :type token: str
        """
        WMLResource._validate_type(token, "token", str, mandatory=True)

        self._token = token

        if self._on_token_set:
            self._on_token_set()
