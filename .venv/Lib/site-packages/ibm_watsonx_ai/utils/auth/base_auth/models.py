#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TokenInfo:
    """Data class in which information about token and the token are returned to be put into auth method structures.
    `expiration_datetime` should be only set when token expiration information cannot be extracted from the token,
    and this is when token is not JWT token. Otherwise, `expiration_datetime` should be set to None.

    :param token: token to be used with service
    :type token: str

    :param expiration_datetime: datetime of token expiration, if the token is not JWT, otherwise should be set to None
    :type expiration_datetime: datetime or None, optional
    """

    token: str
    expiration_datetime: datetime | None = field(default=None)
