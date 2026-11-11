#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

import base64
import json
from typing import Any


def get_token_payload(token: str) -> dict[str, Any]:
    """Get info (aka payload part) from token.

    :param token: token with encoded information
    :type token: str

    :returns: info from token
    :rtype: dict[str, Any]

    """
    token_parts = token.split(".")
    token_padded = token_parts[1] + "==="

    try:
        token_info = json.loads(
            base64.b64decode(token_padded).decode("utf-8", errors="ignore")
        )
    except ValueError:
        # If there is a problem with decoding (e.g. special char in token), add altchars
        token_info = json.loads(
            base64.b64decode(token_padded, altchars="_-").decode(
                "utf-8", errors="ignore"
            )
        )

    return token_info
