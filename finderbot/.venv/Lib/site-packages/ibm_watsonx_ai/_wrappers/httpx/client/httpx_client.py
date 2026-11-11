#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import httpx

from ibm_watsonx_ai._wrappers.httpx.client.encode_json_payload import (
    encode_json_payload,
)


class HTTPXClient(httpx.Client):
    """Wrapper for httpx Sync Client"""

    # Apply `encode_json_payload` decorator to `post` and `stream` methods
    post = encode_json_payload(httpx.Client.post)
    stream = encode_json_payload(httpx.Client.stream)

    def __del__(self) -> None:
        try:
            # Closing the connection pool when the object is deleted
            self.close()
        except Exception:
            pass
