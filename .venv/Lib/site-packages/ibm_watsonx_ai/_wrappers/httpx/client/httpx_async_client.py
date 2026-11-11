#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import asyncio

import httpx

from ibm_watsonx_ai._wrappers.httpx.client.encode_json_payload import (
    encode_json_payload,
)


class HTTPXAsyncClient(httpx.AsyncClient):
    """Wrapper for httpx Async Client"""

    # Apply `encode_json_payload` decorator to `post` and `stream` methods
    post = encode_json_payload(httpx.AsyncClient.post)
    stream = encode_json_payload(httpx.AsyncClient.stream)

    def __del__(self) -> None:
        try:
            # Closing the connection pool when the object is deleted
            asyncio.get_running_loop().create_task(self.aclose())
        except Exception:
            pass
