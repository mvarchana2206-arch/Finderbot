#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient


def get_batch_inference_headers(
    client: APIClient,
    content_type: str | None = "application/json",
    include_user_agent: bool = True,
) -> dict:
    headers = client.get_headers(
        content_type=content_type,
        include_user_agent=include_user_agent,
        include_container_id=True,
    )

    # Authorization must happen via API key
    headers["Authorization"] = f"Bearer {client.credentials.api_key}"

    return headers
