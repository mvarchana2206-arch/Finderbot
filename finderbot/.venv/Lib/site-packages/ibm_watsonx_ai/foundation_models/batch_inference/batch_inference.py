#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ibm_watsonx_ai.wml_client_error import InvalidValue, MissingValue, WMLClientError
from ibm_watsonx_ai.wml_resource import WMLResource

from .files import Files
from .utils import get_batch_inference_headers

if TYPE_CHECKING:
    from pandas import DataFrame

    from ibm_watsonx_ai import APIClient


class BatchInference(WMLResource):
    """Initialize BatchInference resource.

    :param api_client: APIClient instance with a set project ID or space ID and authenticated with `api_key`
    :type api_client: APIClient

    **Example:**

    .. code-block:: python

        from ibm_watsonx_ai import APIClient, Credentials
        from ibm_watsonx_ai.foundation_models import BatchInference

        credentials = Credentials(url="<url>", api_key=IAM_API_KEY)
        client = APIClient(credentials, project_id="<project_id>")

        batch_inference = BatchInference(api_client=client)

    """

    def __init__(self, api_client: APIClient) -> None:
        WMLResource.__init__(self, __name__, api_client)

        if self._client.credentials.api_key is None:
            raise MissingValue(
                value_name="api_key",
                reason="API key is required for authentication. Pass it to `Credentials` instance before the client initialization.",
            )
        self._client._check_if_either_is_set()

        self.files = Files(api_client)

    def create(
        self,
        endpoint: str,
        completion_window: str,
        metadata: dict | None = None,
        input_file_path: str | Path | None = None,
        input_file_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a batch inference job. If provided with file_path first upload the batch file.

        :param endpoint: API endpoint to use for processing each batch item, e.g. "/v1/chat/completions"
        :type endpoint: str

        :param completion_window: time window for completion of the batch job, e.g. "24h"
        :type completion_window: str

        :param metadata: metadata properties
        :type metadata: dict, optional

        :param input_file_path: Path to the file to upload for the batch job
        :type input_file_path: str | Path, optional

        :param input_file_id: ID of the uploaded input file for the batch job
        :type input_file_id: str, optional

        :return: batch job details
        :rtype: dict
        :raises MissingValue: If both input_file_path and input_file_id are None

        **Example using file path:**

        .. code-block:: python

            batch_details = batch_inference.create(
                endpoint="/v1/chat/completions",
                completion_window="24h",
                input_file_path="/path/to/batch_file.jsonl",
            )

        **Example using file id obtained by uploading file by File API:**

        .. code-block:: python

            batch_details = batch_inference.create(
                endpoint="/v1/chat/completions",
                completion_window="24h",
                input_file_id=file_id,
            )


        """

        self._validate_type(endpoint, "endpoint", str, True)
        self._validate_type(completion_window, "completion_window", str, True)
        self._validate_type(metadata, "metadata", dict, False)
        self._validate_type(
            input_file_path, "input_file_path", [str, Path], False, True
        )
        self._validate_type(input_file_id, "input_file_id", str, False)

        if input_file_path is None and input_file_id is None:
            raise MissingValue("input_file_path | input_file_id")

        if input_file_id is not None and input_file_path is not None:
            raise WMLClientError("Both input_file_id and input_file_path can't be set")

        if input_file_id is None and input_file_path is not None:
            file_details = self.files.upload(input_file_path)
            input_file_id = self.files.get_id(file_details)

        payload: dict[str, Any] = {
            "input_file_id": input_file_id,
            "endpoint": endpoint,
            "completion_window": completion_window,
        }

        if metadata is not None:
            payload["metadata"] = metadata

        response = self._client.httpx_client.post(
            url=self._client._href_definitions.get_batches_href(),
            params=self._client._params(skip_for_create=True),
            headers=get_batch_inference_headers(self._client),
            json=payload,
        )

        return self._handle_response(200, "create_batch", response)

    def get_details(
        self, batch_id: str | None = None, limit: int | None = None
    ) -> dict[str, Any]:
        """Get batch details. If no `batch_id` is provided, returns details for all batches.

        :param batch_id: ID of the batch, defaults to None
        :type batch_id: str, optional

        :param limit: limit number of fetched records, defaults to None
        :type limit: int, optional

        :return: batch details
        :rtype: dict

        **Example:**

        .. code-block:: python

            # Get single batch details
            batch_details = batch_inference.get_details(batch_id)

            # Get all batches details
            all_batches = batch_inference.get_details()
            batches_list = all_batches.get("data")

        """
        self._validate_type(batch_id, "batch_id", str, False)
        self._validate_type(limit, "limit", int, False)

        if batch_id is not None:
            response = self._client.httpx_client.get(
                url=self._client._href_definitions.get_batch_href(batch_id),
                params=self._client._params(skip_for_create=True),
                headers=get_batch_inference_headers(self._client),
            )
        elif limit is None or 1 <= limit <= 100:
            params = self._client._params(skip_for_create=True)
            if limit is not None:
                params["limit"] = limit

            response = self._client.httpx_client.get(
                url=self._client._href_definitions.get_batches_href(),
                params=params,
                headers=get_batch_inference_headers(self._client),
            )
        else:
            raise InvalidValue(
                value_name="limit",
                reason=f"The given value `{limit}` must be between 1 and 100",
            )

        return self._handle_response(200, "get_batch_details", response)

    def get_status(self, batch_id: str) -> str:
        """Get batch status.

        :param batch_id: ID of the batch
        :type batch_id: str

        :return: batch status (e.g., "completed", "failed")
        :rtype: str

        **Example:**

        .. code-block:: python

            status = batch_inference.get_status(batch_id)

        """
        self._validate_type(batch_id, "batch_id", str, True)

        details = self.get_details(batch_id)
        status = details.get("status")

        if status is None:
            raise WMLClientError("Status not found in batch details")

        return status

    def cancel(self, batch_id: str) -> dict[str, Any]:
        """Cancel a batch job.

        :param batch_id: ID of the batch to cancel
        :type batch_id: str

        :return: cancellation response
        :rtype: dict

        **Example:**

        .. code-block:: python

            response = batch_inference.cancel(batch_id)

        """
        self._validate_type(batch_id, "batch_id", str, True)

        response = self._client.httpx_client.post(
            url=self._client._href_definitions.get_batch_cancel_href(batch_id),
            params=self._client._params(skip_for_create=True),
            headers=get_batch_inference_headers(self._client),
        )

        return self._handle_response(200, "stop_batch", response)

    def list(self, limit: int | None = None) -> DataFrame:
        """List batches in a table format.

        :param limit: limit number of fetched records
        :type limit: int, optional

        :return: pandas.DataFrame with listed batches
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            batch_inference.list(limit=10)

        """
        self._validate_type(limit, "limit", int, False)

        details = self.get_details(limit=limit)

        batches_list = details.get("data", [])

        values = []
        for batch_details in batches_list:
            values.append(
                [
                    batch_details.get("id", ""),
                    batch_details.get("input_file_id", ""),
                    batch_details.get("endpoint", ""),
                    batch_details.get("completion_window", ""),
                    batch_details.get("status", ""),
                    batch_details.get("output_file_id", ""),
                ]
            )

        return self._list(
            values,
            [
                "BATCH_ID",
                "INPUT_FILE_ID",
                "ENDPOINT",
                "COMPLETION_WINDOW",
                "STATUS",
                "OUTPUT_FILE_ID",
            ],
            limit=None,
            sort_by=None,
        )

    @staticmethod
    def get_id(batch_details: dict[str, Any]) -> str:
        """Get batch ID from batch details.

        :param batch_details: batch details
        :type batch_details: dict

        :return: batch ID
        :rtype: str

        **Example:**

        .. code-block:: python

            batch_id = batch_inference.get_id(batch_details)

        """
        WMLResource._validate_type(batch_details, "batch_details", dict, True)

        return WMLResource._get_required_element_from_dict(
            batch_details, "batch_details", ["id"], str
        )

    @staticmethod
    def get_output_file_id(batch_details: dict[str, Any]) -> str | None:
        """Get output file ID from batch details.

        :param batch_details: batch details dictionary
        :type batch_details: dict

        :return: output file ID
        :rtype: str | None

        **Example:**

        .. code-block:: python

            batch_details = batch_inference.get_details(batch_id)
            output_file_id = batch_inference.get_output_file_id(batch_details)

        """
        WMLResource._validate_type(batch_details, "batch_details", dict, True)

        return WMLResource._get_required_element_from_dict(
            batch_details, "batch_details", ["output_file_id"], str
        )
