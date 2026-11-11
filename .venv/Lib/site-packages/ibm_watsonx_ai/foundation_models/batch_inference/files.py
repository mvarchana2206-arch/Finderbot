#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ibm_watsonx_ai.wml_client_error import (
    ApiRequestFailure,
    InvalidValue,
    WMLClientError,
)
from ibm_watsonx_ai.wml_resource import WMLResource

from .utils import get_batch_inference_headers

if TYPE_CHECKING:
    from pandas import DataFrame

    from ibm_watsonx_ai import APIClient


class Files(WMLResource):
    """Initialize batch files.

    :param api_client: APIClient instance with a set project ID or space ID and authenticated with `api_key`
    :type api_client: APIClient

    """

    def __init__(self, api_client: APIClient) -> None:
        WMLResource.__init__(self, __name__, api_client)

    def upload(self, file_path: str | Path) -> dict[str, Any]:
        """Upload a batch file.

        :param file_path: path to the file to upload
        :type file_path: str | Path

        :return: details of the uploaded file
        :rtype: dict

        **Example:**

        .. code-block:: python

            file_details = batch_inference.files.upload(
                file_path="/path/to/batch_file.jsonl"
            )

        """
        self._validate_type(file_path, "file_path", [str, Path], True, True)

        if isinstance(file_path, str):
            file_path = Path(file_path)

        if not file_path.exists():
            raise WMLClientError(f"File not found: {file_path}")

        filename = file_path.name

        with file_path.open("rb") as file:
            files = {"file": (filename, file, "application/octet-stream")}
            data = {"purpose": "batch"}

            headers = get_batch_inference_headers(self._client, content_type=None)
            params = self._client._params(skip_for_create=True)

            response = self._client.httpx_client.post(
                url=self._client.service_instance._href_definitions.get_files_href(),
                params=params,
                headers=headers,
                files=files,
                data=data,
            )

        return self._handle_response(200, "upload_file", response)

    def download(self, file_id: str, filename: str | Path | None = None) -> str:
        """Download a batch output file.

        :param file_id: ID of the file to download
        :type file_id: str

        :param filename: name for the downloaded file
        :type filename: str | Path, optional

        :return: path to the downloaded file
        :rtype: str

        **Example:**

        .. code-block:: python

            file_path = batch_inference.files.download(
                file_id, filename="output_file.jsonl"
            )

        """
        self._validate_type(file_id, "file_id", str, True)
        self._validate_type(filename, "filename", [str, Path], False, True)

        # Retrieve file
        response = self._client.httpx_client.get(
            url=self._client.service_instance._href_definitions.get_file_content_href(
                file_id
            ),
            params=self._client._params(skip_for_create=True),
            headers=get_batch_inference_headers(self._client),
        )

        # Validate response
        if response.status_code != 200:
            raise ApiRequestFailure(
                f"Failure during downloading batch file with id '{file_id}'.",
                response,
            )

        content = response.content

        # Determine output filename
        if filename is None:
            file_details = self.get_details(file_id)
            filename = file_details.get("filename", f"{file_id}.jsonl")

        if isinstance(filename, str):
            filename = Path(filename)

        # Write to file
        try:
            filename.write_bytes(content)
            print(f"File downloaded successfully to: {str(filename)}")
        except IOError as e:
            raise WMLClientError(
                f"Saving batch file with id '{file_id}' failed.",
                str(e),
            )

        return str(filename)

    def get_details(
        self,
        file_id: str | None = None,
        limit: int | None = None,
        order: Literal["desc", "asc"] = "desc",
        purpose: Literal["batch", "batch_output"] | None = None,
    ) -> dict[str, Any]:
        """Get file details. If no `file_id` is provided, returns details for all files.

        :param file_id: ID of the file
        :type file_id: str, optional

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param order: sort order by creation timestamp, defaults to "desc" (descending)
        :type order: Literal["desc", "asc"], optional

        :param purpose: only return files with the specified purpose
        :type purpose: Literal["batch", "batch_output"], optional

        :return: file details
        :rtype: dict

        **Example:**

        .. code-block:: python

            # Get single file details
            file_details = batch_inference.files.get_details(file_id)

            # Get all files details
            all_files = batch_inference.files.get_details()

        """
        self._validate_type(file_id, "file_id", str, False)
        self._validate_type(limit, "limit", int, False)
        self._validate_type(order, "order", str, False)
        self._validate_type(purpose, "purpose", str, False)

        if file_id is not None:
            response = self._client.httpx_client.get(
                url=self._client.service_instance._href_definitions.get_file_href(
                    file_id
                ),
                params=self._client._params(skip_for_create=True),
                headers=get_batch_inference_headers(self._client),
            )
        elif limit is None or 1 <= limit <= 10_000:
            params = self._client._params(skip_for_create=True)
            params["order"] = order
            if limit is not None:
                params["limit"] = limit
            if purpose is not None:
                params["purpose"] = purpose

            response = self._client.httpx_client.get(
                url=self._client.service_instance._href_definitions.get_files_href(),
                params=params,
                headers=get_batch_inference_headers(self._client),
            )
        else:
            raise InvalidValue(
                value_name="limit",
                reason=f"The given value `{limit}` must be between 1 and 10000",
            )

        return self._handle_response(200, "get_file_details", response)

    def list(
        self,
        limit: int | None = None,
        order: Literal["desc", "asc"] = "desc",
        purpose: Literal["batch", "batch_output"] | None = None,
    ) -> DataFrame:
        """List files in a table format.

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param order: sort order by creation timestamp, defaults to "desc" (descending)
        :type order: Literal["desc", "asc"], optional

        :param purpose: only return files with the specified purpose
        :type purpose: Literal["batch", "batch_output"], optional

        :return: pandas.DataFrame with files
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            batch_inference.files.list(limit=10)

        """
        self._validate_type(limit, "limit", int, False)
        self._validate_type(order, "order", str, False)
        self._validate_type(purpose, "purpose", str, False)

        details = self.get_details(limit=limit, order=order, purpose=purpose)

        files_list = details.get("data", [])

        values = []
        for file_info in files_list:
            values.append(
                [
                    file_info.get("id", ""),
                    file_info.get("filename", ""),
                    file_info.get("purpose", ""),
                    file_info.get("bytes", ""),
                ]
            )

        return self._list(
            values, ["FILE_ID", "FILENAME", "PURPOSE", "SIZE"], limit=None
        )

    @staticmethod
    def get_id(file_details: dict[str, Any]) -> str:
        """Get file ID from file details.

        :param file_details: file details
        :type file_details: dict

        :return: file ID
        :rtype: str

        **Example:**

        .. code-block:: python

            file_id = batch_inference.files.get_id(file_details)

        """
        WMLResource._validate_type(file_details, "file_details", dict, True)

        return WMLResource._get_required_element_from_dict(
            file_details, "file_details", ["id"], str
        )
