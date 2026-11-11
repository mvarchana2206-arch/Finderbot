#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from ibm_watsonx_ai.foundation_models.schema import BaseSchema, CreateSchemasParameters
from ibm_watsonx_ai.foundation_models.semantic_schema.base_custom_schemas import (
    BaseCustomSchemas,
)
from ibm_watsonx_ai.helpers import DataConnection
from ibm_watsonx_ai.utils.utils import inherited_docstring

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient


class CreateSchemas(BaseCustomSchemas):
    """Handle schema creation operations.

    This class provides methods to create new schemas from documents through job-based
    operations. Schema creation analyzes document structure and generates appropriate
    schema definitions automatically.
    """

    def __init__(self, api_client: APIClient):
        BaseCustomSchemas.__init__(self, api_client)

    def run_job(
        self,
        document_reference: DataConnection,
        parameters: CreateSchemasParameters | dict | None = None,
    ) -> dict:
        """Execute a schema creation job from documents.

        :param document_reference: data connection reference to documents for schema creation
        :type document_reference: DataConnection

        :param parameters: schema creation parameters including schema name and options
        :type parameters: CreateSchemasParameters, dict, optional

        :return: job details including job_id and initial status
        :rtype: dict

        :raises WMLClientError: if job creation fails
        :raises ApiRequestFailure: if API request fails

        **Example:**

        .. code-block:: python

            from ibm_watsonx_ai.helpers import DataConnection, ContainerLocation

            document_reference = DataConnection(
                location=ContainerLocation(path="files/document.pdf")
            )

            job_details = semantic_schema.create.run_job(
                document_reference=document_reference,
                parameters={
                    "mode": "high_quality",
                    "ocr_mode": "enabled",
                    "enable_grounding": False,
                    "auto_rotation_correction": False,
                    "languages": ["en", "latn"],
                },
            )

        """
        self._validate_type(
            document_reference, "document_reference", DataConnection, True
        )
        self._validate_type(
            parameters, "parameters", [CreateSchemasParameters, dict], False, True
        )

        if isinstance(parameters, BaseSchema):
            parameters = parameters.to_dict()

        payload: dict[str, Any] = {
            "document_reference": document_reference.to_dict(),
            "parameters": parameters,
        }

        if self._client.default_space_id is not None:
            payload["space_id"] = self._client.default_space_id
        elif self._client.default_project_id is not None:
            payload["project_id"] = self._client.default_project_id

        response = self._client.httpx_client.post(
            url=self._client._href_definitions.get_text_schemas_creates_href(),
            json=payload,
            params=self._client._params(skip_for_create=True),
            headers=self._client._get_headers(),
        )
        return self._handle_response(201, "run_job", response)

    @inherited_docstring(
        BaseCustomSchemas.list_jobs, {"{handler}": "create"}, "{handler}"
    )
    def list_jobs(self, limit: int | None = None) -> pd.DataFrame:
        return super().list_jobs(limit)

    @inherited_docstring(
        BaseCustomSchemas.get_results, {"{handler}": "create"}, "{handler}"
    )
    def get_results(self, job_id: str) -> dict:
        return super().get_results(job_id)

    @inherited_docstring(
        BaseCustomSchemas.get_status, {"{handler}": "create"}, "{handler}"
    )
    def get_status(self, job_id: str) -> str:
        return super().get_status(job_id)

    @inherited_docstring(
        BaseCustomSchemas.get_job_details, {"{handler}": "create"}, "{handler}"
    )
    def get_job_details(
        self, job_id: str | None = None, limit: int | None = None
    ) -> dict:
        return super().get_job_details(job_id, limit)

    @inherited_docstring(
        BaseCustomSchemas.delete_job, {"{handler}": "create"}, "{handler}"
    )
    def delete_job(self, job_id: str) -> Literal["SUCCESS"]:
        return super().delete_job(job_id)

    @classmethod
    @inherited_docstring(
        BaseCustomSchemas.get_job_id, {"{handler}": "create"}, "{handler}"
    )
    def get_job_id(cls, job_details: dict) -> str:
        return super().get_job_id(job_details)

    def _get_single_job_url(self, job_id: str) -> str:
        """Get the URL for a single creation job operation."""
        return self._client._href_definitions.get_text_schemas_create_href(job_id)

    def _get_all_jobs_url(self) -> str:
        """Get the URL for listing all creation jobs."""
        return self._client._href_definitions.get_text_schemas_creates_href()

    def _get_job_id_column_name(self) -> str:
        """Get the column name for job ID in the DataFrame."""
        return "CREATE_SCHEMA_JOB_ID"
