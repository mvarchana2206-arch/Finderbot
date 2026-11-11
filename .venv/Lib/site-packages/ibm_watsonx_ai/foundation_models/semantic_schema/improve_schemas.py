#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from ibm_watsonx_ai.foundation_models.schema import BaseSchema, ImproveSchemasParameters
from ibm_watsonx_ai.foundation_models.semantic_schema.base_custom_schemas import (
    BaseCustomSchemas,
)
from ibm_watsonx_ai.utils.utils import inherited_docstring

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient


class ImproveSchemas(BaseCustomSchemas):
    """Handle schema improvement operations.

    This class provides methods to improve existing schemas through job-based operations.
    Schema improvement can include refining field definitions, adding metadata, optimizing
    structure, and enhancing schema quality based on data analysis.
    """

    def __init__(self, api_client: APIClient):
        BaseCustomSchemas.__init__(self, api_client)

    def run_job(
        self,
        parameters: ImproveSchemasParameters | dict,
    ) -> dict:
        """Execute a schema improvement job.

        :param parameters: improvement parameters and options
        :type parameters: ImproveSchemasParameters, dict

        :return: job details including job_id and initial status
        :rtype: dict

        :raises WMLClientError: if job creation fails
        :raises ApiRequestFailure: if API request fails

        **Example:**

        .. code-block:: python

            job_details = semantic_schema.improve.run_job(
                parameters={
                    "schema": {
                        "document_type": "Passport",
                        "document_description": "Passport document to get the schema",
                        "fields": {
                            "description": "Name",
                            "example": "name of the user",
                        },
                    }
                }
            )

        """
        self._validate_type(
            parameters, "parameters", [ImproveSchemasParameters, dict], True, True
        )

        if isinstance(parameters, BaseSchema):
            parameters = parameters.to_dict()

        payload: dict[str, Any] = {
            "parameters": parameters,
        }

        if self._client.default_space_id is not None:
            payload["space_id"] = self._client.default_space_id
        elif self._client.default_project_id is not None:
            payload["project_id"] = self._client.default_project_id

        response = self._client.httpx_client.post(
            url=self._client._href_definitions.get_text_schemas_improves_href(),
            json=payload,
            params=self._client._params(skip_for_create=True),
            headers=self._client._get_headers(),
        )
        return self._handle_response(201, "run_job", response)

    @inherited_docstring(
        BaseCustomSchemas.list_jobs, {"{handler}": "improve"}, "{handler}"
    )
    def list_jobs(self, limit: int | None = None) -> pd.DataFrame:
        return super().list_jobs(limit)

    @inherited_docstring(
        BaseCustomSchemas.get_results, {"{handler}": "improve"}, "{handler}"
    )
    def get_results(self, job_id: str) -> dict:
        return super().get_results(job_id)

    @inherited_docstring(
        BaseCustomSchemas.get_status, {"{handler}": "improve"}, "{handler}"
    )
    def get_status(self, job_id: str) -> str:
        return super().get_status(job_id)

    @inherited_docstring(
        BaseCustomSchemas.get_job_details, {"{handler}": "improve"}, "{handler}"
    )
    def get_job_details(
        self, job_id: str | None = None, limit: int | None = None
    ) -> dict:
        return super().get_job_details(job_id, limit)

    @inherited_docstring(
        BaseCustomSchemas.delete_job, {"{handler}": "improve"}, "{handler}"
    )
    def delete_job(self, job_id: str) -> Literal["SUCCESS"]:
        return super().delete_job(job_id)

    @classmethod
    @inherited_docstring(
        BaseCustomSchemas.get_job_id, {"{handler}": "improve"}, "{handler}"
    )
    def get_job_id(cls, job_details: dict) -> str:
        return super().get_job_id(job_details)

    def _get_single_job_url(self, job_id: str) -> str:
        """Get the URL for a single improvement job operation."""
        return self._client._href_definitions.get_text_schemas_improve_href(job_id)

    def _get_all_jobs_url(self) -> str:
        """Get the URL for listing all improvement jobs."""
        return self._client._href_definitions.get_text_schemas_improves_href()

    def _get_job_id_column_name(self) -> str:
        """Get the column name for job ID in the DataFrame."""
        return "IMPROVE_SCHEMA_JOB_ID"
