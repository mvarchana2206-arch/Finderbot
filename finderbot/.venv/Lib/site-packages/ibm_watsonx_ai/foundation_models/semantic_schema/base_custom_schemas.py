#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from ibm_watsonx_ai.utils import get_from_json
from ibm_watsonx_ai.wml_client_error import InvalidValue
from ibm_watsonx_ai.wml_resource import WMLResource

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient


class BaseCustomSchemas(WMLResource, ABC):
    """Abstract base class for schema operation utilities in watsonx.ai

    This class defines the contract for schema-related operations such as creating,
    improving, merging, and clustering schemas. Subclasses must implement all abstract
    methods to provide specific schema operation functionality.

    :param api_client: APIClient instance
    :type api_client: APIClient
    """

    def __init__(self, api_client: APIClient):
        WMLResource.__init__(self, __name__, api_client)

    @abstractmethod
    def run_job(self, *args: Any, **kwargs: Any) -> dict:
        """Execute a schema operation job.

        Subclasses can define their own specific parameters.

        :return: job details
        :rtype: dict
        """
        ...

    @abstractmethod
    def _get_single_job_url(self, job_id: str) -> str:
        """Get the URL for a single job operation.

        :param job_id: ID of the job
        :type job_id: str

        :return: URL for the job
        :rtype: str
        """
        ...

    @abstractmethod
    def _get_all_jobs_url(self) -> str:
        """Get the URL for listing all jobs.

        :return: URL for listing jobs
        :rtype: str
        """
        ...

    @abstractmethod
    def _get_job_id_column_name(self) -> str:
        """Get the column name for job ID in the DataFrame.

        :return: column name for job ID
        :rtype: str
        """
        ...

    def list_jobs(self, limit: int | None = None) -> pd.DataFrame:
        """List all jobs.

        :param limit: limit number of fetched records, defaults to None
        :type limit: int, optional

        :return: DataFrame containing all jobs with their status
        :rtype: pd.DataFrame

        **Example:**

        .. code-block:: python

            jobs_list = semantic_schema.{handler}.list_jobs(limit=10)
            jobs_list

        """
        columns = ["metadata.id", "metadata.created_at", "entity.results.status"]

        details = self.get_job_details(limit=limit)
        resources = details["resources"]
        data_normalize = pd.json_normalize(resources)
        extraction_data = data_normalize.reindex(columns=columns)

        df_details = pd.DataFrame(extraction_data, columns=columns)
        df_details.rename(
            columns={
                "metadata.id": self._get_job_id_column_name(),
                "metadata.created_at": "CREATED",
                "entity.results.status": "STATUS",
            },
            inplace=True,
        )

        return df_details

    def get_results(self, job_id: str) -> dict:
        """Retrieve results of a specific job.

        :param job_id: ID of the job to retrieve results for
        :type job_id: str

        :return: job results
        :rtype: dict

        :raises WMLClientError: if job_id is invalid or job not found

        **Example:**

        .. code-block:: python

            job_id = "<{handler}_schemas_job_id>"
            results = semantic_schema.{handler}.get_results(job_id)
            results

        """
        self._validate_type(job_id, "job_id", str, True)

        job_details = self.get_job_details(job_id)

        return get_from_json(job_details, ["entity", "results"])

    def get_status(self, job_id: str) -> str:
        """Retrieve status of a specific job.

        :param job_id: ID of the job to retrieve status for
        :type job_id: str

        :return: job status
        :rtype: str

        :raises WMLClientError: if job_id is invalid or job not found

        **Example:**

        .. code-block:: python

            job_id = "<{handler}_schemas_job_id>"
            status = semantic_schema.{handler}.get_status(job_id)
            print(f"Job status: {status}")

        """
        self._validate_type(job_id, "job_id", str, True)

        job_details = self.get_job_details(job_id)

        return get_from_json(job_details, ["entity", "results", "status"])

    def get_job_details(
        self, job_id: str | None = None, limit: int | None = None
    ) -> dict:
        """Retrieve details of a specific job or all jobs.

        :param job_id: ID of the job to retrieve. If None, returns all jobs
        :type job_id: str | None

        :param limit: limit number of fetched records, defaults to None
        :type limit: int, optional

        :return: job details or list of all jobs
        :rtype: dict

        :raises WMLClientError: if job_id is provided but job not found

        **Example:**

        .. code-block:: python

            # Get details of a specific {handler} job
            job_id = "<{handler}_schemas_job_id>"
            job_details = semantic_schema.{handler}.get_job_details(job_id)

            # Get all {handler} jobs with limit
            all_jobs_details = semantic_schema.{handler}.get_job_details(limit=50)
        """
        self._validate_type(job_id, "job_id", str, False)

        if job_id is not None:
            response = self._client.httpx_client.get(
                url=self._get_single_job_url(job_id),
                params=self._client._params(skip_userfs=True),
                headers=self._client._get_headers(),
            )
        elif limit is None or 1 <= limit <= 200:
            params = self._client._params(skip_userfs=True)
            if limit is not None:
                params["limit"] = limit

            response = self._client.httpx_client.get(
                url=self._get_all_jobs_url(),
                params=params,
                headers=self._client._get_headers(),
            )
        else:
            raise InvalidValue(
                value_name="limit",
                reason=f"The given value {limit} is not in between 1 and 200",
            )

        return self._handle_response(200, "get_job_details", response)

    def delete_job(self, job_id: str) -> Literal["SUCCESS"]:
        """Delete a specific job.

        :param job_id: ID of the job to delete
        :type job_id: str

        :return: "SUCCESS" if the deletion succeeds
        :rtype: str

        :raises WMLClientError: if deletion fails or job not found

        **Example:**

        .. code-block:: python

            job_id = "<{handler}_schemas_job_id>"
            semantic_schema.{handler}.delete_job(job_id)

        """
        self._validate_type(job_id, "job_id", str, True)

        params = self._client._params(skip_userfs=True)
        params["hard_delete"] = True

        response = self._client.httpx_client.delete(
            url=self._get_single_job_url(job_id),
            params=params,
            headers=self._client._get_headers(),
        )

        return self._handle_response(204, "delete_job", response)  # type: ignore[return-value]

    @classmethod
    def get_job_id(cls, job_details: dict) -> str:
        """Extract job ID from job details dictionary.

        :param job_details: dictionary containing job details
        :type job_details: dict

        :return: job ID
        :rtype: str

        :raises WMLClientError: if job_details is invalid or job ID not found

        **Example:**

        .. code-block:: python

            job_id = semantic_schema.{handler}.get_job_id(job_details)
            print(f"Job ID: {job_id}")

        """
        cls._validate_type(job_details, "job_details", dict, True)

        return cls._get_required_element_from_dict(
            job_details, "job_details", ["metadata", "id"], str
        )
