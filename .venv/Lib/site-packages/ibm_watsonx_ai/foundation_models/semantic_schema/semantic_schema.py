#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models.semantic_schema.cluster_schemas import (
    ClusterSchemas,
)
from ibm_watsonx_ai.foundation_models.semantic_schema.create_schemas import (
    CreateSchemas,
)
from ibm_watsonx_ai.foundation_models.semantic_schema.improve_schemas import (
    ImproveSchemas,
)
from ibm_watsonx_ai.foundation_models.semantic_schema.merge_schemas import MergeSchemas
from ibm_watsonx_ai.messages.messages import Messages
from ibm_watsonx_ai.wml_client_error import (
    InvalidMultipleArguments,
    WMLClientError,
)
from ibm_watsonx_ai.wml_resource import WMLResource


class SemanticSchema(WMLResource):
    """Manage semantic schema operations for watsonx.ai.

    This class provides a unified interface for managing semantic schemas through
    specialized operation handlers. Each operation type is accessible through a
    dedicated attribute that provides specific functionality.

    :param credentials: credentials for the watsonx.ai instance
    :type credentials: Credentials or dict, optional

    :param project_id: ID of the project
    :type project_id: str, optional

    :param space_id: ID of the space
    :type space_id: str, optional

    :param api_client: initialized APIClient object with a set project ID or space ID
    :type api_client: APIClient, optional

    **Attributes:**

    :ivar create: Handler for schema creation operations from documents
    :vartype create: CreateSchemas

    :ivar improve: Handler for schema improvement operations
    :vartype improve: ImproveSchemas

    :ivar merge: Handler for schema merging operations
    :vartype merge: MergeSchemas

    :ivar cluster: Handler for schema clustering operations
    :vartype cluster: ClusterSchemas

    .. note::
        * You must provide one of: ['credentials', 'api_client']
        * When 'credentials' is passed, you must provide one of: ['project_id', 'space_id']

    **Example:**

    .. code-block:: python

        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models.semantic_schema import (
            SemanticSchema,
        )

        semantic_schema = SemanticSchema(
            credentials=Credentials(
                url="https://us-south.ml.cloud.ibm.com",
                api_key="your_api_key",
            ),
            project_id="your_project_id",
        )

    """

    def __init__(
        self,
        *,
        credentials: Credentials | None = None,
        project_id: str | None = None,
        space_id: str | None = None,
        api_client: APIClient | None = None,
    ) -> None:
        if api_client is None:
            if credentials is None:
                raise InvalidMultipleArguments(
                    params_names_list=["credentials", "api_client"],
                    reason="None of the arguments were provided.",
                )
            api_client = APIClient(credentials)

        WMLResource.__init__(self, __name__, api_client)

        if space_id is not None:
            self._client.set.default_space(space_id)
        elif project_id is not None:
            self._client.set.default_project(project_id)
        elif not api_client:
            raise InvalidMultipleArguments(
                params_names_list=["space_id", "project_id"],
                reason="None of the arguments were provided.",
            )

        if self._client.ICP_PLATFORM_SPACES and self._client.CPD_version < 5.4:
            raise WMLClientError(
                Messages.get_message(">= 5.4", message_id="invalid_cpd_version")
            )

        self.create: CreateSchemas = CreateSchemas(self._client)
        self.improve: ImproveSchemas = ImproveSchemas(self._client)
        self.merge: MergeSchemas = MergeSchemas(self._client)
        self.cluster: ClusterSchemas = ClusterSchemas(self._client)
