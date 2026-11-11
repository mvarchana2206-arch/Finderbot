#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025-2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from typing import Any, Literal

import pandas

from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.wml_client_error import WMLClientError
from ibm_watsonx_ai.wml_resource import WMLResource


class VectorIndexes(WMLResource):
    """Initiate the Vector Indexes class.

    :param api_client: instance of APIClient with default project_id set
    :type api_client: APIClient

    **Example:**

        .. code-block:: python

            vector_indexes = VectorIndexes(api_client=api_client)

    """

    def __init__(self, api_client: APIClient) -> None:
        if api_client.default_project_id is None:
            raise WMLClientError(
                error_msg=(
                    "Vector Index Assets are available only in project scope. "
                    "Please set project_id: `api_client.set.default_project('<project_id>')`"
                )
            )

        WMLResource.__init__(self, __name__, api_client)

        if not self._client.CLOUD_PLATFORM_SPACES and self._client.CPD_version < 5.1:
            raise WMLClientError(error_msg="Operation is unsupported for this release.")

    def create(
        self,
        name: str,
        description: str | None = None,
        store: dict | None = None,
        settings: dict | None = None,
        tags: list[str] | None = None,
        sample_questions: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """Create a new Vector Index Asset.

        :param name: name for vector index asset
        :type name: str

        :param description: optional description for the vector index asset, defaults to None
        :type description: str | None

        :param store: store parameters, defaults to None
        :type store: dict | None, optional

        :param settings: settings of vector index, defaults to None
        :type settings: dict | None, optional

        :param tags: tags attached to the asset, defaults to None
        :type tags: list[str] | None, optional

        :param sample_questions: sample asked questions, defaults to None
        :type sample_questions: list[str] | None, optional

        :param data_assets: IDs of the associated data assets used in the vector index, defaults to None
        :type data_assets: list[str] | None, optional

        :param build: the associated build to process the data for external vector stores, defaults to None
        :type build: dict | None, optional

        :param status: the status of the vector index, defaults to None
        :type status: str | None, optional

        :return: metadata of the created Vector Index Asset
        :rtype: dict

        **Example:**

        .. code-block:: python

            params = dict(
                name="test_sdk_vector_index",
                description="Description",
                settings={
                    "embedding_model_id": "<model_id>",
                    "top_k": 1,
                    "schema_fields": {"text": "text"},
                },
                store={
                    "type": "watsonx.data",
                    "connection_id": "<connection_id>",
                    "index": "<index_name>",
                    "new_index": False,
                    "database": "default",
                },
                tags=["test_tag"],
                sample_questions=["Sample question"],
                status="ready",
            )

            vector_index_details = vector_indexes.create(**params)

        """

        WMLResource._validate_type(name, "name", str)
        WMLResource._validate_type(description, "description", str, False)
        WMLResource._validate_type(store, "store", [dict, str], False)
        WMLResource._validate_type(settings, "settings", dict, False)

        data_assets = kwargs.pop("data_assets", None)
        WMLResource._validate_type(data_assets, "data_assets", list, False)

        build = kwargs.pop("build", None)
        WMLResource._validate_type(build, "build", dict, False)

        status = kwargs.pop("status", None)
        WMLResource._validate_type(status, "status", str, False)

        payload = self._drop_nones_from_payload(
            name=name,
            description=description,
            data_assets=data_assets,
            store=store,
            settings=settings,
            build=build,
            tags=tags,
            sample_questions=sample_questions,
            status=status,
        )

        response = self._client.httpx_client.post(
            url=self._client._href_definitions.get_vector_indexes_href(),
            json=payload,
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        return self._handle_response(201, "create asset", response)

    def create_transactional(
        self,
        name: str,
        *,
        description: str | None = None,
        store: str | dict[str, Any] | None = None,
        data_assets: list[str] | None = None,
        settings: dict[str, Any] | None = None,
        sample_questions: list[str] | None = None,
    ) -> dict:
        """Create a new Vector Index asset using transactional API.

        :param name: name of the vector index asset
        :type name: str

        :param description: description of the vector index asset, defaults to None
        :type description: str, optional

        :param store: vector store settings
        :type store: str | dict, optional

        :param data_assets: IDs of data assets used in the vector index, defaults to None
        :type data_assets: list[str], optional

        :param settings: settings of vector index, defaults to None
        :type settings: dict, optional

        :param sample_questions: sample asked questions, defaults to None
        :type sample_questions: list[str], optional

        :return: metadata of the created Vector Index asset
        :rtype: dict

        **Example:**

        .. code-block:: python

            vector_index_details = vector_indexes.create_transactional(
                "test_sdk_vector_index",
                description="Description",
                store={
                    "type": "watsonx.data",
                    "connection_id": "<connection_id>",
                    "index": "<index_name>",
                    "new_index": False,
                    "database": "default",
                },
                data_assets=[
                    "713259df-0540-4301-982e-693a81da462c",
                    "713259df-0540-4301-982e-693a81da573d",
                ],
                settings={
                    "embedding_model_id": "<model_id>",
                    "top_k": 1,
                    "schema_fields": {"text": "text"},
                },
                sample_questions=["summarize the document, name of the author"],
            )

        """

        self._validate_type(name, "name", str)
        self._validate_type(description, "description", str, False)
        self._validate_type(store, "store", [dict, str], False)
        self._validate_type(data_assets, "data_assets", list, False)
        self._validate_type(settings, "settings", dict, False)
        self._validate_type(sample_questions, "sample_questions", list, False)

        payload = self._drop_nones_from_payload(
            name=name,
            description=description,
            store=store,
            data_assets=data_assets,
            settings=settings,
            sample_questions=sample_questions,
        )

        response = self._client.httpx_client.post(
            self._client._href_definitions.get_transactional_vector_indexes_href(),
            json=payload,
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        return self._handle_response(201, "create asset", response)

    def update(
        self,
        index_id: str,
        name: str | None = None,
        description: str | None = None,
        store: dict | None = None,
        settings: dict | None = None,
        tags: list[str] | None = None,
        sample_questions: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """Update a Vector Index Asset with the given id.

        :param index_id: ID of the vector index asset
        :type index_id: str

        :param name: name for vector index asset, defaults to None
        :type name: str | None

        :param description: optional description for the vector index asset, defaults to None
        :type description: str | None

        :param store: store parameters, defaults to None
        :type store: dict | None, optional

        :param settings: settings of vector index, defaults to None
        :type settings: dict | None, optional

        :param tags: tags attached to the asset, defaults to None
        :type tags: list[str] | None, optional

        :param sample_questions: sample asked questions, defaults to None
        :type sample_questions: list[str] | None, optional

        :param data_assets: IDs of the associated data assets used in the vector index, defaults to None
        :type data_assets: list[str] | None, optional

        :param build: the associated build to process the data for external vector stores, defaults to None
        :type build: dict | None, optional

        :param status: the status of the vector index, defaults to None
        :type status: str | None, optional

        :return: metadata of the updated Vector Index Asset
        :rtype: dict

        **Example:**

        .. code-block:: python

            vector_indexes.update(
                index_id, name="new_name", description="new_description"
            )

        """

        WMLResource._validate_type(name, "name", str, False)
        WMLResource._validate_type(description, "description", str, False)
        WMLResource._validate_type(store, "store", dict, False)
        WMLResource._validate_type(settings, "settings", dict, False)

        data_assets = kwargs.pop("data_assets", None)
        WMLResource._validate_type(data_assets, "data_assets", list, False)

        build = kwargs.pop("build", None)
        WMLResource._validate_type(build, "build", dict, False)

        status = kwargs.pop("status", None)
        WMLResource._validate_type(status, "status", str, False)

        payload = self._drop_nones_from_payload(
            name=name,
            description=description,
            data_assets=data_assets,
            store=store,
            settings=settings,
            build=build,
            tags=tags,
            sample_questions=sample_questions,
            status=status,
        )

        response = self._client.httpx_client.patch(
            url=self._client._href_definitions.get_vector_index_href(index_id),
            json=payload,
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        return self._handle_response(200, "update asset", response)

    def update_transactional(
        self,
        vector_index_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        store: str | dict[str, Any] | None = None,
        data_assets: list[str] | None = None,
        settings: dict[str, Any] | None = None,
        sample_questions: list[str] | None = None,
        build: dict[str, Any] | None = None,
        status: str | None = None,
    ) -> dict:
        """Update a Vector Index asset with the given ID using transactional API.

        :param vector_index_id: ID of the vector index asset
        :type vector_index_id: str

        :param name: name of the vector index asset
        :type name: str, optional

        :param description: description of the vector index asset, defaults to None
        :type description: str, optional

        :param store: vector store settings
        :type store: str | dict, optional

        :param data_assets: IDs of data assets used in the vector index, defaults to None
        :type data_assets: list[str], optional

        :param settings: settings of vector index, defaults to None
        :type settings: dict, optional

        :param sample_questions: sample asked questions, defaults to None
        :type sample_questions: list[str], optional

        :param build: the associated build to process the data for external vector stores, defaults to None
        :type build: dict, optional

        :param status: the status of the vector index, defaults to None
        :type status: str, optional

        :return: metadata of the updated Vector Index asset
        :rtype: dict

        **Example:**

        .. code-block:: python

            vector_index_details = vector_indexes.update_transactional(
                vector_index_id,
                name="new_vector_index_name",
                description="new_vector_index_description",
            )

        """

        self._validate_type(name, "name", str, False)
        self._validate_type(description, "description", str, False)
        self._validate_type(store, "store", [dict, str], False)
        self._validate_type(data_assets, "data_assets", list, False)
        self._validate_type(settings, "settings", dict, False)
        self._validate_type(sample_questions, "sample_questions", list, False)
        self._validate_type(build, "build", dict, False)
        self._validate_type(status, "status", str, False)

        payload = self._drop_nones_from_payload(
            name=name,
            description=description,
            store=store,
            data_assets=data_assets,
            settings=settings,
            sample_questions=sample_questions,
            build=build,
            status=status,
        )

        response = self._client.httpx_client.patch(
            self._client._href_definitions.get_transactional_vector_index_href(
                vector_index_id
            ),
            json=payload,
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        return self._handle_response(200, "update asset", response)

    def get_details(self, index_id: str) -> dict:
        """Get details of Vector Index Asset with the given ID.

        :param index_id: ID of the vector index asset
        :type index_id: str

        :return: details of the Vector Index Asset with the given ID
        :rtype: dict

        **Example:**

        .. code-block:: python

            vector_indexes.get_details(index_id)

        """
        response = self._client.httpx_client.get(
            url=self._client._href_definitions.get_vector_index_href(index_id),
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        return self._handle_response(200, "get asset", response)

    def get_details_transactional(self, vector_index_id: str) -> dict:
        """Get details of Vector Index asset with the given ID using transactional API.

        :param vector_index_id: ID of the vector index asset
        :type vector_index_id: str

        :return: details of vector index asset with the given ID
        :rtype: dict

        **Example:**

        .. code-block:: python

            vector_indexes.get_details_transactional(vector_index_id)

        """

        response = self._client.httpx_client.get(
            url=self._client._href_definitions.get_transactional_vector_index_status_href(
                vector_index_id
            ),
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        return self._handle_response(200, "get asset", response)

    def delete(self, index_id: str) -> Literal["SUCCESS"]:
        """Delete a vector index with the given id.

        :param index_id: Vector Index id
        :type index_id: str

        :return: "SUCCESS" if deletion succeeded
        :rtype: str
        :raises WMLClientError: if deletion failed

        **Example:**

        .. code-block:: python

            vector_indexes.delete(vector_index_id)

        """

        response = self._client.httpx_client.delete(
            url=self._client._href_definitions.get_vector_index_href(index_id),
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        return self._handle_response(204, "delete", response)  # type: ignore[return-value]

    def _get_details(self, limit: int | None = None) -> list:
        """Get details of all vector indexes. If limit is set to None,
        then all vector indexes are fetched.

        :param limit: limit number of fetched records, defaults to None.
        :type limit: int | None

        :return: List of vector indexes metadata
        :rtype: list
        """
        headers = self._client._get_headers()
        url = self._client._href_definitions.get_vector_indexes_all_href()
        json_data: dict[str, int | str] = {
            "query": "asset.asset_type:vector_index",
            "sort": "-asset.created_at<string>",
        }
        if limit is not None:
            if limit < 1:
                raise WMLClientError("Limit cannot be lower than 1.")
            elif limit > 200:
                raise WMLClientError("Limit cannot be larger than 200.")

            json_data.update({"limit": limit})
        else:
            json_data.update({"limit": 200})
        vector_indexes_list = []
        bookmark = True
        while bookmark is not None:
            response = self._client.httpx_client.post(
                url=url, json=json_data, headers=headers, params=self._client._params()
            )

            details_json = self._handle_response(200, "Get next details", response)
            bookmark = details_json.get("next", {"href": None}).get("bookmark", None)
            vector_indexes_list.extend(details_json.get("results", []))
            if limit is not None:
                break
            json_data.update({"bookmark": bookmark})
        return vector_indexes_list

    def list(self, *, limit: int | None = None) -> pandas.DataFrame:
        """List all available Vector Index assets in the ``DataFrame`` format.

        :param limit: limit number of fetched records, defaults to None
        :type limit: int, optional

        :return: ``DataFrame`` of fundamental properties of available Vector Index assets
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            # list of 5 recently created vector index assets
            df = vector_indexes.list(limit=5)

        """

        details = [
            "metadata.asset_id",
            "metadata.name",
            "metadata.created_at",
            "metadata.usage.last_updated_at",
        ]
        prompts_details = self._get_details(limit=limit)

        data_normalize = pandas.json_normalize(prompts_details)
        prompts_data = data_normalize.reindex(columns=details)

        df_details = pandas.DataFrame(prompts_data, columns=details)

        df_details.rename(
            columns={
                "metadata.asset_id": "ID",
                "metadata.name": "NAME",
                "metadata.created_at": "CREATED",
                "metadata.usage.last_updated_at": "LAST MODIFIED",
            },
            inplace=True,
        )

        return df_details
