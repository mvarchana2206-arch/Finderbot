#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025-2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
import copy
import json
import warnings
from typing import Any, cast

from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watsonx_ai.utils.utils import get_from_json
from ibm_watsonx_ai.wml_client_error import (
    ApiRequestFailure,
    MissingToolRequiredProperties,
    ResourceByNameNotFound,
    WMLClientError,
)
from ibm_watsonx_ai.wml_resource import WMLResource


class Tool(WMLResource):
    """Instantiate the utility agent tool.

    :param api_client: initialized APIClient object
    :type api_client: APIClient

    :param name: name of the tool
    :type name: str

    :param description: description of what the tool is used for
    :type description: str

    :param agent_description: the precise instruction to agent LLMs and should be treated as part of the system prompt, if not provided, `description` can be used in its place
    :type agent_description: str, optional

    :param input_schema: schema of the input that is provided when running the tool if applicable
    :type input_schema: dict, optional

    :param config_schema: schema of the config that is provided when running the tool if applicable
    :type config_schema: dict, optional

    :param config: configuration options that can be passed for some tools, must match the config schema for the tool
    :type config: dict, optional

    """

    def __init__(
        self,
        api_client: APIClient,
        name: str,
        description: str,
        agent_description: str | None = None,
        input_schema: dict | None = None,
        config_schema: dict | None = None,
        config: dict | None = None,
    ):
        self._client = api_client

        Tool._validate_type(name, "name", str)
        Tool._validate_type(input_schema, "input_schema", dict, False)
        Tool._validate_type(config_schema, "config_schema", dict, False)
        Tool._validate_type(config, "config", dict, False)

        self.name = name
        self.description = description
        self.agent_description = agent_description
        self.input_schema = input_schema
        self.config_schema = config_schema
        self.config = config

        WMLResource.__init__(self, __name__, self._client)

        if self.input_schema is not None:
            self._input_schema_required: list | None = self.input_schema.get("required")

    def _validate_tool_input(self, input: str | dict) -> None:
        if self.input_schema is None:
            Tool._validate_type(input, "input", str)
        else:
            Tool._validate_type(input, "input", dict)
            if self._input_schema_required and any(
                req not in input for req in self._input_schema_required
            ):
                raise MissingToolRequiredProperties(self._input_schema_required)

    def run(
        self,
        input: str | dict,
        config: dict | None = None,
    ) -> dict:
        """Run a utility agent tool given `input`.

        :param input: input to be used when running tool
        :type input:
            - **str** - if running tool has no `input_schema`
            - **dict** - if running tool has `input_schema`

        :param config: configuration options that can be passed for some tools, must match the config schema for the tool
        :type config: dict, optional

        :return: the output from running the tool
        :rtype: dict

        **Example for the tool without input schema:**

        .. code-block:: python

            toolkit = Toolkit(api_client=api_client)
            google_search = toolkit.get_tool(tool_name="GoogleSearch")
            result = google_search.run(input="Search IBM")

        **Example for the tool with input schema:**

        .. code-block:: python

            toolkit = Toolkit(api_client=api_client)
            weather_tool = toolkit.get_tool(tool_name="Weather")
            tool_input = {"location": "New York"}
            result = weather_tool.run(input=tool_input)

        """
        self._validate_tool_input(input)

        payload = {
            "input": input,
            "tool_name": self.name,
        }

        config = config or self.config

        if config and self.config_schema:
            payload["config"] = config

        response = self._client.httpx_client.post(
            url=self._client._href_definitions.get_utility_agent_tools_run_href(),
            json=payload,
            headers=self._client._get_headers(),
        )

        return self._handle_response(200, "run tool", response)

    def __getitem__(self, key: str) -> Any:
        # For backward compatibility in Toolkit.get_tools
        try:
            return getattr(self, key)
        except AttributeError as e:
            raise KeyError(key) from e

    def get(self, key: str, default: Any = None) -> Any:
        # For backward compatibility in Toolkit.get_tools
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def __repr__(self) -> str:
        return (
            f'Tool(name="{self.name}", description="{self.description}", '
            f'agent_description="{self.agent_description}", '
            f"input_schema={self.input_schema}, "
            f"config_schema={self.config_schema}, "
            f"config={self.config}, "
            f"api_client={self._client})"
        )


class _RAGQuery(Tool):
    """Instantiate the RAGQuery tool which is SDK's implementation as a replacement of the deprecated utility agent tool.

    :param api_client: initialized APIClient object
    :type api_client: APIClient
    """

    def __init__(
        self,
        api_client: APIClient,
    ):
        super().__init__(
            api_client=api_client,
            name="RAGQuery",
            description="Search the documents in a vector index.",
            agent_description="Search information in documents to provide context to a user query. Useful when asked to ground the answer in specific knowledge about {indexName}",
            input_schema=None,
            config_schema={
                "title": "config schema for RAGQuery tool",
                "type": "object",
                "properties": {
                    "vectorIndexId": {
                        "title": "Vector index identifier",
                        "type": "string",
                    },
                    "vectorIndexIds": {
                        "title": "Vector index identifiers",
                        "type": "array",
                    },
                    "projectId": {"title": "Project identifier", "type": "string"},
                    "spaceId": {"title": "Space identifier", "type": "string"},
                },
                "allOf": [
                    {
                        "oneOf": [
                            {"required": ["vectorIndexId"]},
                            {"required": ["vectorIndexIds"]},
                        ]
                    },
                    {"oneOf": [{"required": ["projectId"]}, {"required": ["spaceId"]}]},
                ],
            },
            config=None,
        )

    def run(
        self,
        input: str | dict,
        config: dict | None = None,
    ) -> dict:
        """Run the SDK's implementation of RAGQuery tool (does not delegate to the deprecated API endpoint) with given `input` and `config`.

        :param input: input to be used when running tool
        :type input: str

        :param config: configuration options, must match the config schema for the tool
        :type config: dict, optional

        :return: the output from running the tool
        :rtype: dict

        """
        self._validate_tool_input(input)
        Tool._validate_type(config, "config", dict)

        input = cast(str, input)
        config = cast(dict[str, Any], config)

        if "projectId" not in config and "spaceId" not in config:
            raise MissingToolRequiredProperties(
                'One of ["projectId", "spaceId"]', schema_type="config"
            )

        if vi_id := config.get("vectorIndexId"):
            vector_index_ids = [vi_id]
        elif vi_ids := config.get("vectorIndexIds"):
            vector_index_ids = vi_ids
        else:
            raise MissingToolRequiredProperties(
                'One of ["vectorIndexId", "vectorIndexIds"]', schema_type="config"
            )

        page_contents: list[str] = []
        for vector_index_id in vector_index_ids:
            docs = self._process_vector_index(
                vector_index_id=vector_index_id,
                query=input,
                project_id=config.get("projectId"),
                space_id=config.get("spaceId"),
            )
            page_contents.extend(docs)
        result = "\n\n".join(page_contents)

        return {"output": result}

    def _query_vector_store(
        self,
        api_client: APIClient,
        index_details: dict[str, Any],
        query: str,
        index_id: str,
    ) -> list:
        from ibm_watsonx_ai.foundation_models.extensions.rag import VectorStore

        store_type = index_details["store"].get("type")
        embedding_model_id = index_details["settings"]["embedding_model_id"]

        if distance_metric := index_details["settings"].get("distance_metric"):
            distance_metric = distance_metric.lower()

        match store_type:
            case "memory":
                import gzip

                documents: list[dict] = []
                for _, content in cast(
                    list,
                    api_client.data_assets.get_content(index_id, list_format=True),
                ):
                    decompressed_bytes = gzip.decompress(content)
                    embedded_docs: list[dict] = json.loads(decompressed_bytes)
                    documents.extend(embedded_docs)

                if distance_metric == "l2":
                    distance_metric = "euclidean"

                vector_store = VectorStore(
                    api_client=api_client,
                    embeddings=self._prepare_embeddings(api_client, embedding_model_id),
                    datasource_type="chroma",
                    distance_metric=distance_metric,
                )
                vector_store.clear()  # to avoid duplicates when called several times since chroma is process-global
                chroma_store = vector_store.get_client()
                chroma_store.add_texts(
                    texts=[doc["content"] for doc in documents],
                )

            case "elasticsearch":
                schema_fields = index_details["settings"].get("schema_fields", {})
                es_kwargs: dict[str, Any] = {}
                if query_field := schema_fields.get("text"):
                    es_kwargs["query_field"] = query_field
                if vector_query_field := schema_fields.get("vector_query"):
                    es_kwargs["vector_query_field"] = vector_query_field

                if distance_metric == "l2":
                    distance_metric = "euclidean"

                vector_store = VectorStore(
                    api_client=api_client,
                    connection_id=index_details["store"].get("connection_id"),
                    index_name=index_details["store"].get("index"),
                    distance_metric=distance_metric,
                    model_id=index_details["settings"]["embedding_model_id"],
                    **es_kwargs,
                )

            case "watsonx.data":  # Milvus
                schema_fields = index_details["settings"].get("schema_fields", {})
                milvus_kwargs: dict[str, Any] = {}
                if text_field := schema_fields.get("text"):
                    milvus_kwargs["text_field"] = text_field

                vector_store = VectorStore(
                    api_client=api_client,
                    connection_id=index_details["store"].get("connection_id"),
                    embeddings=self._prepare_embeddings(api_client, embedding_model_id),
                    index_name=index_details["store"].get("index"),
                    distance_metric=distance_metric,
                    **milvus_kwargs,
                )
            case _:
                raise WMLClientError(
                    f"Unknown vector store type ('{store_type}') for vector index id: {index_id}."
                )

        results = vector_store.search(query, index_details["settings"]["top_k"])

        if store_type == "elasticsearch":
            # reverse the order of results to maintain compatibility with the original tool
            results.reverse()

        return results

    def _extract_docs_content(self, documents: list) -> list[str]:
        if not documents:
            return []
        return [result.page_content for result in documents]

    def _rerank_results(
        self,
        api_client: APIClient,
        index_details: dict[str, Any],
        query: str,
        docs: list[str],
    ) -> list[str]:
        from ibm_watsonx_ai.foundation_models import Rerank

        default_reranking_model_id = "intfloat/multilingual-e5-large"

        reranking_model_id = index_details["settings"].get(
            "reranking_model_id", default_reranking_model_id
        )
        reranking_model_spec = cast(
            dict,
            api_client.foundation_models.get_model_specs(
                model_id=reranking_model_id, filters=None
            ),
        )
        max_sequence_length = get_from_json(
            reranking_model_spec, ["model_limits", "max_sequence_length"]
        )

        rerank_params: dict = {}
        if max_sequence_length:
            rerank_params["truncate_input_tokens"] = max_sequence_length
        if top_n := index_details["settings"].get("top_n"):
            rerank_params["return_options"] = {"top_n": top_n}

        rerank = Rerank(
            api_client=api_client,
            model_id=reranking_model_id,
            params=rerank_params,
        )

        scored_results = rerank.generate(
            query=query,
            inputs=docs,  # type: ignore[arg-type]
        )["results"]

        ranked_results = sorted(scored_results, key=lambda x: x["score"], reverse=True)
        reranked_docs = cast(list[str], [docs[r["index"]] for r in ranked_results])

        return reranked_docs

    def _process_vector_index(
        self,
        vector_index_id: str,
        query: str,
        project_id: str | None = None,
        space_id: str | None = None,
    ) -> list[str]:
        api_client = APIClient(
            credentials=self._client.credentials,
            project_id=project_id,
            space_id=space_id,
        )

        index_details = api_client.data_assets.get_details(vector_index_id)["entity"][
            "vector_index"
        ]
        query_results = self._query_vector_store(
            api_client, index_details, query, vector_index_id
        )
        docs_content = self._extract_docs_content(query_results)

        if index_details["settings"].get("rerank"):
            docs_content = self._rerank_results(
                api_client, index_details, query, docs_content
            )

        return docs_content

    def _prepare_embeddings(
        self, api_client: APIClient, embedding_model_id: str
    ) -> Embeddings:
        embedding_model_spec = cast(
            dict,
            api_client.foundation_models.get_model_specs(
                model_id=embedding_model_id, filters=None
            ),
        )

        max_sequence_length = get_from_json(
            embedding_model_spec, ["model_limits", "max_sequence_length"]
        )

        parameters: dict = {}
        if max_sequence_length:
            parameters = {"truncate_input_tokens": max_sequence_length}

        return Embeddings(
            model_id=embedding_model_id,
            params=parameters,
            api_client=api_client,
        )


class Toolkit(WMLResource):
    """Toolkit for utility agent tools.

    :param api_client: initialized APIClient object
    :type api_client: APIClient

    :param params: dict of config parameters for each tool, e.g. {"GoogleSearch": {"maxResults": 2}}
    :type params: dict[str, dict], optional

    **Example:**

    .. code-block:: python

        from ibm_watsonx_ai import APIClient, Credentials
        from ibm_watsonx_ai.foundation_models.utils import Toolkit

        credentials = Credentials(url="<url>", api_key=IAM_API_KEY)
        tools_params = {"GoogleSearch": {"maxResults": 2}}

        api_client = APIClient(credentials)
        toolkit = Toolkit(api_client=api_client, params=tools_params)

    """

    def __init__(
        self,
        api_client: APIClient,
        params: dict[str, dict] | None = None,
    ):
        self._client = api_client
        self.params = params

        self._tools: list[Tool] = self._retrieve_tools_from_api()

        WMLResource.__init__(self, __name__, self._client)

    def _retrieve_tools_from_api(self) -> list[Tool]:
        response = self._client.httpx_client.get(
            url=self._client._href_definitions.get_utility_agent_tools_href(),
            headers=self._client._get_headers(),
        )

        try:
            details = self._handle_response(
                200,
                "getting utility agent tools",
                response,
                _silent_response_logging=True,
            )
        except ApiRequestFailure:
            # Endpoint removed (August 2026, CPD 5.4)
            toolkit_removal_message = (
                "watsonx.ai Agent Lab and Utility Agent Tools are no longer supported. "
                "The replacement offering is watsonx Orchestrate Agent Lab/builder. "
                "Only the `RAGQuery` tool is available to maintain backward compatibility, "
                "but it is not recommended for new solutions."
            )

            with warnings.catch_warnings():
                warnings.simplefilter("default", category=DeprecationWarning)
                warnings.warn(toolkit_removal_message, category=DeprecationWarning)

            return [_RAGQuery(self._client)]

        toolkit_deprecation_message = (
            "watsonx.ai Agent Lab and Utility Agent Tools are being deprecated. "
            "The replacement offering is watsonx Orchestrate Agent Lab/builder."
        )
        with warnings.catch_warnings():
            warnings.simplefilter("default", category=DeprecationWarning)
            warnings.warn(toolkit_deprecation_message, category=DeprecationWarning)

        return [
            Tool(
                api_client=self._client,
                name=r["name"],
                description=r["description"],
                agent_description=r.get("agent_description"),
                input_schema=r.get("input_schema"),
                config_schema=r.get("config_schema"),
                config=(self.params or {}).get(r["name"]),
            )
            for r in details.get("resources", [])
        ]

    def get_tools(self) -> list[Tool]:
        """Get list of available utility agent tools.

        :return: list of available tools
        :rtype: list[Tool]

        **Examples**

        .. code-block:: python

            toolkit = Toolkit(api_client=api_client)
            tools = toolkit.get_tools()

        """
        return self._tools

    def get_tool(self, tool_name: str) -> Tool:
        """Get a utility agent tool with the given `tool_name`.

        :param tool_name: name of a specific tool
        :type tool_name: str

        :return: tool with a given name
        :rtype: Tool

        **Examples**

        .. code-block:: python

            toolkit = Toolkit(api_client=api_client)
            google_search = toolkit.get_tool(tool_name="GoogleSearch")

        """
        Toolkit._validate_type(tool_name, "tool_name", str)

        try:
            return next(filter(lambda el: el.name == tool_name, self._tools))
        except StopIteration:
            raise ResourceByNameNotFound(tool_name, "utility agent tool") from None


def convert_to_watsonx_tool(utility_tool: Tool) -> dict:
    """Convert utility agent tool to watsonx tool format.

    :param utility_tool: utility agent tool
    :type utility_tool: Tool

    :return: watsonx tool structure
    :rtype: dict

    **Examples**

    .. code-block:: python

        from ibm_watsonx_ai.foundation_models.utils import Toolkit

        toolkit = Toolkit(api_client)
        weather_tool = toolkit.get_tool("Weather")
        convert_to_watsonx_tool(weather_tool)

        # Return
        # {
        #     "type": "function",
        #     "function": {
        #         "name": "Weather",
        #         "description": "Find the weather for a city.",
        #         "parameters": {
        #             "type": "object",
        #             "properties": {
        #                 "location": {
        #                     "title": "location",
        #                     "description": "Name of the location",
        #                     "type": "string",
        #                 },
        #                 "country": {
        #                     "title": "country",
        #                     "description": "Name of the state or country",
        #                     "type": "string",
        #                 },
        #             },
        #             "required": ["location"],
        #         },
        #     },
        # }

    """

    def parse_parameters(input_schema: dict | None) -> dict:
        if input_schema:
            parameters = copy.deepcopy(input_schema)
        else:
            parameters = {
                "type": "object",
                "properties": {
                    "input": {
                        "description": "Input to be used when running tool.",
                        "type": "string",
                    },
                },
                "required": ["input"],
            }

        return parameters

    tool = {
        "type": "function",
        "function": {
            "name": utility_tool.name,
            "description": utility_tool.description,
            "parameters": parse_parameters(utility_tool.input_schema),
        },
    }
    return tool


def convert_to_utility_tool_call(tool_call: dict) -> dict:
    """Convert json format tool call to utility tool call format.

    :param tool_call: watsonx tool call
    :type tool_call: dict

    :return: utility tool call
    :rtype: dict

    **Examples**

    .. code-block:: python

        tool_call = {
            "id": "rcWg61ytv",
            "type": "function",
            "function": {
                "name": "GoogleSearch",
                "arguments": '{"input": "IBM"}',
            },
        }
        convert_to_utility_tool_call(tool_call)

        # Return
        # {"input": "IBM", "tool_name": "GoogleSearch"}

    """
    tool_name = tool_call["function"]["name"]
    arguments = tool_call["function"]["arguments"]
    try:
        json_arguments = json.loads(arguments)
    except json.JSONDecodeError:
        raise Exception(f"Could not parse {arguments} as json.")
    input_data = json_arguments.get("input", {}) or {
        k: v for k, v in json_arguments.items()
    }

    return {
        "tool_name": tool_name,
        "input": input_data,
    }
