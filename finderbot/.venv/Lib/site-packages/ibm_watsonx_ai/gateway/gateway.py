#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025-2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
import copy
import json
from collections.abc import AsyncIterator, Iterator
from typing import Any, Literal, overload

import httpx

# Model Gateway may return 201 and 202 status codes, which come directly from the external API.
# This isn't reflected in the API docs, but this happened for Red Hat AI provider.
ACCEPTED_CREATE_STATUS_CODES = frozenset({200, 201, 202})

from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai._wrappers.httpx import RateLimitedRetryDecorator, TokenBucket
from ibm_watsonx_ai.gateway.models import Models
from ibm_watsonx_ai.gateway.policies import Policies
from ibm_watsonx_ai.gateway.providers import Providers
from ibm_watsonx_ai.gateway.rate_limits import RateLimits
from ibm_watsonx_ai.gateway.tenant import Tenant
from ibm_watsonx_ai.wml_client_error import InvalidMultipleArguments, WMLClientError
from ibm_watsonx_ai.wml_resource import WMLResource

# Type aliases for gateway inputs and outputs
PromptInput = str | list[str] | list[int]
BatchedPromptInput = list[str | list[str] | list[int]]


def _streaming_create(api_client: APIClient, url: str, request_json: dict) -> Iterator:
    with api_client.httpx_client.stream(
        method="POST",
        url=url,
        json=request_json,
        headers=api_client._get_headers(include_container_id=True),
    ) as resp:
        if resp.status_code in ACCEPTED_CREATE_STATUS_CODES:
            resp_iter = resp.iter_lines()

            for chunk in resp_iter:
                field_name, _, response = chunk.partition(":")

                if response.strip() == "[DONE]":
                    break

                if field_name == "data" and response:
                    try:
                        parsed_response = json.loads(response)
                    except json.JSONDecodeError:
                        raise Exception(f"Could not parse {response} as json")
                    yield parsed_response
        else:
            resp.read()
            raise WMLClientError(
                f"Request failed with: {resp.text} ({resp.status_code})"
            )


async def _streaming_acreate(
    api_client: APIClient, url: str, request_json: dict
) -> AsyncIterator:
    async with api_client.async_httpx_client.stream(
        method="POST",
        url=url,
        json=request_json,
        headers=await api_client._aget_headers(include_container_id=True),
    ) as resp:
        if resp.status_code in ACCEPTED_CREATE_STATUS_CODES:
            resp_iter = resp.aiter_lines()

            async for chunk in resp_iter:
                field_name, _, response = chunk.partition(":")

                if response.strip() == "[DONE]":
                    break

                if field_name == "data" and response:
                    try:
                        parsed_response = json.loads(response)
                    except json.JSONDecodeError:
                        raise Exception(f"Could not parse {response} as json")
                    yield parsed_response

        else:
            await resp.aread()
            raise WMLClientError(
                f"Request failed with: ({resp.text} {resp.status_code})"
            )


class Gateway(WMLResource):
    """Model Gateway client.

    Provides access to providers, models, policies, rate limits, and inference
    endpoints (chat completions, text completions, and embeddings) registered
    in IBM watsonx.ai Model Gateway.

    :param credentials: credentials for the watsonx.ai instance;
        mutually exclusive with ``api_client``
    :type credentials: Credentials, optional

    :param verify: SSL certificate verification setting:

        * path to a CA_BUNDLE file or directory of trusted CA certificates
        * ``True`` — use the default truststore
        * ``False`` — disable verification (not recommended for production)
    :type verify: bool or str, optional

    :param api_client: pre-initialised :class:`~ibm_watsonx_ai.APIClient` with a
        project or space ID already set; if provided, ``credentials`` is not required
    :type api_client: APIClient, optional

    :param max_retries: maximum number of retries when a request fails with a
        status code in ``retry_status_codes``; defaults to ``10``
    :type max_retries: int, optional

    :param delay_time: base delay (in seconds) for the exponential back-off
        formula ``delay_time * 2 ** attempt``; defaults to ``0.5``
    :type delay_time: float, optional

    :param retry_status_codes: HTTP status codes that trigger the retry
        mechanism; defaults to ``[429, 503, 504, 520]``
    :type retry_status_codes: list[int], optional

    **Attributes:**

    :ivar providers: manages providers registered in Model Gateway
    :vartype providers: ~ibm_watsonx_ai.gateway.providers.Providers

    :ivar models: manages models registered in Model Gateway
    :vartype models: ~ibm_watsonx_ai.gateway.models.Models

    :ivar policies: manages access policies
    :vartype policies: ~ibm_watsonx_ai.gateway.policies.Policies

    :ivar rate_limits: manages rate limits
    :vartype rate_limits: ~ibm_watsonx_ai.gateway.rate_limits.RateLimits

    :ivar tenant: manages the tenant and its configuration
    :vartype tenant: ~ibm_watsonx_ai.gateway.tenant.Tenant

    :ivar chat: chat interface; use ``chat.completions.create(...)`` to run chat completions
    :vartype chat: object

    :ivar completions: text completions interface; use ``completions.create(...)``
    :vartype completions: object

    :ivar embeddings: embeddings interface; use ``embeddings.create(...)``
    :vartype embeddings: object

    :raises InvalidMultipleArguments: if neither ``credentials`` nor ``api_client``
        is provided
    :raises WMLClientError: if the connected platform release does not support
        Model Gateway (CPD < 5.2)
    """

    def __init__(
        self,
        *,
        credentials: Credentials | None = None,
        verify: bool | str | None = None,
        api_client: APIClient | None = None,
        max_retries: int | None = None,
        delay_time: float | None = None,
        retry_status_codes: list[int] | None = None,
    ):
        if credentials:
            api_client = APIClient(credentials, verify=verify)
        elif not api_client:
            raise InvalidMultipleArguments(
                params_names_list=["credentials", "api_client"],
                reason="None of the arguments were provided.",
            )

        WMLResource.__init__(self, __name__, api_client)

        if self._client.ICP_PLATFORM_SPACES and self._client.CPD_version < 5.2:
            raise WMLClientError("AI Gateway is not supported for this release.")

        self.providers = Providers(self._client)
        self.models = Models(self._client)
        self.policies = Policies(self._client)
        self.rate_limits = RateLimits(self._client)
        self.tenant = Tenant(self._client)

        _retry_decorator = RateLimitedRetryDecorator(
            api_client=self._client,
            rate_limiter=TokenBucket(rate=8, capacity=8),
            retry_status_codes=retry_status_codes,
            max_retries=max_retries,
            delay_time=delay_time,
        )
        _post_with_retry = _retry_decorator.rate_limited_retry(
            self._client.httpx_client.post
        )
        _async_post_with_retry = _retry_decorator.rate_limited_async_retry(
            self._client.async_httpx_client.post
        )

        # Chat completions
        class _ChatCompletions(WMLResource):
            def __init__(
                self,
                api_client: APIClient,
                post_with_retry: Any,
                async_post_with_retry: Any,
            ):
                WMLResource.__init__(self, __name__, api_client)
                self._post_with_retry = post_with_retry
                self._async_post_with_retry = async_post_with_retry

            @overload
            def create(
                self,
                model: str,
                messages: list[dict],
                *,
                stream: Literal[False] = False,
                **kwargs: Any,
            ) -> dict: ...

            @overload
            def create(
                self,
                model: str,
                messages: list[dict],
                *,
                stream: Literal[True],
                **kwargs: Any,
            ) -> Iterator: ...

            def create(
                self,
                model: str,
                messages: list[dict],
                *,
                stream: bool = False,
                **kwargs: Any,
            ) -> dict | Iterator | httpx.Response:
                """Generate chat completions for given model and messages.

                :param model: name of model for given provider or alias
                :type model: str

                :param messages: messages to be processed during call
                :type messages: list[dict]

                :param stream: if True will stream the response, defaults to False
                :type stream: bool, optional

                :returns: model answer
                :rtype: dict | Iterator
                """
                request_json = {"messages": messages, "model": model, **kwargs}
                if stream:
                    request_json["stream"] = True

                url = self._client._href_definitions.get_gateway_chat_completions_href()

                if stream:
                    return _streaming_create(
                        api_client=self._client, url=url, request_json=request_json
                    )

                response = self._post_with_retry(
                    url=url,
                    headers=self._client._get_headers(include_container_id=True),
                    json=request_json,
                )

                return self._handle_response(
                    ACCEPTED_CREATE_STATUS_CODES,
                    "chat completion creation",
                    response,
                )

            @overload
            async def acreate(
                self,
                model: str,
                messages: list[dict],
                *,
                stream: Literal[False] = False,
                **kwargs: Any,
            ) -> dict: ...

            @overload
            async def acreate(
                self,
                model: str,
                messages: list[dict],
                *,
                stream: Literal[True],
                **kwargs: Any,
            ) -> AsyncIterator: ...

            async def acreate(
                self,
                model: str,
                messages: list[dict],
                *,
                stream: bool = False,
                **kwargs: Any,
            ) -> dict | AsyncIterator | httpx.Response:
                """Generate chat completions for given model and messages asynchronously.

                :param model: name of model for given provider or alias
                :type model: str

                :param messages: messages to be processed during call
                :type messages: list[dict]

                :param stream: if True will stream the response, defaults to False
                :type stream: bool, optional

                :returns: model answer
                :rtype: dict | AsyncIterator
                """
                request_json = {"messages": messages, "model": model, **kwargs}
                if stream:
                    request_json["stream"] = True

                url = self._client._href_definitions.get_gateway_chat_completions_href()

                if stream:
                    return _streaming_acreate(
                        api_client=self._client, url=url, request_json=request_json
                    )

                response = await self._async_post_with_retry(
                    url=url,
                    headers=await self._client._aget_headers(include_container_id=True),
                    json=request_json,
                )

                return self._handle_response(
                    ACCEPTED_CREATE_STATUS_CODES,
                    "chat completion creation",
                    response,
                )

        class _Chat:
            def __init__(
                self,
                api_client: APIClient,
                post_with_retry: Any,
                async_post_with_retry: Any,
            ):
                self.completions = _ChatCompletions(
                    api_client, post_with_retry, async_post_with_retry
                )

        self.chat = _Chat(self._client, _post_with_retry, _async_post_with_retry)

        # Text completions
        class _Completions(WMLResource):
            def __init__(
                self,
                api_client: APIClient,
                post_with_retry: Any,
                async_post_with_retry: Any,
            ):
                WMLResource.__init__(self, __name__, api_client)
                self._post_with_retry = post_with_retry
                self._async_post_with_retry = async_post_with_retry

            @overload
            def create(
                self,
                model: str,
                prompt: PromptInput,
                *,
                stream: Literal[False] = False,
                **kwargs: Any,
            ) -> dict: ...

            @overload
            def create(
                self,
                model: str,
                prompt: PromptInput,
                *,
                stream: Literal[True],
                **kwargs: Any,
            ) -> Iterator: ...

            def create(
                self,
                model: str,
                prompt: PromptInput,
                *,
                stream: bool = False,
                **kwargs: Any,
            ) -> dict | Iterator:
                """Generate text completions for given model and prompt.

                :param model: name of model for given provider or alias
                :type model: str

                :param prompt: prompt for processing
                :type prompt: str or list[str] or list[int]

                :param stream: if True will stream the response, defaults to False
                :type stream: bool, optional

                :returns: model answer
                :rtype: dict | Iterator
                """
                request_json = {"prompt": prompt, "model": model, **kwargs}
                if stream:
                    request_json["stream"] = True

                url = self._client._href_definitions.get_gateway_text_completions_href()

                if stream:
                    return _streaming_create(
                        api_client=self._client, url=url, request_json=request_json
                    )
                else:
                    response = self._post_with_retry(
                        url=url,
                        headers=self._client._get_headers(include_container_id=True),
                        json=request_json,
                    )

                    return self._handle_response(
                        ACCEPTED_CREATE_STATUS_CODES,
                        "text completion creation",
                        response,
                    )

            @overload
            async def acreate(
                self,
                model: str,
                prompt: PromptInput,
                *,
                stream: Literal[False] = False,
                **kwargs: Any,
            ) -> dict: ...

            @overload
            async def acreate(
                self,
                model: str,
                prompt: PromptInput,
                *,
                stream: Literal[True],
                **kwargs: Any,
            ) -> AsyncIterator: ...

            async def acreate(
                self,
                model: str,
                prompt: PromptInput,
                *,
                stream: bool = False,
                **kwargs: Any,
            ) -> dict | AsyncIterator:
                """Generate text completions for given model and prompt asynchronously.

                :param model: name of model for given provider or alias
                :type model: str

                :param prompt: prompt for processing
                :type prompt: str or list[str] or list[int]

                :param stream: if True will stream the response, defaults to False
                :type stream: bool, optional

                :returns: model answer
                :rtype: dict | AsyncIterator
                """
                request_json = {"prompt": prompt, "model": model, **kwargs}
                if stream:
                    request_json["stream"] = True

                url = self._client._href_definitions.get_gateway_text_completions_href()

                if stream:
                    return _streaming_acreate(
                        api_client=self._client, url=url, request_json=request_json
                    )
                else:
                    response = await self._async_post_with_retry(
                        url=url,
                        headers=await self._client._aget_headers(
                            include_container_id=True
                        ),
                        json=request_json,
                    )

                    return self._handle_response(
                        ACCEPTED_CREATE_STATUS_CODES,
                        "text completion creation",
                        response,
                    )

        self.completions = _Completions(
            self._client, _post_with_retry, _async_post_with_retry
        )

        # Embeddings
        class _Embeddings(WMLResource):
            # Maximum number of inputs allowed per request by Model Gateway
            _MAX_BATCH_SIZE = 1000

            def __init__(
                self,
                api_client: APIClient,
                post_with_retry: Any,
                async_post_with_retry: Any,
            ):
                WMLResource.__init__(self, __name__, api_client)
                self._post_with_retry = post_with_retry
                self._async_post_with_retry = async_post_with_retry

            def _batch_inputs(self, inputs: PromptInput) -> BatchedPromptInput:
                """Split input into batches of maximum size.

                :param inputs: inputs to be batched
                :type inputs: str or list[str] or list[int]

                :returns: list of batched inputs
                :rtype: list
                """
                # If input is a string, return it as-is (single batch)
                if isinstance(inputs, str):
                    return [inputs]

                # Validate empty list inputs
                if isinstance(inputs, list) and len(inputs) == 0:
                    return [inputs]

                # If input is a list and within limit, return as single batch
                if len(inputs) <= self._MAX_BATCH_SIZE:
                    return [inputs]

                # Split into batches of _MAX_BATCH_SIZE
                batches: BatchedPromptInput = []
                for i in range(0, len(inputs), self._MAX_BATCH_SIZE):
                    batches.append(inputs[i : i + self._MAX_BATCH_SIZE])
                return batches

            @staticmethod
            def _merge_responses(responses: list[dict]) -> dict:
                """Merge multiple batch responses into a single response.

                :param responses: list of response dictionaries
                :type responses: list[dict]

                :returns: merged response
                :rtype: dict

                :raises WMLClientError: if responses list is empty
                """
                if not responses:
                    raise WMLClientError("Cannot merge empty responses list")

                if len(responses) == 1:
                    return responses[0]

                # Merge all embeddings data
                merged_data = []
                for response in responses:
                    if "data" in response:
                        merged_data.extend(response["data"])

                # Use the first response as template and update data
                merged_response = copy.deepcopy(responses[0])
                merged_response["data"] = merged_data

                # Update usage statistics if present
                if any("usage" in r for r in responses):
                    total_tokens = sum(
                        r.get("usage", {}).get("total_tokens", 0) for r in responses
                    )
                    if "usage" not in merged_response:
                        merged_response["usage"] = {}
                    merged_response["usage"]["total_tokens"] = total_tokens

                return merged_response

            def create(self, model: str, input: PromptInput, **kwargs: Any) -> dict:
                """Generate embeddings for given model and input.

                :param model: name of model for given provider or alias
                :type model: str

                :param input: prompt for processing
                :type input: str or list[str] or list[int]

                :returns: embeddings for given model and input
                :rtype: dict

                :raises WMLClientError: if any batch fails, includes information about successful batches
                """
                batches = self._batch_inputs(input)
                responses = []

                for batch_index, batch in enumerate(batches):
                    try:
                        request_json = {"input": batch, "model": model, **kwargs}

                        response = self._post_with_retry(
                            self._client._href_definitions.get_gateway_embeddings_href(),
                            headers=self._client._get_headers(
                                include_container_id=True
                            ),
                            json=request_json,
                        )

                        batch_response = self._handle_response(
                            ACCEPTED_CREATE_STATUS_CODES,
                            "embedding creation",
                            response,
                        )
                        responses.append(batch_response)
                    except Exception as e:
                        total_batches = len(batches)
                        successful_batches = len(responses)
                        raise WMLClientError(
                            f"Batch {batch_index + 1} of {total_batches} failed during embedding creation. "
                            f"Successfully processed {successful_batches} batch(es) before failure. "
                            f"Original error: {str(e)}"
                        ) from e

                return self._merge_responses(responses)

            async def acreate(
                self, model: str, input: PromptInput, **kwargs: Any
            ) -> dict:
                """Generate embeddings for given model and input asynchronously.

                :param model: name of model for given provider or alias
                :type model: str

                :param input: prompt for processing
                :type input: str or list[str] or list[int]

                :returns: embeddings for given model and input
                :rtype: dict

                :raises WMLClientError: if any batch fails, includes information about successful batches
                """
                batches = self._batch_inputs(input)
                responses = []

                for batch_index, batch in enumerate(batches):
                    try:
                        request_json = {"input": batch, "model": model, **kwargs}

                        response = await self._async_post_with_retry(
                            self._client._href_definitions.get_gateway_embeddings_href(),
                            headers=await self._client._aget_headers(
                                include_container_id=True
                            ),
                            json=request_json,
                        )

                        batch_response = self._handle_response(
                            ACCEPTED_CREATE_STATUS_CODES,
                            "embedding creation",
                            response,
                        )
                        responses.append(batch_response)
                    except Exception as e:
                        total_batches = len(batches)
                        successful_batches = len(responses)
                        raise WMLClientError(
                            f"Batch {batch_index + 1} of {total_batches} failed during embedding creation. "
                            f"Successfully processed {successful_batches} batch(es) before failure. "
                            f"Original error: {str(e)}"
                        ) from e

                return self._merge_responses(responses)

        self.embeddings = _Embeddings(
            self._client, _post_with_retry, _async_post_with_retry
        )
