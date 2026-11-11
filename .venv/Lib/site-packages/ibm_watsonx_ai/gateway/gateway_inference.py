#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from ibm_watsonx_ai.wml_client_error import InvalidMultipleArguments
from ibm_watsonx_ai.wml_resource import WMLResource

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient, Credentials


class GatewayInference(WMLResource):
    """Instantiate the AI Gateway model interface.

    :param model: type of model to use
    :type model: str

    .. note::

        The parameters below set **default values** applied to every request made through
        this instance. They can be overridden per-call by passing the same keyword argument
        directly to ``chat`` / ``generate`` (and their async / streaming variants).

    .. rubric:: Shared parameters (``chat`` + ``generate``)

    :param temperature: sampling temperature between 0 and 2; higher values make output more
        random, lower values make it more focused and deterministic
    :type temperature: float, optional

    :param max_tokens: maximum number of tokens that can be generated in the completion
    :type max_tokens: int, optional

    :param top_p: nucleus sampling probability mass; alternative to ``temperature``
    :type top_p: float, optional

    :param n: how many completion choices to generate for each input
    :type n: int, optional

    :param stop: stop sequence(s) — the API will stop generating further tokens when any of
        these sequences is encountered
    :type stop: dict or list[str], optional

    :param frequency_penalty: number between -2.0 and 2.0; positive values penalize new tokens
        based on their existing frequency in the text so far
    :type frequency_penalty: float, optional

    :param presence_penalty: number between -2.0 and 2.0; positive values penalize new tokens
        based on whether they appear in the text so far
    :type presence_penalty: float, optional

    :param logit_bias: modifies the likelihood of specified tokens appearing in the completion
    :type logit_bias: dict, optional

    :param seed: seed for deterministic sampling
    :type seed: int, optional

    :param stream_options: options for streaming responses
    :type stream_options: dict, optional

    :param metadata: metadata to store with the request
    :type metadata: dict, optional

    :param user: unique identifier representing your end-user
    :type user: str, optional

    :param router: model routing configuration for the request
    :type router: dict, optional

    :param cache: caching configuration for the request
    :type cache: dict, optional

    :param logprobs: log probabilities of the output tokens; pass ``True``/``False`` for chat
        endpoints or an integer (0–5) for generate endpoints
    :type logprobs: bool or int, optional

    .. rubric:: Chat-only parameters (``chat`` / ``chat_stream`` / ``achat`` / ``achat_stream``)

    :param max_completion_tokens: upper bound for the number of tokens that can be generated,
        including reasoning tokens; preferred over ``max_tokens`` for o-series models
    :type max_completion_tokens: int, optional

    :param top_logprobs: integer between 0 and 20 specifying the number of most likely tokens
        to return at each token position; requires ``logprobs=True``
    :type top_logprobs: int, optional

    :param reasoning_effort: reasoning effort configuration for the request
    :type reasoning_effort: dict, optional

    :param tools: list of tools the model may call; currently only functions are supported as tools; use this to provide a list of functions the model may generate JSON inputs for
    :type tools: list[dict], optional

    :param tool_choice: controls which tool is called by the model
    :type tool_choice: dict, optional

    :param parallel_tool_calls: whether to enable parallel function calling during tool use
    :type parallel_tool_calls: bool, optional

    :param function_call: controls which (if any) function is called by the model
        (deprecated in favor of ``tool_choice``)
    :type function_call: dict, optional

    :param functions: list of functions the model may generate JSON inputs for
        (deprecated in favor of ``tools``)
    :type functions: dict, optional

    :param response_format: object specifying the format that the model must output
        (e.g. JSON mode or structured outputs)
    :type response_format: dict, optional

    :param modalities: output types that the model should generate,
        e.g. ``["text"]`` or ``["text", "audio"]``
    :type modalities: list[str], optional

    :param audio: parameters for audio output, required when audio output is requested
        with modalities ``["audio"]``
    :type audio: dict, optional

    :param store: whether to store the output of this chat completion for use in model
        distillation or evals
    :type store: bool, optional

    :param service_tier: service tier configuration for the request
    :type service_tier: dict, optional

    :param prediction: prediction configuration for the request
    :type prediction: dict, optional

    .. rubric:: Generate-only parameters (``generate`` / ``generate_stream`` / ``agenerate`` / ``agenerate_stream``)

    :param best_of: generates ``best_of`` completions server-side and returns the best one;
        must be greater than ``n`` when used together; incompatible with streaming
    :type best_of: int, optional

    :param echo: whether to echo back the prompt in addition to the completion
    :type echo: bool, optional

    :param suffix: text that comes after a completion of inserted text
    :type suffix: str, optional

    .. rubric:: SDK / transport parameters

    :param credentials: credentials for the watsonx.ai instance
    :type credentials: Credentials or dict, optional

    :param api_client: initialized APIClient object with a set project ID or space ID.
        If passed, ``credentials`` and ``project_id``/``space_id`` are not required.
    :type api_client: APIClient, optional

    :param project_id: ID of the Watson Studio project
    :type project_id: str, optional

    :param space_id: ID of the Watson Studio space
    :type space_id: str, optional

    :param verify: You can pass one of the following as verify:

        * the path to a CA_BUNDLE file
        * the path of directory with certificates of trusted CAs
        * ``True`` — default path to truststore will be taken
        * ``False`` — no verification will be made
    :type verify: bool or str or Path, optional

    :param max_retries: number of retries performed when request was not successful and
        status code is in ``retry_status_codes``, defaults to 10
    :type max_retries: int, optional

    :param delay_time: delay time to retry request, factor in exponential backoff formula:
        ``wx_delay_time * pow(2.0, attempt)``, defaults to 0.5s
    :type delay_time: float, optional

    :param retry_status_codes: list of status codes which will be considered for retry
        mechanism, defaults to [429, 503, 504, 520]
    :type retry_status_codes: list[int], optional

    """

    def __init__(
        self,
        *,
        model: str,
        # --- shared: chat + completions ---
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        n: int | None = None,
        stop: dict | list[str] | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        logit_bias: dict | None = None,
        seed: int | None = None,
        stream_options: dict | None = None,
        metadata: dict | None = None,
        user: str | None = None,
        router: dict | None = None,
        cache: dict | None = None,
        logprobs: bool | int | None = None,
        # --- chat only ---
        max_completion_tokens: int | None = None,
        top_logprobs: int | None = None,
        reasoning_effort: dict | None = None,
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
        parallel_tool_calls: bool | None = None,
        function_call: dict | None = None,
        functions: dict | None = None,
        response_format: dict | None = None,
        modalities: list[str] | None = None,
        audio: dict | None = None,
        store: bool | None = None,
        service_tier: dict | None = None,
        prediction: dict | None = None,
        # --- completions only ---
        best_of: int | None = None,
        echo: bool | None = None,
        suffix: str | None = None,
        # --- sdk / transport ---
        credentials: dict | Credentials | None = None,
        api_client: APIClient | None = None,
        project_id: str | None = None,
        space_id: str | None = None,
        verify: bool | str | Path | None = None,
        # --- retry ---
        max_retries: int | None = None,
        delay_time: float | None = None,
        retry_status_codes: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        self._model: str = model

        if isinstance(self._model, Enum):
            self._model = self._model.value

        client = self._build_client(credentials, api_client, verify)

        if space_id:
            client.set.default_space(space_id)
        elif project_id:
            client.set.default_project(project_id)

        WMLResource.__init__(self, __name__, client)

        from ibm_watsonx_ai.gateway import Gateway

        self._gateway = Gateway(
            api_client=client,
            max_retries=max_retries,
            delay_time=delay_time,
            retry_status_codes=retry_status_codes,
        )

        # shared keys present in both endpoints
        _shared = {
            **kwargs,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "n": n,
            "stop": stop,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "logit_bias": logit_bias,
            "seed": seed,
            "stream_options": stream_options,
            "metadata": metadata,
            "user": user,
            "router": router,
            "cache": cache,
            "logprobs": logprobs,
        }

        self._chat_params: dict[str, Any] = {
            k: v
            for k, v in {
                **_shared,
                "max_completion_tokens": max_completion_tokens,
                "top_logprobs": top_logprobs,
                "reasoning_effort": reasoning_effort,
                "tools": tools,
                "tool_choice": tool_choice,
                "parallel_tool_calls": parallel_tool_calls,
                "function_call": function_call,
                "functions": functions,
                "response_format": response_format,
                "modalities": modalities,
                "audio": audio,
                "store": store,
                "service_tier": service_tier,
                "prediction": prediction,
            }.items()
            if v is not None
        }

        self._completion_params: dict[str, Any] = {
            k: v
            for k, v in {
                **_shared,
                "best_of": best_of,
                "echo": echo,
                "suffix": suffix,
            }.items()
            if v is not None
        }

    @staticmethod
    def _build_client(
        credentials: dict | Credentials | None,
        api_client: APIClient | None,
        verify: bool | str | Path | None,
    ) -> APIClient:
        if credentials and api_client:
            raise InvalidMultipleArguments(
                params_names_list=["credentials", "api_client"],
                reason="Only one of the arguments should be provided.",
            )

        if credentials:
            from ibm_watsonx_ai import APIClient

            if isinstance(verify, str):
                verify = Path(verify)
            return APIClient(credentials, verify=verify)

        if api_client:
            return api_client

        raise InvalidMultipleArguments(
            params_names_list=["credentials", "api_client"],
            reason="None of the arguments were provided.",
        )

    def chat(
        self,
        messages: list[dict],
        *,
        # --- core sampling ---
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        n: int | None = None,
        stop: dict | None = None,
        # --- sampling / output control ---
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        logit_bias: dict | None = None,
        seed: int | None = None,
        reasoning_effort: dict | None = None,
        # --- tools ---
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
        parallel_tool_calls: bool | None = None,
        function_call: dict | None = None,
        functions: dict | None = None,
        # --- format / modalities ---
        response_format: dict | None = None,
        modalities: list[str] | None = None,
        audio: dict | None = None,
        stream_options: dict | None = None,
        # --- metadata / misc ---
        store: bool | None = None,
        metadata: dict | None = None,
        user: str | None = None,
        service_tier: dict | None = None,
        prediction: dict | None = None,
        router: dict | None = None,
        cache: dict | None = None,
        **kwargs: Any,
    ) -> dict:
        """Generate chat completions.

        :param messages: messages to be processed during call
        :type messages: list[dict]

        :param temperature: sampling temperature between 0 and 2; higher values make output more random
        :type temperature: float, optional

        :param max_completion_tokens: upper bound for the number of tokens that can be generated for a completion, including reasoning tokens
        :type max_completion_tokens: int, optional

        :param max_tokens: maximum number of tokens that can be generated in the chat completion (deprecated in favor of ``max_completion_tokens``)
        :type max_tokens: int, optional

        :param top_p: nucleus sampling probability mass; alternative to ``temperature``
        :type top_p: float, optional

        :param n: how many chat completion choices to generate for each input message
        :type n: int, optional

        :param stop: stop sequence configuration for the request
        :type stop: dict, optional

        :param frequency_penalty: number between -2.0 and 2.0; positive values penalize new tokens based on their existing frequency in the text so far
        :type frequency_penalty: float, optional

        :param presence_penalty: number between -2.0 and 2.0; positive values penalize new tokens based on whether they appear in the text so far
        :type presence_penalty: float, optional

        :param logprobs: whether to return log probabilities of the output tokens
        :type logprobs: bool, optional

        :param top_logprobs: integer between 0 and 20 specifying the number of most likely tokens to return at each token position
        :type top_logprobs: int, optional

        :param logit_bias: modifies the likelihood of specified tokens appearing in the completion
        :type logit_bias: dict, optional

        :param seed: seed for deterministic sampling (Beta for OpenAI)
        :type seed: int, optional

        :param reasoning_effort: reasoning effort configuration for the request
        :type reasoning_effort: dict, optional

        :param tools: list of tools the model may call; currently only functions are supported as tools; use this to provide a list of functions the model may generate JSON inputs for
        :type tools: list[dict], optional

        :param tool_choice: controls which tool is called by the model
        :type tool_choice: dict, optional

        :param parallel_tool_calls: whether to enable parallel function calling during tool use
        :type parallel_tool_calls: bool, optional

        :param function_call: controls which (if any) function is called by the model (deprecated in favor of ``tool_choice``)
        :type function_call: dict, optional

        :param functions: list of functions the model may generate JSON inputs for (deprecated in favor of ``tools``)
        :type functions: dict, optional

        :param response_format: object specifying the format that the model must output (e.g. JSON mode or structured outputs)
        :type response_format: dict, optional

        :param modalities: output types that the model should generate, e.g. ``["text"]`` or ``["text", "audio"]``
        :type modalities: list[str], optional

        :param audio: parameters for audio output, required when audio output is requested with modalities ``["audio"]``
        :type audio: dict, optional

        :param stream_options: options for streaming responses
        :type stream_options: dict, optional

        :param store: whether to store the output of this chat completion for use in model distillation or evals
        :type store: bool, optional

        :param metadata: metadata for the request
        :type metadata: dict, optional

        :param user: unique identifier representing your end-user
        :type user: str, optional

        :param service_tier: service tier configuration for the request
        :type service_tier: dict, optional

        :param prediction: prediction configuration for the request
        :type prediction: dict, optional

        :param router: model routing configurations for the request
        :type router: dict, optional

        :param cache: caching configuration for the request
        :type cache: dict, optional

        :param kwargs: additional keyword arguments passed directly to the gateway
        :type kwargs: Any

        :returns: model answer
        :rtype: dict
        """
        self._validate_type(messages, "messages", list, True)
        call_params = self._drop_nones_from_payload(
            kwargs,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            max_tokens=max_tokens,
            top_p=top_p,
            n=n,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            logit_bias=logit_bias,
            seed=seed,
            reasoning_effort=reasoning_effort,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            function_call=function_call,
            functions=functions,
            response_format=response_format,
            modalities=modalities,
            audio=audio,
            stream_options=stream_options,
            store=store,
            metadata=metadata,
            user=user,
            service_tier=service_tier,
            prediction=prediction,
            router=router,
            cache=cache,
        )
        return self._gateway.chat.completions.create(
            model=self._model, messages=messages, **{**self._chat_params, **call_params}
        )

    def chat_stream(
        self,
        messages: list[dict],
        *,
        # --- core sampling ---
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        n: int | None = None,
        stop: dict | None = None,
        # --- sampling / output control ---
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        logit_bias: dict | None = None,
        seed: int | None = None,
        reasoning_effort: dict | None = None,
        # --- tools ---
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
        parallel_tool_calls: bool | None = None,
        function_call: dict | None = None,
        functions: dict | None = None,
        # --- format / modalities ---
        response_format: dict | None = None,
        modalities: list[str] | None = None,
        audio: dict | None = None,
        stream_options: dict | None = None,
        # --- metadata / misc ---
        store: bool | None = None,
        metadata: dict | None = None,
        user: str | None = None,
        service_tier: dict | None = None,
        prediction: dict | None = None,
        router: dict | None = None,
        cache: dict | None = None,
        **kwargs: Any,
    ) -> Iterator:
        """Generate chat completions with streaming.

        :param messages: messages to be processed during call
        :type messages: list[dict]

        :param temperature: sampling temperature between 0 and 2; higher values make output more random
        :type temperature: float, optional

        :param max_completion_tokens: upper bound for the number of tokens that can be generated for a completion, including reasoning tokens
        :type max_completion_tokens: int, optional

        :param max_tokens: maximum number of tokens that can be generated in the chat completion (deprecated in favor of ``max_completion_tokens``)
        :type max_tokens: int, optional

        :param top_p: nucleus sampling probability mass; alternative to ``temperature``
        :type top_p: float, optional

        :param n: how many chat completion choices to generate for each input message
        :type n: int, optional

        :param stop: stop sequence configuration for the request
        :type stop: dict, optional

        :param frequency_penalty: number between -2.0 and 2.0; positive values penalize new tokens based on their existing frequency in the text so far
        :type frequency_penalty: float, optional

        :param presence_penalty: number between -2.0 and 2.0; positive values penalize new tokens based on whether they appear in the text so far
        :type presence_penalty: float, optional

        :param logprobs: whether to return log probabilities of the output tokens
        :type logprobs: bool, optional

        :param top_logprobs: integer between 0 and 20 specifying the number of most likely tokens to return at each token position
        :type top_logprobs: int, optional

        :param logit_bias: modifies the likelihood of specified tokens appearing in the completion
        :type logit_bias: dict, optional

        :param seed: seed for deterministic sampling (Beta for OpenAI)
        :type seed: int, optional

        :param reasoning_effort: reasoning effort configuration for the request
        :type reasoning_effort: dict, optional

        :param tools: list of tools the model may call; currently only functions are supported as tools; use this to provide a list of functions the model may generate JSON inputs for
        :type tools: list[dict], optional

        :param tool_choice: controls which tool is called by the model
        :type tool_choice: dict, optional

        :param parallel_tool_calls: whether to enable parallel function calling during tool use
        :type parallel_tool_calls: bool, optional

        :param function_call: controls which (if any) function is called by the model (deprecated in favor of ``tool_choice``)
        :type function_call: dict, optional

        :param functions: list of functions the model may generate JSON inputs for (deprecated in favor of ``tools``)
        :type functions: dict, optional

        :param response_format: object specifying the format that the model must output (e.g. JSON mode or structured outputs)
        :type response_format: dict, optional

        :param modalities: output types that the model should generate, e.g. ``["text"]`` or ``["text", "audio"]``
        :type modalities: list[str], optional

        :param audio: parameters for audio output, required when audio output is requested with modalities ``["audio"]``
        :type audio: dict, optional

        :param stream_options: options for streaming responses
        :type stream_options: dict, optional

        :param store: whether to store the output of this chat completion for use in model distillation or evals
        :type store: bool, optional

        :param metadata: metadata for the request
        :type metadata: dict, optional

        :param user: unique identifier representing your end-user
        :type user: str, optional

        :param service_tier: service tier configuration for the request
        :type service_tier: dict, optional

        :param prediction: prediction configuration for the request
        :type prediction: dict, optional

        :param router: model routing configurations for the request
        :type router: dict, optional

        :param cache: caching configuration for the request
        :type cache: dict, optional

        :param kwargs: additional keyword arguments passed directly to the gateway
        :type kwargs: Any

        :returns: iterator of model response chunks
        :rtype: Iterator
        """
        self._validate_type(messages, "messages", list, True)
        call_params = self._drop_nones_from_payload(
            kwargs,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            max_tokens=max_tokens,
            top_p=top_p,
            n=n,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            logit_bias=logit_bias,
            seed=seed,
            reasoning_effort=reasoning_effort,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            function_call=function_call,
            functions=functions,
            response_format=response_format,
            modalities=modalities,
            audio=audio,
            stream_options=stream_options,
            store=store,
            metadata=metadata,
            user=user,
            service_tier=service_tier,
            prediction=prediction,
            router=router,
            cache=cache,
        )
        return self._gateway.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=True,
            **{**self._chat_params, **call_params},
        )

    async def achat(
        self,
        messages: list[dict],
        *,
        # --- core sampling ---
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        n: int | None = None,
        stop: dict | None = None,
        # --- sampling / output control ---
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        logit_bias: dict | None = None,
        seed: int | None = None,
        reasoning_effort: dict | None = None,
        # --- tools ---
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
        parallel_tool_calls: bool | None = None,
        function_call: dict | None = None,
        functions: dict | None = None,
        # --- format / modalities ---
        response_format: dict | None = None,
        modalities: list[str] | None = None,
        audio: dict | None = None,
        stream_options: dict | None = None,
        # --- metadata / misc ---
        store: bool | None = None,
        metadata: dict | None = None,
        user: str | None = None,
        service_tier: dict | None = None,
        prediction: dict | None = None,
        router: dict | None = None,
        cache: dict | None = None,
        **kwargs: Any,
    ) -> dict:
        """Generate chat completions asynchronously.

        :param messages: messages to be processed during call
        :type messages: list[dict]

        :param temperature: sampling temperature between 0 and 2; higher values make output more random
        :type temperature: float, optional

        :param max_completion_tokens: upper bound for the number of tokens that can be generated for a completion, including reasoning tokens
        :type max_completion_tokens: int, optional

        :param max_tokens: maximum number of tokens that can be generated in the chat completion (deprecated in favor of ``max_completion_tokens``)
        :type max_tokens: int, optional

        :param top_p: nucleus sampling probability mass; alternative to ``temperature``
        :type top_p: float, optional

        :param n: how many chat completion choices to generate for each input message
        :type n: int, optional

        :param stop: stop sequence configuration for the request
        :type stop: dict, optional

        :param frequency_penalty: number between -2.0 and 2.0; positive values penalize new tokens based on their existing frequency in the text so far
        :type frequency_penalty: float, optional

        :param presence_penalty: number between -2.0 and 2.0; positive values penalize new tokens based on whether they appear in the text so far
        :type presence_penalty: float, optional

        :param logprobs: whether to return log probabilities of the output tokens
        :type logprobs: bool, optional

        :param top_logprobs: integer between 0 and 20 specifying the number of most likely tokens to return at each token position
        :type top_logprobs: int, optional

        :param logit_bias: modifies the likelihood of specified tokens appearing in the completion
        :type logit_bias: dict, optional

        :param seed: seed for deterministic sampling (Beta for OpenAI)
        :type seed: int, optional

        :param reasoning_effort: reasoning effort configuration for the request
        :type reasoning_effort: dict, optional

        :param tools: list of tools the model may call; currently only functions are supported as tools; use this to provide a list of functions the model may generate JSON inputs for
        :type tools: list[dict], optional

        :param tool_choice: controls which tool is called by the model
        :type tool_choice: dict, optional

        :param parallel_tool_calls: whether to enable parallel function calling during tool use
        :type parallel_tool_calls: bool, optional

        :param function_call: controls which (if any) function is called by the model (deprecated in favor of ``tool_choice``)
        :type function_call: dict, optional

        :param functions: list of functions the model may generate JSON inputs for (deprecated in favor of ``tools``)
        :type functions: dict, optional

        :param response_format: object specifying the format that the model must output (e.g. JSON mode or structured outputs)
        :type response_format: dict, optional

        :param modalities: output types that the model should generate, e.g. ``["text"]`` or ``["text", "audio"]``
        :type modalities: list[str], optional

        :param audio: parameters for audio output, required when audio output is requested with modalities ``["audio"]``
        :type audio: dict, optional

        :param stream_options: options for streaming responses
        :type stream_options: dict, optional

        :param store: whether to store the output of this chat completion for use in model distillation or evals
        :type store: bool, optional

        :param metadata: metadata for the request
        :type metadata: dict, optional

        :param user: unique identifier representing your end-user
        :type user: str, optional

        :param service_tier: service tier configuration for the request
        :type service_tier: dict, optional

        :param prediction: prediction configuration for the request
        :type prediction: dict, optional

        :param router: model routing configurations for the request
        :type router: dict, optional

        :param cache: caching configuration for the request
        :type cache: dict, optional

        :param kwargs: additional keyword arguments passed directly to the gateway
        :type kwargs: Any

        :returns: model answer
        :rtype: dict
        """
        self._validate_type(messages, "messages", list, True)
        call_params = self._drop_nones_from_payload(
            kwargs,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            max_tokens=max_tokens,
            top_p=top_p,
            n=n,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            logit_bias=logit_bias,
            seed=seed,
            reasoning_effort=reasoning_effort,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            function_call=function_call,
            functions=functions,
            response_format=response_format,
            modalities=modalities,
            audio=audio,
            stream_options=stream_options,
            store=store,
            metadata=metadata,
            user=user,
            service_tier=service_tier,
            prediction=prediction,
            router=router,
            cache=cache,
        )
        return await self._gateway.chat.completions.acreate(
            model=self._model, messages=messages, **{**self._chat_params, **call_params}
        )

    async def achat_stream(
        self,
        messages: list[dict],
        *,
        # --- core sampling ---
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        n: int | None = None,
        stop: dict | None = None,
        # --- sampling / output control ---
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        logit_bias: dict | None = None,
        seed: int | None = None,
        reasoning_effort: dict | None = None,
        # --- tools ---
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
        parallel_tool_calls: bool | None = None,
        function_call: dict | None = None,
        functions: dict | None = None,
        # --- format / modalities ---
        response_format: dict | None = None,
        modalities: list[str] | None = None,
        audio: dict | None = None,
        stream_options: dict | None = None,
        # --- metadata / misc ---
        store: bool | None = None,
        metadata: dict | None = None,
        user: str | None = None,
        service_tier: dict | None = None,
        prediction: dict | None = None,
        router: dict | None = None,
        cache: dict | None = None,
        **kwargs: Any,
    ) -> AsyncIterator:
        """Generate chat completions asynchronously with streaming.

        :param messages: messages to be processed during call
        :type messages: list[dict]

        :param temperature: sampling temperature between 0 and 2; higher values make output more random
        :type temperature: float, optional

        :param max_completion_tokens: upper bound for the number of tokens that can be generated for a completion, including reasoning tokens
        :type max_completion_tokens: int, optional

        :param max_tokens: maximum number of tokens that can be generated in the chat completion (deprecated in favor of ``max_completion_tokens``)
        :type max_tokens: int, optional

        :param top_p: nucleus sampling probability mass; alternative to ``temperature``
        :type top_p: float, optional

        :param n: how many chat completion choices to generate for each input message
        :type n: int, optional

        :param stop: stop sequence configuration for the request
        :type stop: dict, optional

        :param frequency_penalty: number between -2.0 and 2.0; positive values penalize new tokens based on their existing frequency in the text so far
        :type frequency_penalty: float, optional

        :param presence_penalty: number between -2.0 and 2.0; positive values penalize new tokens based on whether they appear in the text so far
        :type presence_penalty: float, optional

        :param logprobs: whether to return log probabilities of the output tokens
        :type logprobs: bool, optional

        :param top_logprobs: integer between 0 and 20 specifying the number of most likely tokens to return at each token position
        :type top_logprobs: int, optional

        :param logit_bias: modifies the likelihood of specified tokens appearing in the completion
        :type logit_bias: dict, optional

        :param seed: seed for deterministic sampling (Beta for OpenAI)
        :type seed: int, optional

        :param reasoning_effort: reasoning effort configuration for the request
        :type reasoning_effort: dict, optional

        :param tools: list of tools the model may call; currently only functions are supported as tools; use this to provide a list of functions the model may generate JSON inputs for
        :type tools: list[dict], optional

        :param tool_choice: controls which tool is called by the model
        :type tool_choice: dict, optional

        :param parallel_tool_calls: whether to enable parallel function calling during tool use
        :type parallel_tool_calls: bool, optional

        :param function_call: controls which (if any) function is called by the model (deprecated in favor of ``tool_choice``)
        :type function_call: dict, optional

        :param functions: list of functions the model may generate JSON inputs for (deprecated in favor of ``tools``)
        :type functions: dict, optional

        :param response_format: object specifying the format that the model must output (e.g. JSON mode or structured outputs)
        :type response_format: dict, optional

        :param modalities: output types that the model should generate, e.g. ``["text"]`` or ``["text", "audio"]``
        :type modalities: list[str], optional

        :param audio: parameters for audio output, required when audio output is requested with modalities ``["audio"]``
        :type audio: dict, optional

        :param stream_options: options for streaming responses
        :type stream_options: dict, optional

        :param store: whether to store the output of this chat completion for use in model distillation or evals
        :type store: bool, optional

        :param metadata: metadata for the request
        :type metadata: dict, optional

        :param user: unique identifier representing your end-user
        :type user: str, optional

        :param service_tier: service tier configuration for the request
        :type service_tier: dict, optional

        :param prediction: prediction configuration for the request
        :type prediction: dict, optional

        :param router: model routing configurations for the request
        :type router: dict, optional

        :param cache: caching configuration for the request
        :type cache: dict, optional

        :param kwargs: additional keyword arguments passed directly to the gateway
        :type kwargs: Any

        :returns: async iterator of model response chunks
        :rtype: AsyncIterator
        """
        self._validate_type(messages, "messages", list, True)
        call_params = self._drop_nones_from_payload(
            kwargs,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            max_tokens=max_tokens,
            top_p=top_p,
            n=n,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            logit_bias=logit_bias,
            seed=seed,
            reasoning_effort=reasoning_effort,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            function_call=function_call,
            functions=functions,
            response_format=response_format,
            modalities=modalities,
            audio=audio,
            stream_options=stream_options,
            store=store,
            metadata=metadata,
            user=user,
            service_tier=service_tier,
            prediction=prediction,
            router=router,
            cache=cache,
        )
        return await self._gateway.chat.completions.acreate(
            model=self._model,
            messages=messages,
            stream=True,
            **{**self._chat_params, **call_params},
        )

    def generate(
        self,
        prompt: Union[str, list[str], list[int]],
        *,
        # --- core sampling ---
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        n: int | None = None,
        best_of: int | None = None,
        stop: list[str] | None = None,
        # --- sampling / output control ---
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        logprobs: int | None = None,
        logit_bias: dict | None = None,
        seed: int | None = None,
        echo: bool | None = None,
        suffix: str | None = None,
        stream_options: dict | None = None,
        # --- metadata / misc ---
        metadata: dict | None = None,
        user: str | None = None,
        router: dict | None = None,
        cache: dict | None = None,
        **kwargs: Any,
    ) -> dict:
        """Generate text completions.

        :param prompt: prompt to be processed during call
        :type prompt: str or list[str] or list[int]

        :param temperature: sampling temperature between 0 and 2; higher values make output more random,
            lower values make it more focused and deterministic
        :type temperature: float, optional

        :param max_tokens: maximum number of tokens that can be generated in the completion;
            token count of prompt plus ``max_tokens`` cannot exceed the model's context length
        :type max_tokens: int, optional

        :param top_p: nucleus sampling probability mass; alternative to ``temperature``
        :type top_p: float, optional

        :param n: how many completions to generate for each prompt
        :type n: int, optional

        :param best_of: generates ``best_of`` completions server-side and returns the best one;
            must be greater than ``n`` when used together; cannot be used with streaming
        :type best_of: int, optional

        :param stop: up to 4 sequences where the API will stop generating further tokens;
            the returned text will not contain the stop sequence
        :type stop: list[str], optional

        :param frequency_penalty: number between -2.0 and 2.0; positive values penalize new tokens
            based on their existing frequency in the text so far
        :type frequency_penalty: float, optional

        :param presence_penalty: number between -2.0 and 2.0; positive values penalize new tokens
            based on whether they appear in the text so far
        :type presence_penalty: float, optional

        :param logprobs: number of most likely output tokens (0–5) to include log probabilities for
        :type logprobs: int, optional

        :param logit_bias: modifies the likelihood of specified tokens appearing in the completion;
            maps token IDs to a bias value from -100 to 100
        :type logit_bias: dict, optional

        :param seed: seed for deterministic sampling; repeated requests with the same seed and
            parameters should return the same result
        :type seed: int, optional

        :param echo: whether to echo back the prompt in addition to the completion
        :type echo: bool, optional

        :param suffix: text that comes after a completion of inserted text
        :type suffix: str, optional

        :param stream_options: options for streaming responses; only applicable when ``stream=True``
        :type stream_options: dict, optional

        :param metadata: metadata to store with the completion
        :type metadata: dict, optional

        :param user: unique identifier representing your end-user
        :type user: str, optional

        :param router: model routing configuration for the request
        :type router: dict, optional

        :param cache: caching configuration for the request; only supported for non-streaming requests
        :type cache: dict, optional

        :param kwargs: additional keyword arguments passed directly to the gateway
        :type kwargs: Any

        :returns: model answer
        :rtype: dict
        """
        self._validate_type(prompt, "prompt", [str, list], True)
        call_params = self._drop_nones_from_payload(
            kwargs,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n=n,
            best_of=best_of,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logprobs=logprobs,
            logit_bias=logit_bias,
            seed=seed,
            echo=echo,
            suffix=suffix,
            stream_options=stream_options,
            metadata=metadata,
            user=user,
            router=router,
            cache=cache,
        )
        return self._gateway.completions.create(
            model=self._model,
            prompt=prompt,
            **{**self._completion_params, **call_params},
        )

    async def agenerate(
        self,
        prompt: Union[str, list[str], list[int]],
        *,
        # --- core sampling ---
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        n: int | None = None,
        best_of: int | None = None,
        stop: list[str] | None = None,
        # --- sampling / output control ---
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        logprobs: int | None = None,
        logit_bias: dict | None = None,
        seed: int | None = None,
        echo: bool | None = None,
        suffix: str | None = None,
        stream_options: dict | None = None,
        # --- metadata / misc ---
        metadata: dict | None = None,
        user: str | None = None,
        router: dict | None = None,
        cache: dict | None = None,
        **kwargs: Any,
    ) -> dict:
        """Generate text completions asynchronously.

        :param prompt: prompt to be processed during call
        :type prompt: str or list[str] or list[int]

        :param temperature: sampling temperature between 0 and 2; higher values make output more random,
            lower values make it more focused and deterministic
        :type temperature: float, optional

        :param max_tokens: maximum number of tokens that can be generated in the completion;
            token count of prompt plus ``max_tokens`` cannot exceed the model's context length
        :type max_tokens: int, optional

        :param top_p: nucleus sampling probability mass; alternative to ``temperature``
        :type top_p: float, optional

        :param n: how many completions to generate for each prompt
        :type n: int, optional

        :param best_of: generates ``best_of`` completions server-side and returns the best one;
            must be greater than ``n`` when used together; cannot be used with streaming
        :type best_of: int, optional

        :param stop: up to 4 sequences where the API will stop generating further tokens;
            the returned text will not contain the stop sequence
        :type stop: list[str], optional

        :param frequency_penalty: number between -2.0 and 2.0; positive values penalize new tokens
            based on their existing frequency in the text so far
        :type frequency_penalty: float, optional

        :param presence_penalty: number between -2.0 and 2.0; positive values penalize new tokens
            based on whether they appear in the text so far
        :type presence_penalty: float, optional

        :param logprobs: number of most likely output tokens (0–5) to include log probabilities for
        :type logprobs: int, optional

        :param logit_bias: modifies the likelihood of specified tokens appearing in the completion;
            maps token IDs to a bias value from -100 to 100
        :type logit_bias: dict, optional

        :param seed: seed for deterministic sampling; repeated requests with the same seed and
            parameters should return the same result
        :type seed: int, optional

        :param echo: whether to echo back the prompt in addition to the completion
        :type echo: bool, optional

        :param suffix: text that comes after a completion of inserted text
        :type suffix: str, optional

        :param stream_options: options for streaming responses; only applicable when ``stream=True``
        :type stream_options: dict, optional

        :param metadata: metadata to store with the completion
        :type metadata: dict, optional

        :param user: unique identifier representing your end-user
        :type user: str, optional

        :param router: model routing configuration for the request
        :type router: dict, optional

        :param cache: caching configuration for the request; only supported for non-streaming requests
        :type cache: dict, optional

        :param kwargs: additional keyword arguments passed directly to the gateway
        :type kwargs: Any

        :returns: model answer
        :rtype: dict
        """
        self._validate_type(prompt, "prompt", [str, list], True)
        call_params = self._drop_nones_from_payload(
            kwargs,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n=n,
            best_of=best_of,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logprobs=logprobs,
            logit_bias=logit_bias,
            seed=seed,
            echo=echo,
            suffix=suffix,
            stream_options=stream_options,
            metadata=metadata,
            user=user,
            router=router,
            cache=cache,
        )
        return await self._gateway.completions.acreate(
            model=self._model,
            prompt=prompt,
            **{**self._completion_params, **call_params},
        )

    def generate_stream(
        self,
        prompt: Union[str, list[str], list[int]],
        *,
        # --- core sampling ---
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        n: int | None = None,
        best_of: int | None = None,
        stop: list[str] | None = None,
        # --- sampling / output control ---
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        logprobs: int | None = None,
        logit_bias: dict | None = None,
        seed: int | None = None,
        echo: bool | None = None,
        suffix: str | None = None,
        stream_options: dict | None = None,
        # --- metadata / misc ---
        metadata: dict | None = None,
        user: str | None = None,
        router: dict | None = None,
        cache: dict | None = None,
        **kwargs: Any,
    ) -> Iterator:
        """Generate text completions with streaming.

        :param prompt: prompt to be processed during call
        :type prompt: str or list[str] or list[int]

        :param temperature: sampling temperature between 0 and 2; higher values make output more random,
            lower values make it more focused and deterministic
        :type temperature: float, optional

        :param max_tokens: maximum number of tokens that can be generated in the completion;
            token count of prompt plus ``max_tokens`` cannot exceed the model's context length
        :type max_tokens: int, optional

        :param top_p: nucleus sampling probability mass; alternative to ``temperature``
        :type top_p: float, optional

        :param n: how many completions to generate for each prompt
        :type n: int, optional

        :param best_of: generates ``best_of`` completions server-side and returns the best one;
            must be greater than ``n`` when used together; cannot be used with streaming
        :type best_of: int, optional

        :param stop: up to 4 sequences where the API will stop generating further tokens;
            the returned text will not contain the stop sequence
        :type stop: list[str], optional

        :param frequency_penalty: number between -2.0 and 2.0; positive values penalize new tokens
            based on their existing frequency in the text so far
        :type frequency_penalty: float, optional

        :param presence_penalty: number between -2.0 and 2.0; positive values penalize new tokens
            based on whether they appear in the text so far
        :type presence_penalty: float, optional

        :param logprobs: number of most likely output tokens (0–5) to include log probabilities for
        :type logprobs: int, optional

        :param logit_bias: modifies the likelihood of specified tokens appearing in the completion;
            maps token IDs to a bias value from -100 to 100
        :type logit_bias: dict, optional

        :param seed: seed for deterministic sampling; repeated requests with the same seed and
            parameters should return the same result
        :type seed: int, optional

        :param echo: whether to echo back the prompt in addition to the completion
        :type echo: bool, optional

        :param suffix: text that comes after a completion of inserted text
        :type suffix: str, optional

        :param stream_options: options for streaming responses
        :type stream_options: dict, optional

        :param metadata: metadata to store with the completion
        :type metadata: dict, optional

        :param user: unique identifier representing your end-user
        :type user: str, optional

        :param router: model routing configuration for the request
        :type router: dict, optional

        :param cache: caching configuration for the request
        :type cache: dict, optional

        :param kwargs: additional keyword arguments passed directly to the gateway
        :type kwargs: Any

        :returns: iterator of model response chunks
        :rtype: Iterator
        """
        self._validate_type(prompt, "prompt", [str, list], True)
        call_params = self._drop_nones_from_payload(
            kwargs,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n=n,
            best_of=best_of,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logprobs=logprobs,
            logit_bias=logit_bias,
            seed=seed,
            echo=echo,
            suffix=suffix,
            stream_options=stream_options,
            metadata=metadata,
            user=user,
            router=router,
            cache=cache,
        )
        return self._gateway.completions.create(
            model=self._model,
            prompt=prompt,
            stream=True,
            **{**self._completion_params, **call_params},
        )

    async def agenerate_stream(
        self,
        prompt: Union[str, list[str], list[int]],
        *,
        # --- core sampling ---
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        n: int | None = None,
        best_of: int | None = None,
        stop: list[str] | None = None,
        # --- sampling / output control ---
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        logprobs: int | None = None,
        logit_bias: dict | None = None,
        seed: int | None = None,
        echo: bool | None = None,
        suffix: str | None = None,
        stream_options: dict | None = None,
        # --- metadata / misc ---
        metadata: dict | None = None,
        user: str | None = None,
        router: dict | None = None,
        cache: dict | None = None,
        **kwargs: Any,
    ) -> AsyncIterator:
        """Generate text completions asynchronously with streaming.

        :param prompt: prompt to be processed during call
        :type prompt: str or list[str] or list[int]

        :param temperature: sampling temperature between 0 and 2; higher values make output more random,
            lower values make it more focused and deterministic
        :type temperature: float, optional

        :param max_tokens: maximum number of tokens that can be generated in the completion;
            token count of prompt plus ``max_tokens`` cannot exceed the model's context length
        :type max_tokens: int, optional

        :param top_p: nucleus sampling probability mass; alternative to ``temperature``
        :type top_p: float, optional

        :param n: how many completions to generate for each prompt
        :type n: int, optional

        :param best_of: generates ``best_of`` completions server-side and returns the best one;
            must be greater than ``n`` when used together; cannot be used with streaming
        :type best_of: int, optional

        :param stop: up to 4 sequences where the API will stop generating further tokens;
            the returned text will not contain the stop sequence
        :type stop: list[str], optional

        :param frequency_penalty: number between -2.0 and 2.0; positive values penalize new tokens
            based on their existing frequency in the text so far
        :type frequency_penalty: float, optional

        :param presence_penalty: number between -2.0 and 2.0; positive values penalize new tokens
            based on whether they appear in the text so far
        :type presence_penalty: float, optional

        :param logprobs: number of most likely output tokens (0–5) to include log probabilities for
        :type logprobs: int, optional

        :param logit_bias: modifies the likelihood of specified tokens appearing in the completion;
            maps token IDs to a bias value from -100 to 100
        :type logit_bias: dict, optional

        :param seed: seed for deterministic sampling; repeated requests with the same seed and
            parameters should return the same result
        :type seed: int, optional

        :param echo: whether to echo back the prompt in addition to the completion
        :type echo: bool, optional

        :param suffix: text that comes after a completion of inserted text
        :type suffix: str, optional

        :param stream_options: options for streaming responses
        :type stream_options: dict, optional

        :param metadata: metadata to store with the completion
        :type metadata: dict, optional

        :param user: unique identifier representing your end-user
        :type user: str, optional

        :param router: model routing configuration for the request
        :type router: dict, optional

        :param cache: caching configuration for the request
        :type cache: dict, optional

        :param kwargs: additional keyword arguments passed directly to the gateway
        :type kwargs: Any

        :returns: async iterator of model response chunks
        :rtype: AsyncIterator
        """
        self._validate_type(prompt, "prompt", [str, list], True)
        call_params = self._drop_nones_from_payload(
            kwargs,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n=n,
            best_of=best_of,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logprobs=logprobs,
            logit_bias=logit_bias,
            seed=seed,
            echo=echo,
            suffix=suffix,
            stream_options=stream_options,
            metadata=metadata,
            user=user,
            router=router,
            cache=cache,
        )
        return await self._gateway.completions.acreate(
            model=self._model,
            prompt=prompt,
            stream=True,
            **{**self._completion_params, **call_params},
        )

    def get_details(self) -> dict | list[dict]:
        """Get the details of the model(s) registered under the current model ID / alias.

        Calls :meth:`~ibm_watsonx_ai.gateway.Models.get_details` and filters the
        results to entries whose ``id`` or ``alias`` matches ``self._model``.

        :return: a single model detail dict when exactly one match is found,
            or a list of dicts when multiple providers expose the same model ID / alias
        :rtype: dict | list[dict]

        **Example:**

        .. code-block:: python

            gateway_inference.get_details()

        """
        all_models = self._gateway.models.get_details()
        matches = [
            m for m in all_models["data"] if self._model in {m["id"], m.get("alias")}
        ]

        if len(matches) == 1:
            return matches[0]
        return matches
