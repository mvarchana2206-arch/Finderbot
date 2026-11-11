#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import json as js
from functools import wraps
from typing import (
    Any,
    Callable,
    MutableMapping,
    ParamSpec,
    Sequence,
    TypeGuard,
    TypeVar,
    cast,
)

import httpx
from httpx._types import HeaderTypes, RequestContent

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")

CONTENT_TYPE = "Content-Type"
APPLICATION_JSON = "application/json"


def _is_headers_mutable_mapping(
    headers: httpx._types.HeaderTypes, content_type: type[T]
) -> TypeGuard[MutableMapping[T, T]]:
    if not isinstance(headers, MutableMapping):
        return False

    return all(
        isinstance(k, content_type) and isinstance(v, content_type)
        for k, v in headers.items()
    )


def _is_headers_sequence(
    headers: httpx._types.HeaderTypes, content_type: type[T]
) -> TypeGuard[Sequence[tuple[T, T]]]:
    if not isinstance(headers, Sequence):
        return False

    return all(
        isinstance(k, content_type) and isinstance(v, content_type) for k, v in headers
    )


def _update_value_for_header_sequence(
    headers: Sequence[tuple[T, T]], key: T, new_value: T
) -> Sequence[tuple[T, T]]:
    new_headers: list[tuple[T, T]] = []

    for item in headers:
        if item[0] == key:
            new_headers.append((key, new_value))
        else:
            new_headers.append(item)

    return new_headers


def _update_content_type_to_json(
    headers: httpx._types.HeaderTypes | None,
) -> httpx._types.HeaderTypes | None:
    if not headers:
        return None

    if (
        _is_headers_mutable_mapping(headers, str) or isinstance(headers, httpx.Headers)
    ) and headers.get(CONTENT_TYPE) is not None:
        headers[CONTENT_TYPE] = APPLICATION_JSON
    elif (
        _is_headers_mutable_mapping(headers, bytes)
        and headers.get(CONTENT_TYPE.encode()) is not None
    ):
        headers[CONTENT_TYPE.encode()] = APPLICATION_JSON.encode()
    elif _is_headers_sequence(headers, str):
        headers = _update_value_for_header_sequence(
            headers, CONTENT_TYPE, APPLICATION_JSON
        )
    elif _is_headers_sequence(headers, bytes):
        headers = _update_value_for_header_sequence(
            headers, CONTENT_TYPE.encode(), APPLICATION_JSON.encode()
        )

    return headers


def encode_json_payload(func: Callable[P, R]) -> Callable[P, R]:
    """
    Converts numpy structures provided in `json` parameter
    into Python-native types before sending the request.
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        content = cast(RequestContent | None, kwargs.get("content"))
        json = cast(Any | None, kwargs.get("json"))
        headers = cast(HeaderTypes | None, kwargs.get("headers"))

        if json is not None and content is None:
            from ibm_watsonx_ai.utils.utils import NumpyTypeEncoder

            del kwargs["json"]
            kwargs["content"] = js.dumps(json, cls=NumpyTypeEncoder)
            kwargs["headers"] = _update_content_type_to_json(headers)

        return func(*args, **kwargs)

    return wrapper
