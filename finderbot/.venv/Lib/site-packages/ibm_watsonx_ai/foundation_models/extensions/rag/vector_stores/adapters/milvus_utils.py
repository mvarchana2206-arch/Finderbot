#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025-2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from typing import TYPE_CHECKING, Any, cast

from langchain_core.embeddings import Embeddings as LCEmbeddings
from langchain_milvus.function import BM25BuiltInFunction
from langchain_milvus.utils.sparse import BaseSparseEmbedding

from ibm_watsonx_ai.foundation_models.embeddings import BaseEmbeddings
from ibm_watsonx_ai.utils.utils import ensure_submodule_available, is_lib_installed
from ibm_watsonx_ai.wml_client_error import MissingExtension

__all__ = [
    "MilvusBM25BuiltinFunction",
    "MilvusSpladeEmbeddingFunction",
    "resolve_index_params",
]

# Default index params for dense (float) vectors — used with COSINE, L2, IP.
# HNSW is not compatible with binary vectors.
DEFAULT_INDEX_PARAM = {
    "metric_type": "COSINE",
    "index_type": "HNSW",
    "params": {"M": 8, "efConstruction": 64},
}

# Base index params for binary vectors — used with HAMMING and JACCARD.
# BIN_IVF_FLAT is the standard Milvus index type for binary vectors.
# `metric_type` is intentionally absent — it is always set explicitly at call site
# via {**DEFAULT_BINARY_INDEX_PARAM, "metric_type": <resolved_metric_type>}.
# `nlist=128` is a conservative IVF cluster count suitable for small/medium collections
# (analogous to M=8, efConstruction=64 in the dense HNSW default above).
DEFAULT_BINARY_INDEX_PARAM = {
    "index_type": "BIN_IVF_FLAT",
    "params": {"nlist": 128},
}

# Mapping from user-facing distance_metric strings to Milvus metric_type values.
# Supported Milvus metric types: COSINE, L2, IP, HAMMING, JACCARD
MILVUS_DISTANCE_METRIC_MAP: dict[str, str] = {
    "cosine": "COSINE",
    "euclidean": "L2",
    "l2": "L2",
    "inner_product": "IP",
    "ip": "IP",
    "hamming": "HAMMING",
    "jaccard": "JACCARD",
}

# Metrics that require binary vector index types (BIN_*).
MILVUS_BINARY_METRICS: frozenset[str] = frozenset({"HAMMING", "JACCARD"})


def resolve_index_params(distance_metric: str | None) -> dict | None:
    """Resolve Milvus ``index_params`` from a user-facing *distance_metric* string.

    Returns a ready-to-use ``index_params`` dict when *distance_metric* is a known
    value, or ``None`` when *distance_metric* is ``None`` or unrecognised (so the
    caller can fall back to a default).

    :param distance_metric: user-facing metric name (case-insensitive), e.g.
        ``"cosine"``, ``"euclidean"``, ``"hamming"``.
    :type distance_metric: str | None
    :return: ``index_params`` dict or ``None``
    :rtype: dict | None
    """
    if distance_metric is None:
        return None
    metric_type = MILVUS_DISTANCE_METRIC_MAP.get(distance_metric.lower())
    if metric_type is None:
        return None
    base = (
        DEFAULT_BINARY_INDEX_PARAM
        if metric_type in MILVUS_BINARY_METRICS
        else DEFAULT_INDEX_PARAM
    )
    return {**base, "metric_type": metric_type}


if TYPE_CHECKING:
    from pymilvus import Function


class _LangchainEmbeddings(LCEmbeddings):
    """Helper class to allow passing `ibm_watsonx_ai.foundation_models.embeddings.BaseEmbeddings` to langchain_milvus"""

    def __init__(self, embeddings: BaseEmbeddings) -> None:
        super().__init__()

        self._embedding_func: BaseEmbeddings = embeddings

    def embed_query(self, text: str) -> list[float]:
        return self._embedding_func.embed_query(text=text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedding_func.embed_documents(texts=texts)

    def to_dict(self) -> dict:
        return self._embedding_func.to_dict()


class MilvusBM25BuiltinFunction(BM25BuiltInFunction):
    """
    Milvus BM25 built-in function.

    Wrapper for `langchain_milvus.BM25BuiltinFunction` that can be used together with MilvusVectorStore in RAGPattern.

    See:
    https://milvus.io/docs/full-text-search.md
    """

    def __init__(
        self,
        input_field_names: str = "text",
        output_field_names: str = "sparse",
        analyzer_params: dict[Any, Any] | None = None,
        enable_match: bool = False,
        function_name: str | None = None,
    ) -> None:
        super().__init__(
            input_field_names=input_field_names,
            output_field_names=output_field_names,
            enable_match=enable_match,
            function_name=function_name,
            analyzer_params=analyzer_params,
        )

    def to_dict(self) -> dict:
        """Serialize ``MilvusBM25BuiltinFunction`` into a dict that allows reconstruction using the ``from_dict`` class method.

        :return: dict for the from_dict initialization
        :rtype: dict
        """

        # Always present as it's included in BM25BuiltInFunction's __init__ method
        function = cast("Function", self._function)

        return {
            "__class__": self.__class__.__name__,
            "__module__": self.__module__,
            "input_field_names": function._input_field_names,  # pylint: disable=protected-access
            "output_field_names": function._output_field_names,  # pylint: disable=protected-access
            "analyzer_params": self.analyzer_params,
            "enable_match": self.enable_match,
            "function_name": function._name,  # pylint: disable=protected-access
        }


class MilvusSpladeEmbeddingFunction(BaseSparseEmbedding, BaseEmbeddings):
    """Sparse embedding model based on SPLADE embedding.

    This class uses the one of the SPLADE model to implement sparse vector embedding.

     .. note::
        This model requires pymilvus[model] to be installed.
        `pip install pymilvus[model]`

    For more information please refer to: https://milvus.io/docs/embed-with-splade.md
    """

    def __init__(
        self, model_name: str = "naver/splade-cocondenser-ensembledistil", **kwargs: Any
    ) -> None:
        if not is_lib_installed(ext := "pymilvus"):
            raise MissingExtension(f"{ext}[model]")

        ensure_submodule_available("pymilvus", "model", extra_hint="model")

        from pymilvus import model

        self._splade_ef = model.sparse.SpladeEmbeddingFunction(
            model_name=model_name, **kwargs
        )

    @staticmethod
    def _sparse_to_dict(sparse_array: Any) -> dict[int, float]:
        """Based on the implementation of `langchain_milvus.utils.sparse.BM25SparseEmbedding._sparse_to_dict"""
        row_indices, col_indices = sparse_array.nonzero()
        non_zero_values = sparse_array.data
        result_dict = {}
        for col_index, value in zip(col_indices, non_zero_values):
            result_dict[col_index] = value
        return result_dict

    def embed_documents(self, texts: list[str]) -> list[dict[int, float]]:  # type: ignore[override]
        """Embed search docs."""
        sparse_arrays = self._splade_ef.encode_documents(texts)
        return [
            MilvusSpladeEmbeddingFunction._sparse_to_dict(sparse_array)
            for sparse_array in sparse_arrays
        ]

    def embed_query(self, query: str) -> dict[int, float]:  # type: ignore[override]
        """Embed query text."""
        return self.embed_documents([query])[0]

    def to_dict(self) -> dict:
        """Serialize ``MilvusSpladeEmbeddingFunction`` into a dict that allows reconstruction using the ``from_dict`` class method.

        :return: dict for the from_dict initialization
        :rtype: dict
        """
        class_data = super().to_dict()

        return class_data | {
            "query_instruction": self._splade_ef.query_instruction,
            "doc_instruction": self._splade_ef.doc_instruction,
            "k_tokens_query": self._splade_ef.k_tokens_query,
            "k_tokens_document": self._splade_ef.k_tokens_document,
            **self._splade_ef._model_config,
        }
