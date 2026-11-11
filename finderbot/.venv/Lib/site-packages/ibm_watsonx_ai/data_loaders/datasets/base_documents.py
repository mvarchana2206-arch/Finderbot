#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025-2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

import logging
import os
from collections import Counter
from copy import copy
from itertools import chain
from random import shuffle
from typing import TYPE_CHECKING, Any, Callable, Iterator
from warnings import warn

import pandas as pd
from langchain_core.documents import Document

from ibm_watsonx_ai.data_loaders.text_loader import (
    _asynch_download,
    _prepare_iterator,
)
from ibm_watsonx_ai.helpers import (
    AssetLocation,
)
from ibm_watsonx_ai.helpers.connections import DataConnection
from ibm_watsonx_ai.helpers.remote_document import RemoteDocument
from ibm_watsonx_ai.utils import get_document_path_from_asset_details
from ibm_watsonx_ai.utils.autoai.enums import DocumentsSamplingTypes, SamplingTypes
from ibm_watsonx_ai.utils.autoai.errors import (
    CannotGetFilename,
    DirectoryHasNoFilename,
    FolderDownloadNotSupported,
    InvalidSizeLimit,
    NoDocumentsLoaded,
)
from ibm_watsonx_ai.wml_client_error import WMLClientError

# Note: try to import torch lib if available, this fallback is done based on
# torch dependency removal request
try:
    from torch.utils.data import IterableDataset

except ImportError:
    IterableDataset: type = object  # type: ignore[no-redef]
# --- end note

if TYPE_CHECKING:
    from pandas import DataFrame

DEFAULT_SAMPLE_SIZE_LIMIT = (
    1073741824  # 1GB in Bytes is verified later by _set_sample_size_limit
)
DEFAULT_SAMPLING_TYPE = SamplingTypes.FIRST_VALUES
DEFAULT_DOCUMENTS_SAMPLING_TYPE = DocumentsSamplingTypes.RANDOM

logger = logging.getLogger(__name__)


class BaseDocumentsIterableDataset(IterableDataset):
    def __init__(
        self,
        *,
        connections: list[DataConnection],
        enable_sampling: bool = True,
        include_subfolders: bool = False,
        sample_size_limit: int = DEFAULT_SAMPLE_SIZE_LIMIT,
        sampling_type: str = DEFAULT_DOCUMENTS_SAMPLING_TYPE,
        total_size_limit: int = DEFAULT_SAMPLE_SIZE_LIMIT,
        total_ndocs_limit: int | None = None,
        benchmark_dataset: pd.DataFrame | None = None,
        error_callback: Callable[[str, Exception], None] | None = None,
        **kwargs: Any,
    ) -> None:
        IterableDataset.__init__(self)
        self.enable_sampling = enable_sampling
        self.sample_size_limit = sample_size_limit
        self.sampling_type = sampling_type
        self._set_size_limit(total_size_limit)
        self.total_ndocs_limit = total_ndocs_limit
        self.benchmark_dataset = benchmark_dataset
        self.error_callback = error_callback

        # Validate benchmark dataset schema if provided
        if benchmark_dataset is not None:
            if "correct_answer_document_ids" not in benchmark_dataset.columns:
                raise WMLClientError(
                    "Invalid benchmark dataset schema: The 'correct_answer_document_ids' column is missing. "
                    "Please ensure your benchmark dataset includes this column with a list of document IDs for each row."
                )

        self._download_strategy = kwargs.get(
            "_download_strategy", "n_parallel"
        )  # expected values: "n_parallel", "sequential"

        api_client = kwargs.get("api_client", kwargs.get("_api_client"))

        self._fill_missing_api_client(connections, api_client)

        data_asset_id_name_mapping = self._build_asset_mapping(connections)

        benchmark_document_ids = self._extract_benchmark_document_ids(benchmark_dataset)

        self.remote_documents = self._build_remote_documents(
            connections,
            include_subfolders,
            data_asset_id_name_mapping,
            benchmark_document_ids,
        )

        self._validate_unique_document_ids(self.remote_documents)

        self.last_exception: Exception | None = None

    @staticmethod
    def _fill_missing_api_client(
        connections: list[DataConnection], api_client: Any
    ) -> None:
        """Fill missing API client in connections that don't have one."""
        if api_client is not None:
            for conn in connections:
                if conn._api_client is None:
                    conn.set_client(api_client)

    @staticmethod
    def _extract_benchmark_document_ids(
        benchmark_dataset: pd.DataFrame | None,
    ) -> list[str]:
        """Extract unique benchmark document IDs from the benchmark dataset.

        :param benchmark_dataset: The benchmark dataset containing correct_answer_document_ids
        :type benchmark_dataset: pd.DataFrame | None

        :return: List of unique benchmark document IDs
        :rtype: list[str]
        """
        if benchmark_dataset is None:
            return []

        try:
            correct_document_ids = benchmark_dataset[
                "correct_answer_document_ids"
            ].values

            # Check for any missing values
            if benchmark_dataset["correct_answer_document_ids"].isna().any():
                missing_indices = benchmark_dataset[
                    benchmark_dataset["correct_answer_document_ids"].isna()
                ].index.tolist()
                raise ValueError(
                    f"Missing 'correct_answer_document_ids' in row(s): {missing_indices}. "
                    "All rows in the benchmark dataset must contain a list of document IDs."
                )

            return list(set(chain.from_iterable(correct_document_ids)))
        except (TypeError, ValueError) as e:
            raise WMLClientError(
                "Invalid benchmark dataset schema: Unable to extract 'correct_answer_document_ids'. "
                f"Each row must contain a list of document IDs. "
                f"Error details: {e}"
            ) from e

    @staticmethod
    def _build_asset_mapping(connections: list[DataConnection]) -> dict[str, str]:
        """Build mapping of asset IDs to filenames."""
        data_asset_id_name_mapping = {}

        if any(isinstance(conn.location, AssetLocation) for conn in connections):
            api_clients = {conn._api_client for conn in connections}
            for client in api_clients:
                for res in client.data_assets.get_details(get_all=True)["resources"]:
                    if (filename := get_document_path_from_asset_details(res)) is None:
                        raise CannotGetFilename()
                    data_asset_id_name_mapping[res["metadata"]["asset_id"]] = filename

        return data_asset_id_name_mapping

    @staticmethod
    def _resolve_document_id(
        conn: DataConnection, data_asset_id_name_mapping: dict[str, str]
    ) -> str:
        """Resolve document ID for a connection using asset mapping or connection's method."""
        if isinstance(conn.location, AssetLocation):
            if conn.location.id in data_asset_id_name_mapping:
                return data_asset_id_name_mapping[conn.location.id]
            raise WMLClientError(
                f"The asset{f' {conn.id}' if hasattr(conn, 'id') and conn.id else ''} with id {conn.location.id} could not be found."
            )
        else:
            try:
                return conn._get_document_id()
            except DirectoryHasNoFilename:
                raise FolderDownloadNotSupported()

    @staticmethod
    def _is_plain_filename(path: str) -> bool:
        """Check if path is a plain filename without path separators.

        :param path: The path to check
        :type path: str

        :return: True if path contains no path separators, False otherwise
        :rtype: bool
        """
        return os.sep not in path and (os.altsep is None or os.altsep not in path)

    def _match_benchmark_id(
        self,
        document_id: str,
        benchmark_document_ids: list[str],
    ) -> str | None:
        """Match a document ID against benchmark IDs.

        Tries exact match first, then falls back to filename matching for backward compatibility.

        :param document_id: The document ID to match
        :type document_id: str

        :param benchmark_document_ids: List of benchmark document IDs
        :type benchmark_document_ids: list[str]

        :return: The matched benchmark ID, or None if no match
        :rtype: str | None
        """
        # Try exact match first (new flow with full paths)
        if document_id in benchmark_document_ids:
            return document_id

        # Fall back to filename matching (old flow - backward compatibility)
        doc_basename = os.path.basename(document_id)
        for benchmark_id in benchmark_document_ids:
            # Check if benchmark_id is just a filename (no path separators)
            if self._is_plain_filename(benchmark_id) and doc_basename == benchmark_id:
                return benchmark_id

        return None

    def _build_remote_documents(
        self,
        connections: list[DataConnection],
        include_subfolders: bool,
        data_asset_id_name_mapping: dict[str, str],
        benchmark_document_ids: list[str],
    ) -> list[RemoteDocument]:
        """Build list of remote documents from connections with benchmark ID matching.

        :param connections: List of data connections
        :type connections: list[DataConnection]

        :param include_subfolders: Whether to include subfolders
        :type include_subfolders: bool

        :param data_asset_id_name_mapping: Mapping of asset IDs to filenames
        :type data_asset_id_name_mapping: dict[str, str]

        :param benchmark_document_ids: List of benchmark document IDs
        :type benchmark_document_ids: list[str]

        :return: List of RemoteDocument instances with benchmark IDs matched
        :rtype: list[RemoteDocument]
        """
        remote_docs = []

        # Track ambiguous matches for warnings
        benchmark_filename_matches: dict[str, list[str]] = {}

        for connection in connections:
            for c in connection._get_all_connections(recursive=include_subfolders):
                document_id = self._resolve_document_id(c, data_asset_id_name_mapping)
                benchmark_id = self._match_benchmark_id(
                    document_id, benchmark_document_ids
                )

                # Track filename matches for ambiguity warnings
                if benchmark_id is not None and self._is_plain_filename(benchmark_id):
                    if benchmark_id not in benchmark_filename_matches:
                        benchmark_filename_matches[benchmark_id] = []
                    benchmark_filename_matches[benchmark_id].append(document_id)

                remote_docs.append(
                    RemoteDocument(
                        connection=c,
                        document_id=document_id,
                        benchmark_id=benchmark_id,
                    )
                )

        # Log warnings for ambiguous filename matches
        for benchmark_id, matched_paths in benchmark_filename_matches.items():
            if len(matched_paths) > 1:
                logger.warning(
                    "Benchmark document ID '%s' matched multiple documents: "
                    "%s. All matching documents will be included as benchmark documents. "
                    "To avoid ambiguity, consider using full paths in benchmark_document_ids.",
                    benchmark_id,
                    matched_paths,
                )

        return remote_docs

    @staticmethod
    def _validate_unique_document_ids(remote_documents: list[RemoteDocument]) -> None:
        """Validate that all document IDs are unique."""
        doc_id_counts = Counter(doc.document_id for doc in remote_documents)
        duplicates = {
            doc_id: count for doc_id, count in doc_id_counts.items() if count > 1
        }

        if duplicates:
            duplicate_details = ", ".join(
                f"'{doc_id}' ({count}x)" for doc_id, count in duplicates.items()
            )
            raise WMLClientError(
                f"Duplicate document identifiers found: {duplicate_details}. "
                "Each document must have a unique identifier."
            )

    def _set_size_limit(self, size_limit: int) -> None:
        """If non-default value of total_size_limit was not passed,
        set Sample Size Limit based on T-Shirt size if code is run on training pod:
        For memory < 16 (T-Shirts: XS,S) default is 10MB,
        For memory < 32 & >= 16 (T-Shirts: M) default is 100MB,
        For memory = 32 (T-Shirt L) default is 0.7GB,
        For memory > 32 (T-Shirt XL) or runs outside pod default is 1GB.
        """
        self.total_size_limit: int | None
        from ibm_watsonx_ai.utils.autoai.connection import get_max_sample_size_limit

        max_tshirt_size_limit = (
            get_max_sample_size_limit() if os.getenv("MEM", False) else None
        )  # limit manual setting of sample size limit on autoai clusters #31527

        if self.enable_sampling:
            if max_tshirt_size_limit:
                if (
                    size_limit > max_tshirt_size_limit
                    and size_limit != DEFAULT_SAMPLE_SIZE_LIMIT
                ):
                    raise InvalidSizeLimit(size_limit, max_tshirt_size_limit)
                self.total_size_limit = min(size_limit, max_tshirt_size_limit)
            else:
                self.total_size_limit = size_limit
        else:
            self.total_size_limit = (
                None if size_limit == DEFAULT_SAMPLE_SIZE_LIMIT else size_limit
            )

    @staticmethod
    def _docs_context_sampling(
        remote_documents: list[RemoteDocument],
    ) -> list[RemoteDocument]:
        """Order documents with benchmark documents first, then non-benchmark documents.

        :param remote_documents: documents to sample from (with _benchmark_id already set)
        :type remote_documents: list[RemoteDocument]

        :return: list of documents ordered with benchmark documents first
        :rtype: list[RemoteDocument]
        """
        benchmark: list[RemoteDocument] = []
        non_benchmark: list[RemoteDocument] = []
        for doc in remote_documents:
            (benchmark if doc._benchmark_id is not None else non_benchmark).append(doc)

        shuffle(benchmark)
        shuffle(non_benchmark)

        return benchmark + non_benchmark

    @staticmethod
    def _docs_random_sampling(
        remote_documents: list[RemoteDocument],
    ) -> list[RemoteDocument]:
        """Randomly sample documents from `remote_documents` up to `size_upper_bound`.

        :param remote_documents: documents to sample from
        :type remote_documents: list[RemoteDocument]

        :return: list of sampled documents
        :rtype: list[RemoteDocument]
        """
        sampling_order = list(range(len(remote_documents)))
        shuffle(sampling_order)

        return [remote_documents[i] for i in sampling_order]

    def _load_doc(self, doc: RemoteDocument) -> Document | Iterator["DataFrame"]:
        raise NotImplementedError()

    def _sequential_gen(
        self, sampled_docs: list[RemoteDocument]
    ) -> Iterator[Document | "DataFrame"]:
        for doc in sampled_docs:
            try:
                loaded_doc = self._load_doc(doc)
                yield from _prepare_iterator(loaded_doc)
            except Exception as e:
                self.last_exception = e
                if self.error_callback:
                    self.error_callback(doc.document_id, e)
                else:
                    raise e

    def _parallel_gen(
        self, sampled_docs: list[RemoteDocument]
    ) -> Iterator[Document | "DataFrame"]:
        import multiprocessing.dummy as mp

        thread_no = min(5, len(sampled_docs))

        q_input: mp.Queue = mp.Queue()
        qs_output: list[mp.Queue] = [mp.Queue() for _ in range(len(sampled_docs))]
        args = [(q_input, qs_output)] * thread_no

        for i, doc in enumerate(sampled_docs):
            q_input.put((i, doc))

        with mp.Pool(thread_no) as pool:
            from queue import Empty

            pool.map_async(_asynch_download(self._load_doc), args)

            for i in range(len(qs_output)):
                while True:
                    try:
                        result = qs_output[i].get(timeout=10 * 60)
                        if isinstance(result, str) and result == "End":
                            break
                    except Empty as e:
                        result = e

                    if isinstance(result, Exception):
                        exc = result
                        self.last_exception = exc
                        doc_id = sampled_docs[i].document_id
                        logger.warning("Failed to download the file: `%s`", doc_id)
                        if not isinstance(exc, Empty):
                            logger.warning(exc)
                        if self.error_callback:
                            self.error_callback(doc_id, exc)
                        break
                    else:
                        yield result

    def _get_element_size(self, el: Any) -> int:
        raise NotImplementedError()

    def _prepare_sampled_docs(self) -> list[RemoteDocument]:
        if self.enable_sampling:
            if self.sampling_type == DocumentsSamplingTypes.RANDOM:
                sampled_docs = self._docs_random_sampling(self.remote_documents)
            elif self.sampling_type == DocumentsSamplingTypes.BENCHMARK_DRIVEN:
                if self.benchmark_dataset is None:
                    raise ValueError(
                        "`benchmark_dataset` is mandatory for sample_type: DocumentsSamplingTypes.BENCHMARK_DRIVEN."
                    )
                sampled_docs = self._docs_context_sampling(self.remote_documents)
            else:
                raise ValueError(
                    f"Unsupported documents sampling type: {self.sampling_type}"
                )
        else:
            sampled_docs = copy(self.remote_documents)

        if (
            self.total_ndocs_limit is not None
            and len(sampled_docs) > self.total_ndocs_limit
        ):
            logger.info(
                "Documents sampled with total_ndocs_limit param, "
                "%s docs chosen from %s possible.",
                len(sampled_docs[: self.total_ndocs_limit]),
                len(sampled_docs),
            )
            sampled_docs = sampled_docs[: self.total_ndocs_limit]

        return sampled_docs

    def __iter__(self) -> Iterator:
        """Iterate over documents."""
        size_limit = (
            self.sample_size_limit
            if self.sample_size_limit is not None and self.enable_sampling
            else self.total_size_limit
        )

        sampled_docs = self._prepare_sampled_docs()

        match self._download_strategy:
            case "n_parallel":  # downloading documents entirely in parallel
                it = self._parallel_gen(sampled_docs)

            case _:  # "sequential" - simple sequential downloading
                it = self._sequential_gen(sampled_docs)

        res_size = 0
        el_no = 0
        for el in it:
            el_no += 1
            res_size += self._get_element_size(el)

            if size_limit is not None and res_size > size_limit:
                return

            yield el

        if el_no == 0:
            no_documents_warning = (
                f"No documents were successfully loaded during the document loading process. "
                f"Use `error_callback` parameter of `{self.__class__.__name__}` class to check the exceptions."
            )
            warn(no_documents_warning)
            raise NoDocumentsLoaded(self.last_exception) from self.last_exception
