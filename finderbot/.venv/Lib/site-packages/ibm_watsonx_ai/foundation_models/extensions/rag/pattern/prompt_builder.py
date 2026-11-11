#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024-2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from collections import defaultdict
from typing import Any, cast

from ibm_watsonx_ai.foundation_models.extensions.rag import RAGPattern
from ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores.langchain_vector_store_adapter import (
    merge_metadata,
)
from ibm_watsonx_ai.utils.utils import is_lib_installed
from ibm_watsonx_ai.wml_client_error import MissingExtension

if not is_lib_installed(ext := "langchain-core"):
    raise MissingExtension(ext, extra_info="rag")
from langchain_core.documents import Document


class Chunk:
    """
    An indexed piece of text (a.k.a "chunk").

    Allows access to the text original position within the containing document.
    """

    def __init__(self, document: Document, retrieval_rank: int):
        """A retrieved chunk, with its underlying Document, retrieval rank, and the inputs (question and reference documents).

        :param document: A document retrieved from an index, representing a chunk of text. Includes
        a text string as well as metadata with the position of the chunk with a full document.
        :type document: Document

        :param retrieval_rank: A zero-based rank of the chunk within the retrieval results (0 means the chunk
        was rank first in the retrieval results).
        :type retrieval_rank: int

        """
        self.document = document
        self.retrieval_rank = retrieval_rank

    @property
    def document_id(self) -> str:
        return self.document.metadata["document_id"]

    @property
    def sequence_number(self) -> str:
        return self.document.metadata["sequence_number"]

    @property
    def start_index(self) -> int:
        return self.document.metadata["start_index"]

    @property
    def end_index(self) -> int:
        return self.start_index + len(self.document.page_content)

    def is_followed_by(self, chunk_to_check: "Chunk") -> bool:
        """
        Return true if `chunk_to_check` directly follows this chunk, with or without
        intersection.

        Example:
            If the text is "This is a sentence",
            - "is" follows "This i" (with a non-empty intersection "i")
            - "is" follows "This " (without any intersection)
            - "is" does NOT follow "This" (note the space)
        """
        return self.start_index <= chunk_to_check.start_index <= self.end_index

    def merge_with_following(self, following_chunk: "Chunk") -> "Chunk":
        """
        Merge a chunk with a chunk that follows it.

        A following chunk is:
        - from the same document as self.
        - directly follows self, with or without an intersection.
        """
        assert self.document_id == following_chunk.document_id

        intersection_size = self.end_index - following_chunk.start_index
        merged_text = (
            self.document.page_content
            + following_chunk.document.page_content[intersection_size:]
        )

        merged_doc = Document(page_content=merged_text)
        merged_doc.metadata = merge_metadata(
            [self.document.metadata, following_chunk.document.metadata]
        )

        # Explicitly override position properties, so their values are not lists
        merged_doc.metadata["document_id"] = self.document_id
        merged_doc.metadata["sequence_number"] = self.sequence_number
        merged_doc.metadata["start_index"] = self.start_index

        return Chunk(
            document=merged_doc,
            retrieval_rank=min(self.retrieval_rank, following_chunk.retrieval_rank),
        )


# Defaults
WORD_TO_TOKEN_RATIO = 1.5


def estimate_tokens_count(
    text: str, word_to_token_ratio: float = WORD_TO_TOKEN_RATIO
) -> int:
    """Estimate the number of tokens in a given text.
    The token count is estimated using the number of words in the input text
    times a fixed factor estimating the number of tokens in a single word.

    :param text: the text to count the tokens for
    :type text: str

    :param word_to_token_ratio: Constant representing the average number of tokens per word in a text, used for
        approximating the token count, defaults to 1.5
    :type word_to_token_ratio: float, optional

    :return: the count of the tokens in the text
    :rtype: int
    """
    words = text.split()
    return int(len(words) * word_to_token_ratio)


def merge_overlapping_chunks(documents: list[Document]) -> list[Document]:
    chunks = [
        Chunk(reference_document, retrieval_position)
        for retrieval_position, reference_document in enumerate(documents)
    ]

    grouped_by_documents = defaultdict(list)
    for chunk in chunks:
        grouped_by_documents[chunk.document_id].append(chunk)

    merged_chunks = []
    for document_id in grouped_by_documents.keys():
        chunks_in_document = grouped_by_documents[document_id]
        sorted_chunks_by_sequence_number = sorted(
            chunks_in_document, key=lambda chunk: chunk.sequence_number
        )

        current_chunk = sorted_chunks_by_sequence_number[0]
        for next_chunk in sorted_chunks_by_sequence_number[1:]:
            if current_chunk.is_followed_by(next_chunk):
                current_chunk = current_chunk.merge_with_following(next_chunk)
            else:
                merged_chunks.append(current_chunk)
                current_chunk = next_chunk
        merged_chunks.append(current_chunk)

    # Sort by the original retrieval position
    merged_chunks = sorted(merged_chunks, key=lambda chunk: chunk.retrieval_rank)
    return [chunk.document for chunk in merged_chunks]


def build_prompt(
    prompt_template_text: str,
    context_template_text: str,
    question: str,
    reference_documents: list[Document] | list[str],
    model_max_input_tokens: int,
    word_to_token_ratio: float = WORD_TO_TOKEN_RATIO,
    **kwargs: Any,
) -> str:
    """Build the input prompt from the prompt and context templates, and the inputs (question and reference documents).

    :param prompt_template_text: the text of the prompt template, used to create the RAG prompt
    :type prompt_template_text: str

    :param context_template_text: the text of the context template, used to format each reference document.
    :type context_template_text: str

    :param question: the question text that is to be part of the prompt
    :type question: str

    :param reference_documents: all the reference documents that are to be considered as part of the prompt.
        If the there are too many documents, or they are too long, the last documents will be omitted.
    :type reference_documents: list[str]

    :param model_max_input_tokens: the maximum number of input tokens supported by the model.
    :type model_max_input_tokens: int

    :param word_to_token_ratio: Constant representing the average number of tokens per word in a text, used for
        approximating the token count, defaults to 1.5
    :type word_to_token_ratio: float, optional

    :param system_prompt_text: the text of the system prompt that is used - only applicable for chat scenario, defaults to None
    :type system_prompt_text: str | None

    :return: the constructed prompt containing the instruction and model inputs (question and reference documents).
        The prompt length is under the maximal number of input tokens supported by the model (model_max_input_tokens).
        The prompt may contain only a subset of the reference documents, due to the limited input length.
    :rtype: str
    """

    if isinstance(reference_documents, list) and all(
        isinstance(ref, str) for ref in reference_documents
    ):
        reference_documents_texts = cast(list[str], reference_documents)
    else:
        reference_documents = cast(list[Document], reference_documents)
        reference_documents = merge_overlapping_chunks(reference_documents)
        reference_documents_texts = [doc.page_content for doc in reference_documents]

    system_prompt_text = kwargs.pop("system_prompt_text", None)
    if context_template_text:
        reference_documents_texts = [
            context_template_text.format(document=reference_document)
            for reference_document in reference_documents_texts
        ]

    selected_reference_documents = _select_reference_documents(
        prompt_template_text=prompt_template_text,
        question=question,
        reference_documents=reference_documents_texts,
        model_max_input_tokens=model_max_input_tokens,
        word_to_token_ratio=word_to_token_ratio,
        system_prompt_text=system_prompt_text,
    )

    prompt_variables = {
        "question": question,
        "reference_documents": "\n".join(selected_reference_documents),
    }
    return prompt_template_text.format(**prompt_variables)


def _select_reference_documents(
    prompt_template_text: str,
    question: str,
    reference_documents: list[str],
    model_max_input_tokens: int,
    word_to_token_ratio: float = WORD_TO_TOKEN_RATIO,
    system_prompt_text: str | None = None,
) -> list[str]:
    """Select reference documents according to maximal number of input tokens supported by the model.
    Only using these selected references ensures that the constructed prompt fits (in terms of length) properly
    into the input window supported by the model.

    :param prompt_template_text: the text of the prompt template, used to create the RAG prompt
    :type prompt_template_text: str

    :param question: the question text that is to be part of the prompt
    :type question: str

    :param reference_documents: all the reference documents that are to be considered as part of the prompt.
        If the there are too many documents, or they are too long, the last documents will be omitted.
    :type reference_documents: list[str]

    :param model_max_input_tokens: the maximum number of input tokens supported by the model.
    :type model_max_input_tokens: int

    :param word_to_token_ratio: Constant representing the average number of tokens per word in a text, used for
        approximating the token count, defaults to 1.5
    :type word_to_token_ratio: float, optional

    :param system_prompt_text: the text of the system prompt that is used - only applicable for chat scenario, defaults to None
    :type system_prompt_text: str | None

    :return: the reference documents that may be integrated into the prompt template, while maintaining
        the constraint on the model input window size.
    :rtype: list[str]
    """
    # The number of input tokens available after taking into account the prompt template
    # and the question
    available_input_tokens = (
        model_max_input_tokens
        - estimate_tokens_count(prompt_template_text, word_to_token_ratio)
        - estimate_tokens_count(question, word_to_token_ratio)
        # the placeholders will not be in the final prompt, so their token counts
        # should not be subtracted. count their token counts as available tokens.
        + estimate_tokens_count(RAGPattern.QUESTION_PLACEHOLDER, word_to_token_ratio)
        + estimate_tokens_count(
            RAGPattern.REFERENCE_DOCUMENTS_PLACEHOLDER, word_to_token_ratio
        )
    )

    if system_prompt_text is not None:
        available_input_tokens -= estimate_tokens_count(
            system_prompt_text, word_to_token_ratio
        )

    selected_reference_documents = []
    for reference_document in reference_documents:
        # Select the current reference document if there are enough
        # available input tokens. Add +1 for the newline separator, used for
        # joining reference documents.
        tokens_required_for_reference_document = (
            estimate_tokens_count(reference_document, word_to_token_ratio) + 1
        )
        if tokens_required_for_reference_document <= available_input_tokens:
            available_input_tokens -= tokens_required_for_reference_document
            selected_reference_documents.append(reference_document)
        else:
            break

    return selected_reference_documents
