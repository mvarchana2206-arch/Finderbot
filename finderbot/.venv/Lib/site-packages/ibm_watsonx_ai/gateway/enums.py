#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from ibm_watsonx_ai.utils.utils import StrEnum

__all__ = ["GatewayModelFunctions"]


class GatewayModelFunctions(StrEnum):
    """Allowable values for model functions passed in the ``metadata`` parameter of
    :func:`Gateway.models.create() <ibm_watsonx_ai.gateway.models.Models.create>`.

    Model functions describe the capabilities of a model registered in Model Gateway.
    They are required for AutoAI RAG and other feature-specific workflows that need to
    filter available models by capability.

    Pass one or more values as the ``"functions"`` key inside the ``metadata`` dict:

    .. code-block:: python

        from ibm_watsonx_ai.gateway import GatewayModelFunctions

        gateway.models.create(
            provider_id="...",
            model="my-model",
            metadata={
                "functions": [
                    GatewayModelFunctions.AUTOAI_RAG,
                    GatewayModelFunctions.TEXT_CHAT,
                ],
                "context_window": 8192,
            },
        )

    .. note::
        AutoAI RAG requires at least :attr:`AUTOAI_RAG` (or :attr:`AUTOAI_SQL_RAG`) to
        be present in ``functions``, and a valid ``context_window`` value, so that the
        framework can calculate the maximum number of input tokens for a given model.
    """

    AUTOAI_RAG = "autoai_rag"
    """Model supports AutoAI RAG pipelines."""

    AUTOAI_SQL_RAG = "autoai_sql_rag"
    """Model supports AutoAI SQL RAG pipelines."""

    BASE_FOUNDATION_MODEL_DEPLOYABLE = "base_foundation_model_deployable"
    """Model can be deployed as a base foundation model."""

    EMBEDDING = "embedding"
    """Model produces text embeddings."""

    IMAGE_CHAT = "image_chat"
    """Model supports image-based chat (multimodal)."""

    LORA_FINE_TUNE_TRAINABLE = "lora_fine_tune_trainable"
    """Model can be fine-tuned with LoRA."""

    MULTILINGUAL = "multilingual"
    """Model supports multiple languages."""

    RERANK = "rerank"
    """Model supports document re-ranking."""

    SIMILARITY = "similarity"
    """Model supports semantic similarity scoring."""

    TEXT_CHAT = "text_chat"
    """Model supports chat (text) completions."""

    TEXT_GENERATION = "text_generation"
    """Model supports text generation."""

    TIME_SERIES_FORECAST = "time_series_forecast"
    """Model supports time-series forecasting."""

    VIDEO_CHAT = "video_chat"
    """Model supports video-based chat (multimodal)."""
