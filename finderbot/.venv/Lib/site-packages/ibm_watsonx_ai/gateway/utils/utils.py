#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
import logging

from ibm_watsonx_ai.gateway import Gateway
from ibm_watsonx_ai.utils import get_from_json
from ibm_watsonx_ai.wml_client_error import WMLClientError
from ibm_watsonx_ai.wml_resource import WMLResource

logger = logging.getLogger(__name__)


def get_max_input_tokens(
    gateway: Gateway, model_id: str, max_completion_tokens: int = 1024
) -> int:
    """Get maximum number of tokens allowed as input for a given model.

    :param gateway: initialized Model Gateway instance
    :type gateway: Gateway

    :param model_id: unique model ID
    :type model_id: str

    :param max_completion_tokens: the maximum number of tokens that can be generated in the chat completion, defaults to 1024
    :type max_completion_tokens: int, optional

    :return: the maximum number of tokens allowed as input for a given model
    :rtype: int
    """
    model_details = gateway.models.get_details(model_id=model_id)
    model_context_window = get_from_json(model_details, ["metadata", "context_window"])

    if model_context_window is None:
        error_msg = (
            f"Maximum input tokens for the model id `{model_id}` cannot be calculated"
        )
        reason_msg = "The `context_window` cannot be found in the model metadata"
        raise WMLClientError(error_msg=error_msg, reason=reason_msg)

    return model_context_window - max_completion_tokens


# This is kept here for backward compatibility, the function was defined here
# first and then moved to `WMLResource` to make usage easier across all modules.
# This export will be removed in SDK version 2.0
drop_nones = WMLResource._drop_nones_from_payload  # pylint: disable=protected-access
