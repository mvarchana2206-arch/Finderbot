#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from ibm_watsonx_ai.utils.auth.base_auth.base_auth import BaseAuth

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient


def get_auth_method(
    api_client: APIClient,
    on_token_set: Callable[[], None] | None = None,
    on_token_creation: Callable[[], None] | None = None,
    on_token_refresh: Callable[[], None] | None = None,
) -> BaseAuth:
    """
    Return authentication method using values from API client.

    :param api_client: APIClient object
    :type api_client: APIClient

    :param on_token_set: callback which allows to notify about token set
    :type on_token_set: function which takes no params and returns nothing, optional

    :param on_token_creation: callback which allows to notify about token creation
    :type on_token_creation: function which takes no params and returns nothing, optional

    :param on_token_refresh: callback which allows to notify about token refresh
    :type on_token_refresh: function which takes no params and returns nothing, optional

    :returns: authentication method object
    :rtype: BaseAuth
    """
    creds = api_client.credentials

    if (
        creds.token
        and not (creds._is_env_token and (creds.api_key or creds.password))
        and not api_client.credentials.trusted_profile_id
    ):
        from ibm_watsonx_ai.utils.auth.base_auth import TokenAuth

        # situation one of these:
        # - there is token passed by user (and may be password or apikey)
        # - there is token from env and no additional password or api_key in the credentials
        return TokenAuth(
            token=creds.token,
            on_token_set=on_token_set,
        )

    if any(getattr(creds, key, None) for key in ["token_function", "atoken_function"]):
        from ibm_watsonx_ai.utils.auth.jwt_token_function_auth import (
            JWTTokenFunctionAuth,
        )

        # token function passed
        return JWTTokenFunctionAuth(
            api_client,
            on_token_creation=on_token_creation,
            on_token_refresh=on_token_refresh,
        )

    if api_client.ICP_PLATFORM_SPACES:
        from ibm_watsonx_ai.utils.auth.icp_auth import ICPAuth

        # CPD
        return ICPAuth(
            api_client,
            on_token_creation=on_token_creation,
            on_token_refresh=on_token_refresh,
        )

    if isinstance(api_client.credentials.url, str) and (
        "aws" in api_client.credentials.url
        or "ibmforusgov" in api_client.credentials.url
    ):
        from ibm_watsonx_ai.utils.auth.aws_auth import AWSTokenAuth

        # Cloud AWS or GovCloud
        return AWSTokenAuth(
            api_client,
            on_token_creation=on_token_creation,
            on_token_refresh=on_token_refresh,
        )

    if api_client.credentials.trusted_profile_id:
        from ibm_watsonx_ai.utils.auth.trusted_profile_auth import TrustedProfileAuth

        # Cloud with trusted profile
        return TrustedProfileAuth(
            api_client,
            on_token_creation=on_token_creation,
            on_token_refresh=on_token_refresh,
            on_token_set=on_token_set,
        )

    from ibm_watsonx_ai.utils.auth.iam_auth import IAMTokenAuth

    # Cloud
    return IAMTokenAuth(
        api_client,
        on_token_creation=on_token_creation,
        on_token_refresh=on_token_refresh,
    )
