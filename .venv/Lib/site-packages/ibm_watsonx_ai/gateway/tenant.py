#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from typing import Any, Literal, cast

from ibm_watsonx_ai.client import APIClient
from ibm_watsonx_ai.wml_resource import WMLResource


class TenantConfig(WMLResource):
    """Model Gateway tenant config class."""

    def __init__(self, api_client: APIClient):
        WMLResource.__init__(self, __name__, api_client)

    def get_details(self) -> dict:
        """Get details of currently authenticated tenant's config.

        :returns: tenant config details
        :rtype: dict
        """

        response = self._client.httpx_client.get(
            self._client._href_definitions.get_gateway_tenant_config_href(),
            headers=self._client._get_headers(),
        )

        return self._handle_response(200, "getting tenant config details", response)

    def update(
        self,
        default_algorithm: str | None = None,
        wxai_runtime_instance_id: str | None = None,
        **kwargs: Any,
    ) -> dict:
        """Update details of currently authenticated tenant's config.

        :param default_algorithm: the load balancing algorithm to use for distributing requests
        :type default_algorithm: str, optional

        :param wxai_runtime_instance_id: instance ID of watsonx.ai runtime service
        :type wxai_runtime_instance_id: str, optional

        :returns: tenant config details
        :rtype: dict
        """

        payload = self._drop_nones_from_payload(
            kwargs,
            default_algorithm=default_algorithm,
            wxai_runtime_instance_id=wxai_runtime_instance_id,
        )

        response = self._client.httpx_client.put(
            self._client._href_definitions.get_gateway_tenant_config_href(),
            headers=self._client._get_headers(),
            json=payload,
        )

        return self._handle_response(200, "tenant config update", response)

    def delete(self) -> Literal["SUCCESS"]:
        """Delete currently authenticated tenant's configuration.

        :return: status "SUCCESS" if deletion is successful
        :rtype: Literal["SUCCESS"]
        """

        response = self._client.httpx_client.delete(
            self._client._href_definitions.get_gateway_tenant_config_href(),
            headers=self._client._get_headers(),
        )

        return cast(
            Literal["SUCCESS"],
            self._handle_response(
                204, "tenant config delete", response, json_response=False
            ),
        )


class Tenant(WMLResource):
    """Model Gateway tenant class."""

    def __init__(self, api_client: APIClient):
        WMLResource.__init__(self, __name__, api_client)

        self.config = TenantConfig(api_client)

    def get_details(self) -> dict:
        """Get details of currently authenticated tenant.

        :returns: tenant details
        :rtype: dict
        """

        response = self._client.httpx_client.get(
            self._client._href_definitions.get_gateway_tenant_href(),
            headers=self._client._get_headers(),
        )

        return self._handle_response(200, "getting tenant details", response)

    def update(self, name: str | None = None, **kwargs: Any) -> dict:
        """Update details of currently authenticated tenant.

        :param name: name of the tenant
        :type name: str, optional

        :returns: tenant details
        :rtype: dict
        """

        payload = self._drop_nones_from_payload(kwargs, name=name)

        response = self._client.httpx_client.patch(
            self._client._href_definitions.get_gateway_tenant_href(),
            headers=self._client._get_headers(),
            json=payload,
        )

        return self._handle_response(200, "tenant update", response)

    def replace(self, name: str | None = None, **kwargs: Any) -> dict:
        """Replace details of currently authenticated tenant.

        :param name: name of the tenant
        :type name: str, optional

        :returns: tenant details
        :rtype: dict
        """

        payload = self._drop_nones_from_payload(kwargs, name=name)

        response = self._client.httpx_client.put(
            self._client._href_definitions.get_gateway_tenant_href(),
            headers=self._client._get_headers(),
            json=payload,
        )

        return self._handle_response(200, "tenant replace", response)

    def delete(self) -> Literal["SUCCESS"]:
        """Delete currently authenticated tenant.

        :return: status "SUCCESS" if deletion is successful
        :rtype: Literal["SUCCESS"]
        """

        response = self._client.httpx_client.delete(
            self._client._href_definitions.get_gateway_tenant_href(),
            headers=self._client._get_headers(),
        )

        return cast(
            Literal["SUCCESS"],
            self._handle_response(204, "tenant delete", response, json_response=False),
        )

    @staticmethod
    def get_id(tenant_details: dict) -> str:
        """Get tenant ID from tenant details.

        :param tenant_details: details of the tenant
        :type tenant_details: dict

        :returns: ID of the tenant
        :rtype: str
        """

        return tenant_details["uuid"]
