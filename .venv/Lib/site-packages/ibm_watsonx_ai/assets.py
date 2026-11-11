#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias, cast
from warnings import warn

from httpx import Response

from ibm_watsonx_ai.messages.messages import Messages
from ibm_watsonx_ai.metanames import AssetsMetaNames
from ibm_watsonx_ai.utils import DATA_ASSETS_DETAILS_TYPE
from ibm_watsonx_ai.utils.enums import AssetDuplicateAction
from ibm_watsonx_ai.utils.utils import AsyncFileReader, _get_id_from_deprecated_uid
from ibm_watsonx_ai.wml_client_error import ApiRequestFailure, WMLClientError
from ibm_watsonx_ai.wml_resource import WMLResource

if TYPE_CHECKING:
    from pandas import DataFrame

    from ibm_watsonx_ai import APIClient

ListType: TypeAlias = list


class Assets(WMLResource):
    """Store and manage data assets."""

    ConfigurationMetaNames = AssetsMetaNames()
    """MetaNames for Data Assets creation."""

    def __init__(self, client: APIClient) -> None:
        WMLResource.__init__(self, __name__, client)

    def get_details(
        self,
        asset_id: str | None = None,
        get_all: bool | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> dict:
        """Get data asset details. If no asset_id is passed, details for all assets are returned.

        :param asset_id: unique ID of the asset
        :type asset_id: str, optional

        :param limit:  limit number of fetched records
        :type limit: int, optional

        :param get_all:  if True, it will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :return: metadata of the stored data asset
        :rtype: dict

        **Example:**

        .. code-block:: python

            asset_details = client.data_assets.get_details(asset_id)

        """
        asset_id = _get_id_from_deprecated_uid(
            kwargs, asset_id, "asset", can_be_none=True
        )

        return self._get_asset_based_resource(
            asset_id,
            "data_asset",
            self._get_required_element_from_response,
            limit=limit,
            get_all=get_all,
        )

    async def aget_details(
        self,
        asset_id: str | None = None,
        get_all: bool | None = None,
        limit: int | None = None,
    ) -> dict:
        """Get data asset details asynchronously. If no asset_id is passed, details for all assets are returned.

        :param asset_id: unique ID of the asset
        :type asset_id: str, optional

        :param limit:  limit number of fetched records
        :type limit: int, optional

        :param get_all:  if True, it will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :return: metadata of the stored data asset
        :rtype: dict

        **Example:**

        .. code-block:: python

            asset_details = await client.data_assets.aget_details(asset_id)

        """
        return await self._aget_asset_based_resource(
            asset_id,
            "data_asset",
            self._get_required_element_from_response,
            limit=limit,
            get_all=get_all,
        )

    @staticmethod
    def _prepare_file_path(file_path: str | Path) -> Path:
        if isinstance(file_path, str):
            return Path(file_path)
        return file_path

    @staticmethod
    def _extract_enum_value(value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value
        return value

    @staticmethod
    def _validate_and_prepare_create_inputs(
        name: str,
        file_path: str | Path,
        duplicate_action: AssetDuplicateAction | None = None,
    ) -> tuple[str, Path, str | None]:
        Assets._validate_type(name, "name", str, True)
        Assets._validate_type(file_path, "file_path", [str, Path], True, True)

        prepared_path = Assets._prepare_file_path(file_path)
        duplicate_action_value = Assets._extract_enum_value(duplicate_action)

        return name, prepared_path, duplicate_action_value

    def create(
        self,
        name: str,
        file_path: str | Path,
        duplicate_action: AssetDuplicateAction | None = None,
    ) -> dict[str, Any]:
        """Create a data asset and upload content to it.

        :param name: name to be given to the data asset
        :type name: str

        :param file_path: path to the content file to be uploaded
        :type file_path: str | Path

        :param duplicate_action: determines behaviour when asset with the same name already exists,
            if not specified, the value from catalogs/projects/spaces will be used
        :type duplicate_action: AssetDuplicateAction, optional

        :return: metadata of the stored data asset
        :rtype: dict

        **Example:**

        .. code-block:: python

            asset_details = client.data_assets.create(
                name="sample_asset", file_path="/path/to/file"
            )

        """
        # quick support for COS credentials instead of local path
        # TODO add error handling and cleaning (remove the file)
        name, prepared_path, duplicate_action_value = (
            self._validate_and_prepare_create_inputs(name, file_path, duplicate_action)
        )

        return self._create_asset(
            name, prepared_path, duplicate_action=duplicate_action_value
        )

    async def acreate(
        self,
        name: str,
        file_path: str | Path,
        duplicate_action: AssetDuplicateAction | None = None,
    ) -> dict[str, Any]:
        """Create a data asset and upload content to it asynchronously.

        :param name: name to be given to the data asset
        :type name: str

        :param file_path: path to the content file to be uploaded
        :type file_path: str

        :param duplicate_action: determines behaviour when asset with the same name already exists,
            if not specified, the value from catalogs/projects/spaces will be used
        :type duplicate_action: AssetDuplicateAction, optional

        :return: metadata of the stored data asset
        :rtype: dict

        **Example:**

        .. code-block:: python

            asset_details = await client.data_assets.acreate(
                name="sample_asset", file_path="/path/to/file"
            )

        """
        # quick support for COS credentials instead of local path
        # TODO add error handling and cleaning (remove the file)
        name, prepared_path, duplicate_action_value = (
            self._validate_and_prepare_create_inputs(name, file_path, duplicate_action)
        )

        return await self._acreate_asset(
            name, prepared_path, duplicate_action=duplicate_action_value
        )

    def store(self, meta_props: dict) -> dict[str, Any]:
        """Create a data asset and upload content to it.

        :param meta_props: metadata of the space configuration. To see available meta names, use:

            .. code-block:: python

                client.data_assets.ConfigurationMetaNames.get()

        :type meta_props: dict

        **Example:**

        Example of data asset creation for files:

        .. code-block:: python

            metadata = {
                client.data_assets.ConfigurationMetaNames.NAME: "my data assets",
                client.data_assets.ConfigurationMetaNames.DESCRIPTION: "sample description",
                client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: "sample.csv",
            }
            asset_details = client.data_assets.store(meta_props=metadata)

        Example of data asset creation using a connection:

        .. code-block:: python

            metadata = {
                client.data_assets.ConfigurationMetaNames.NAME: "my data assets",
                client.data_assets.ConfigurationMetaNames.DESCRIPTION: "sample description",
                client.data_assets.ConfigurationMetaNames.CONNECTION_ID: "39eaa1ee-9aa4-4651-b8fe-95d3ddae",
                client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: "t1/sample.csv",
            }
            asset_details = client.data_assets.store(meta_props=metadata)

        Example of data asset creation with a database sources type connection:

        .. code-block:: python

            metadata = {
                client.data_assets.ConfigurationMetaNames.NAME: "my data assets",
                client.data_assets.ConfigurationMetaNames.DESCRIPTION: "sample description",
                client.data_assets.ConfigurationMetaNames.CONNECTION_ID: "23eaf1ee-96a4-4651-b8fe-95d3dadfe",
                client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: "t1",
            }
            asset_details = client.data_assets.store(meta_props=metadata)

        """
        name, file_path, description, connection_id, duplicate_action = (
            self._get_asset_details_from_metaprops(meta_props)
        )
        duplicate_action_value = self._extract_enum_value(duplicate_action)

        return self._create_asset(
            name,
            file_path,
            connection_id=connection_id,
            description=description,
            duplicate_action=duplicate_action_value,
        )

    async def astore(self, meta_props: dict) -> dict[str, Any]:
        """Create a data asset and upload content to it asynchronously.

        :param meta_props: metadata of the space configuration. To see available meta names, use:

            .. code-block:: python

                client.data_assets.ConfigurationMetaNames.get()

        :type meta_props: dict

        **Example:**

        Example of data asset creation for files:

        .. code-block:: python

            metadata = {
                client.data_assets.ConfigurationMetaNames.NAME: "my data assets",
                client.data_assets.ConfigurationMetaNames.DESCRIPTION: "sample description",
                client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: "sample.csv",
            }
            asset_details = await client.data_assets.astore(meta_props=metadata)

        Example of data asset creation using a connection:

        .. code-block:: python

            metadata = {
                client.data_assets.ConfigurationMetaNames.NAME: "my data assets",
                client.data_assets.ConfigurationMetaNames.DESCRIPTION: "sample description",
                client.data_assets.ConfigurationMetaNames.CONNECTION_ID: "39eaa1ee-9aa4-4651-b8fe-95d3ddae",
                client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: "t1/sample.csv",
            }
            asset_details = await client.data_assets.astore(meta_props=metadata)

        Example of data asset creation with a database sources type connection:

        .. code-block:: python

            metadata = {
                client.data_assets.ConfigurationMetaNames.NAME: "my data assets",
                client.data_assets.ConfigurationMetaNames.DESCRIPTION: "sample description",
                client.data_assets.ConfigurationMetaNames.CONNECTION_ID: "23eaf1ee-96a4-4651-b8fe-95d3dadfe",
                client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: "t1",
            }
            asset_details = await client.data_assets.astore(meta_props=metadata)

        """
        name, file_path, description, connection_id, duplicate_action = (
            self._get_asset_details_from_metaprops(meta_props)
        )
        duplicate_action_value = self._extract_enum_value(duplicate_action)

        return await self._acreate_asset(
            name,
            file_path,
            connection_id=connection_id,
            description=description,
            duplicate_action=duplicate_action_value,
        )

    def _get_asset_details_from_metaprops(
        self, meta_props: dict
    ) -> tuple[str, Path, str, str | None, str | None]:
        """Retrieves properties from meta_props for store and astore method returning them as a tuple in following order:
        name, file_path, description, connection_id, duplicate_action
        """
        Assets._validate_type(meta_props, "meta_props", dict, True)

        name = meta_props[self.ConfigurationMetaNames.NAME]
        file_path = Path(meta_props[self.ConfigurationMetaNames.DATA_CONTENT_NAME])
        description = ""

        connection_id = meta_props.get(self.ConfigurationMetaNames.CONNECTION_ID)

        if not connection_id and not file_path.is_file():
            no_connection_id_specified_warning = (
                f"No connection_id specified and file: {str(file_path)} does not exist."
            )
            warn(no_connection_id_specified_warning)

        if self.ConfigurationMetaNames.DESCRIPTION in meta_props:
            description = meta_props[self.ConfigurationMetaNames.DESCRIPTION]

        duplicate_action = meta_props.pop(
            self.ConfigurationMetaNames.DUPLICATE_ACTION, None
        )
        return name, file_path, description, connection_id, duplicate_action

    @staticmethod
    def _get_mime_type(file_path: Path) -> str:
        try:
            import mimetypes
        except Exception as e:
            raise WMLClientError(
                Messages.get_message(message_id="module_mimetypes_not_found"), str(e)
            )

        mime_type = mimetypes.MimeTypes().guess_type(file_path)[0]
        if mime_type is None:
            mime_type = "application/octet-stream"

        return mime_type

    @staticmethod
    def _build_asset_metadata(
        name: str, description: str | None, mime_type: str, connection_id: str | None
    ) -> dict[str, Any]:
        desc = description if description is not None else ""

        asset_meta: dict[str, Any] = {
            "metadata": {
                "name": name,
                "description": desc,
                "asset_type": "data_asset",
                "origin_country": "us",
                "asset_category": "USER",
            },
            "entity": {"data_asset": {"mime_type": mime_type}},
        }

        if connection_id is not None:
            asset_meta["metadata"]["tags"] = ["connected-data"]

        return asset_meta

    def _prepare_params_with_duplicate_action(
        self, duplicate_action: str | None
    ) -> dict:
        params = self._client._params()

        if duplicate_action:
            params["duplicate_action"] = duplicate_action

        return params

    @staticmethod
    def _build_attachment_metadata(
        file_path: Path, mime_type: str, connection_id: str | None
    ) -> dict[str, Any]:
        attachment_meta: dict[str, Any] = {
            "asset_type": "data_asset",
            "name": file_path.name,
            "mime": mime_type,
        }

        if connection_id is not None:
            attachment_meta.update(
                {
                    "connection_id": connection_id,
                    "connection_path": str(file_path),
                    "is_remote": True,
                }
            )

        return attachment_meta

    @staticmethod
    def _validate_file_not_empty(file_path: Path) -> None:
        if file_path.stat().st_size == 0:
            raise WMLClientError(
                Messages.get_message(message_id="cannot_create_empty_asset")
            )

    def _upload_attachment(
        self,
        asset_id: str,
        name: str,
        file_path: Path,
        connection_id: str | None = None,
    ) -> None:

        mime_type = self._get_mime_type(file_path)

        attachment_meta = self._build_attachment_metadata(
            file_path, mime_type, connection_id
        )

        # Create attachment
        attachment_response = self._client.httpx_client.post(
            url=self._client._href_definitions.get_attachments_href(asset_id),
            headers=self._client._get_headers(),
            params=self._client._params(),
            json=attachment_meta,
        )

        attachment_details = self._handle_response(
            201, "creating new attachment", attachment_response
        )
        if connection_id is not None:
            return

        attachment_id = attachment_details["attachment_id"]
        attachment_url = attachment_details["url1"]

        self._validate_file_not_empty(file_path)

        # Put content to attachment
        with file_path.open("rb") as _file:
            if not self._client.ICP_PLATFORM_SPACES:
                put_response = self._client.httpx_client.put(
                    url=attachment_url, content=_file
                )
            else:
                put_response = self._client.httpx_client.put(
                    url=self._credentials.url + attachment_url,
                    files={"file": (name, _file, "file")},
                )

        if put_response.status_code not in {200, 201}:
            self._handle_response(200, "upload attachment", put_response)

        # Complete attachment
        complete_response = self._client.httpx_client.post(
            url=self._client._href_definitions.get_attachment_complete_href(
                asset_id, attachment_id
            ),
            headers=self._client._get_headers(),
            params=self._client._params(),
        )

        self._handle_response(200, "complete attachment", complete_response)

    async def _aupload_attachment(
        self,
        asset_id: str,
        name: str,
        file_path: Path,
        connection_id: str | None = None,
    ) -> None:

        mime_type = self._get_mime_type(file_path)

        attachment_meta = self._build_attachment_metadata(
            file_path, mime_type, connection_id
        )

        # Create attachment
        attachment_response = await self._client.async_httpx_client.post(
            url=self._client._href_definitions.get_attachments_href(asset_id),
            headers=await self._client._aget_headers(),
            params=self._client._params(),
            json=attachment_meta,
        )

        attachment_details = self._handle_response(
            201, "creating new attachment", attachment_response
        )
        if connection_id is not None:
            return

        attachment_id = attachment_details["attachment_id"]
        attachment_url = attachment_details["url1"]

        self._validate_file_not_empty(file_path)

        # Put content to attachment
        if not self._client.ICP_PLATFORM_SPACES:
            put_response = await self._client.async_httpx_client.put(
                url=attachment_url, content=AsyncFileReader(file_path)
            )
        else:
            with file_path.open("rb") as _file:
                put_response = await self._client.async_httpx_client.put(
                    url=self._credentials.url + attachment_url,
                    files={"file": (name, _file, "file")},
                )

        if put_response.status_code not in {200, 201}:
            self._handle_response(200, "upload attachment", put_response)

        # Complete attachment
        complete_response = await self._client.async_httpx_client.post(
            url=self._client._href_definitions.get_attachment_complete_href(
                asset_id, attachment_id
            ),
            headers=await self._client._aget_headers(),
            params=self._client._params(),
        )

        self._handle_response(200, "complete attachment", complete_response)

    def _create_asset(
        self,
        name: str,
        file_path: Path,
        connection_id: str | None = None,
        description: str | None = None,
        duplicate_action: str | None = None,
    ) -> dict:
        mime_type = self._get_mime_type(file_path)

        asset_meta = self._build_asset_metadata(
            name, description, mime_type, connection_id
        )

        params = self._prepare_params_with_duplicate_action(duplicate_action)

        # Create an asset
        print(Messages.get_message(message_id="creating_data_asset"))

        creation_response = self._client.httpx_client.post(
            url=self._client._href_definitions.get_data_assets_href(),
            headers=self._client._get_headers(),
            params=params,
            json=asset_meta,
        )

        asset_details = self._handle_response(
            201, "creating new asset", creation_response
        )
        # Create attachment

        asset_id = asset_details["metadata"]["asset_id"]

        try:
            self._upload_attachment(asset_id, name, file_path, connection_id)
        except Exception as e:
            try:
                self.delete(asset_id)
            except Exception:
                pass

            if isinstance(e, ApiRequestFailure) or not isinstance(e, WMLClientError):
                raise WMLClientError(
                    Messages.get_message(
                        message_id="failed_while_creating_a_data_asset"
                    )
                ) from e
            else:
                raise e

        print(Messages.get_message(message_id="success"))
        return self._get_required_element_from_response(asset_details)

    async def _acreate_asset(
        self,
        name: str,
        file_path: Path,
        connection_id: str | None = None,
        description: str | None = None,
        duplicate_action: str | None = None,
    ) -> dict:
        mime_type = self._get_mime_type(file_path)

        asset_meta = self._build_asset_metadata(
            name, description, mime_type, connection_id
        )

        params = self._prepare_params_with_duplicate_action(duplicate_action)

        # Create an asset
        print(Messages.get_message(message_id="creating_data_asset"))

        creation_response = await self._client.async_httpx_client.post(
            url=self._client._href_definitions.get_data_assets_href(),
            headers=await self._client._aget_headers(),
            params=params,
            json=asset_meta,
        )

        asset_details = self._handle_response(
            201, "creating new asset", creation_response
        )
        # Create attachment

        asset_id = asset_details["metadata"]["asset_id"]

        try:
            await self._aupload_attachment(asset_id, name, file_path, connection_id)
        except Exception as e:
            try:
                await self.adelete(asset_id)
            except Exception:
                pass

            if isinstance(e, ApiRequestFailure) or not isinstance(e, WMLClientError):
                raise WMLClientError(
                    Messages.get_message(
                        message_id="failed_while_creating_a_data_asset"
                    )
                ) from e
            else:
                raise e

        print(Messages.get_message(message_id="success"))
        return self._get_required_element_from_response(asset_details)

    def list(self, limit: int | None = None) -> DataFrame:
        """Lists stored data assets in a table format.

        :param limit: limit number for fetched records
        :type limit: int, optional

        :rtype: DataFrame
        :return: listed elements

        **Example:**

        .. code-block:: python

            client.data_assets.list()

        """
        return self._list_asset_based_resource(
            url=self._client._href_definitions.get_search_data_asset_href(),
            column_names=["NAME", "ASSET_TYPE", "SIZE", "ASSET_ID"],
            limit=limit,
        )

    @staticmethod
    def _write_content_to_file(content: bytes, filename: Path) -> str:
        try:
            filename.write_bytes(content)
            print(
                Messages.get_message(
                    filename, message_id="successfully_saved_data_asset_content_to_file"
                )
            )
            return str(filename.resolve())
        except IOError as e:
            raise WMLClientError(
                Messages.get_message(
                    filename, message_id="saving_data_asset_to_local_file_failed"
                ),
                str(e),
            )

    def _determine_final_downloaded_attachment_name(
        self,
        filename: str | Path | None,
        attachment_name: str | None,
        index_of_attachment: int,
    ) -> Path:
        final_filename = filename

        if filename is None and attachment_name is None:
            raise TypeError(
                "Missing positional argument 'filename' while attachment details not containing filename."
            )
        elif filename and index_of_attachment > 0:
            final_filename = self._prepare_file_path(filename)
            final_filename = (
                f"{final_filename.stem}_{index_of_attachment}{final_filename.suffix}"
            )

        resolved = final_filename if final_filename else attachment_name
        if resolved is None:
            raise ValueError(
                "Cannot determine a filename: 'filename' was not provided and the attachment details contain no filename."
            )
        return self._prepare_file_path(resolved)

    def download(
        self,
        asset_id: str | None = None,  # asset_id is optional for backward compatibility,
        filename: str | Path | None = None,
        get_all: bool = False,
        **kwargs: Any,
    ) -> str | ListType[str]:
        """Download and store the content of a data asset. Only first attachment on list will be downloaded by default.

        If `get_all` is set to True, the function will attempt to download all existing attachments
        and will return list of filenames of downloaded attachments. The name will be created from passed `filename`,
        or if no `filename` will be passed, taken from attachment details.

        :param asset_id: unique ID of the data asset to be downloaded
        :type asset_id: str

        :param filename: filename to be used for the downloaded file
        :type filename: str | Path, optional

        :param get_all: all attachment will be downloaded, and list of filenames will be returned, defaults to False
        :type get_all: bool, optional

        :return: normalized path to the downloaded asset content or list of paths if ``get_all`` is True
        :rtype: str | list[str]

        **Example:**

        .. code-block:: python

            client.data_assets.download(asset_id, "sample_asset.csv")

        """
        asset_id = _get_id_from_deprecated_uid(kwargs, asset_id, "asset")
        Assets._validate_type(asset_id, "asset_id", str, True)

        if not get_all:
            if filename is None:
                raise TypeError("Missing required positional argument 'filename'")

            return self._write_content_to_file(
                cast(bytes, self.get_content(asset_id)),
                self._prepare_file_path(filename),
            )

        final_filenames: list[str] = []
        for index, att_details in enumerate(
            cast(list, self.get_content(asset_id, list_format=True))
        ):
            attachment_name, content = att_details
            final_filename = self._determine_final_downloaded_attachment_name(
                filename, attachment_name, index
            )
            final_filenames.append(self._write_content_to_file(content, final_filename))

        return final_filenames

    async def adownload(self, asset_id: str, filename: str | Path) -> str:
        """Download and store the content of a data asset asynchronously.

        :param asset_id: unique ID of the data asset to be downloaded
        :type asset_id: str

        :param filename: filename to be used for the downloaded file
        :type filename: str | Path

        :return: normalized path to the downloaded asset content
        :rtype: str

        **Example:**

        .. code-block:: python

            await client.data_assets.adownload(asset_id, "sample_asset.csv")

        """
        prepared_filename = self._prepare_file_path(filename)

        content = await self.aget_content(asset_id)
        return self._write_content_to_file(content, prepared_filename)

    @staticmethod
    def _validate_attachment_response(response: Response) -> None:
        if response.status_code != 200:
            raise WMLClientError(
                Messages.get_message(message_id="failure_during_downloading_data_asset")
            )

    @staticmethod
    def _validate_final_download_response(att_response: Response) -> bytes:
        if att_response.status_code != 200:
            raise ApiRequestFailure(
                Messages.get_message(
                    message_id="failure_during_downloading_data_asset"
                ),
                att_response,
            )

        return att_response.content

    def get_content(
        self,
        asset_id: str | None = None,  # asset_id is optional for backward compatibility
        list_format: bool = False,
        **kwargs: Any,
    ) -> bytes | ListType[tuple[str, bytes]]:
        """Download the content of a data asset. If there will be more than one attachment, the function will attempt to download all of them.

        :param asset_id: unique ID of the data asset to be downloaded
        :type asset_id: str

        :param list_format: determines if the result is returned in list format with attachment name and content, defaults to False
        :type: bool, optional

        :return: the asset content or list with attachment name and content if ``list_format`` is True
        :rtype: bytes | list[tuple[str, bytes]]

        **Example:**

        .. code-block:: python

            content = client.data_assets.get_content(asset_id).decode("ascii")

        """
        asset_id = _get_id_from_deprecated_uid(kwargs, asset_id, "asset")

        Assets._validate_type(asset_id, "asset_id", str, True)

        asset_response = self._client.httpx_client.get(
            url=self._client._href_definitions.get_data_asset_href(asset_id),
            params=self._client._params(),
            headers=self._client._get_headers(),
        )
        asset_details = self._handle_response(200, "get assets", asset_response)
        errors: list[Exception] = []
        result: list[tuple[str, bytes]] = []

        for index, attachment_details in enumerate(asset_details["attachments"]):
            try:
                result.append(
                    self._get_attachment_with_file_name(
                        asset_id, attachment_details["id"], asset_details
                    )
                )

                if not list_format:  # return first content right after it is downloaded
                    return result[0][1]
            except Exception as e:
                errors.append(e)
                download_asset_warning = f"Downloading the attachment {attachment_details['id']} failed with error: {e}"
                warn(download_asset_warning)

        # if all downloads fail, throw an error
        if len(errors) == len(asset_details["attachments"]) and len(errors) > 0:
            if len(errors) == 1:
                raise errors[0]
            raise ExceptionGroup("All attachment downloads failed", errors)

        return result

    def _get_attachment_with_file_name(
        self, asset_id: str, attachment_id: str, asset_details: dict
    ) -> tuple[str, bytes]:
        response = self._client.httpx_client.get(
            url=self._client._href_definitions.get_attachment_href(
                asset_id, attachment_id
            ),
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        self._validate_attachment_response(response)
        attachment_details = response.json()
        attachment_name = attachment_details.get("name")

        if (
            "connection_id" in asset_details["attachments"][0]
            and asset_details["attachments"][0]["connection_id"] is not None
        ):
            conn_details = self._client.connections.get_details(
                asset_details["attachments"][0]["connection_id"]
            )
            attachment_data_source_type = conn_details["entity"].get("datasource_type")
            cos_conn_data_source_id = (
                self._client.connections.get_datasource_type_id_by_name(
                    "cloudobjectstorage"
                )
            )
            if attachment_data_source_type == cos_conn_data_source_id:
                attachment_signed_url = attachment_details["url"]
                att_response = self._client.httpx_client.get(url=attachment_signed_url)
            else:
                raise WMLClientError(
                    Messages.get_message(
                        message_id="download_api_not_supported_for_this_connection_type"
                    )
                )
        else:
            attachment_signed_url = response.json()["url"]
            if self._client.CLOUD_PLATFORM_SPACES:
                att_response = self._client.httpx_client.get(url=attachment_signed_url)
            else:
                att_response = self._client.httpx_client.get(
                    url=self._credentials.url + attachment_signed_url
                )

        return attachment_name, self._validate_final_download_response(att_response)

    async def aget_content(self, asset_id: str) -> bytes:
        """Download the content of a data asset asynchronously.

        :param asset_id: unique ID of the data asset to be downloaded
        :type asset_id: str

        :return: the asset content
        :rtype: bytes

        **Example:**

        .. code-block:: python

            content = (await client.data_assets.aget_content(asset_id)).decode(
                "ascii"
            )

        """

        Assets._validate_type(asset_id, "asset_id", str, True)

        asset_response = await self._client.async_httpx_client.get(
            url=self._client._href_definitions.get_data_asset_href(asset_id),
            params=self._client._params(),
            headers=await self._client._aget_headers(),
        )
        asset_details = self._handle_response(200, "get assets", asset_response)

        attachment_id = asset_details["attachments"][0]["id"]
        response = await self._client.async_httpx_client.get(
            url=self._client._href_definitions.get_attachment_href(
                asset_id, attachment_id
            ),
            params=self._client._params(),
            headers=await self._client._aget_headers(),
        )

        self._validate_attachment_response(response)

        if (
            "connection_id" in asset_details["attachments"][0]
            and asset_details["attachments"][0]["connection_id"] is not None
        ):
            conn_details = await self._client.connections.aget_details(
                asset_details["attachments"][0]["connection_id"]
            )
            attachment_data_source_type = conn_details["entity"].get("datasource_type")
            cos_conn_data_source_id = await (
                self._client.connections.aget_datasource_type_id_by_name(
                    "cloudobjectstorage"
                )
            )
            if attachment_data_source_type == cos_conn_data_source_id:
                attachment_signed_url = response.json()["url"]
                att_response = await self._client.async_httpx_client.get(
                    url=attachment_signed_url
                )
            else:
                raise WMLClientError(
                    Messages.get_message(
                        message_id="download_api_not_supported_for_this_connection_type"
                    )
                )
        else:
            attachment_signed_url = response.json()["url"]
            if self._client.CLOUD_PLATFORM_SPACES:
                att_response = await self._client.async_httpx_client.get(
                    url=attachment_signed_url
                )
            else:
                att_response = await self._client.async_httpx_client.get(
                    url=self._credentials.url + attachment_signed_url
                )

        return self._validate_final_download_response(att_response)

    @staticmethod
    def get_id(asset_details: dict) -> str:
        """Get the unique ID of a stored data asset.

        :param asset_details: details of the stored data asset
        :type asset_details: dict

        :return: unique ID of the stored data asset
        :rtype: str

        **Example:**

        .. code-block:: python

            asset_id = client.data_assets.get_id(asset_details)

        """
        Assets._validate_type(asset_details, "asset_details", object, True)
        Assets._validate_type_of_details(asset_details, DATA_ASSETS_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(
            asset_details, "data_assets_details", ["metadata", "guid"], str
        )

    @staticmethod
    def get_href(asset_details: dict) -> str:
        """Get the URL of a stored data asset.

        :param asset_details: details of the stored data asset
        :type asset_details: dict

        :return: href of the stored data asset
        :rtype: str

        **Example:**

        .. code-block:: python

            asset_details = client.data_assets.get_details(asset_id)
            asset_href = client.data_assets.get_href(asset_details)

        """
        Assets._validate_type(asset_details, "asset_details", object, True)
        Assets._validate_type_of_details(asset_details, DATA_ASSETS_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(
            asset_details, "asset_details", ["metadata", "href"], str
        )

    def delete(
        self,
        asset_id: str | None = None,  # asset_id is optional for backward compatibility
        purge_on_delete: bool | None = None,
        **kwargs: Any,
    ) -> dict | str:
        """Soft delete the stored data asset. The asset will be moved to trashed assets
        and will not be visible in asset list. To permanently delete assets set `purge_on_delete` parameter to True.

        :param asset_id: unique ID of the data asset
        :type asset_id: str

        :param purge_on_delete: if set to True will purge the asset
        :type purge_on_delete: bool, optional

        :return: status "SUCCESS" or dictionary, if deleted asynchronously
        :rtype: str or dict

        **Example:**

        .. code-block:: python

            client.data_assets.delete(asset_id)

        """
        asset_id = cast(str, asset_id)  # TODO remove when v2

        return self._delete_asset_based_resource(
            asset_id,
            self._get_required_element_from_response,
            purge_on_delete,
            **kwargs,
        )

    async def adelete(
        self,
        asset_id: str,
        purge_on_delete: bool | None = None,
    ) -> dict | str:
        """Soft delete the stored data asset asynchronously. The asset will be moved to trashed assets
        and will not be visible in asset list. To permanently delete assets set `purge_on_delete` parameter to True.

        :param asset_id: unique ID of the data asset
        :type asset_id: str

        :param purge_on_delete: if set to True will purge the asset
        :type purge_on_delete: bool, optional

        :return: status "SUCCESS" or dictionary, if deleted asynchronously
        :rtype: str or dict

        **Example:**

        .. code-block:: python

            await client.data_assets.adelete(asset_id)

        """
        return await self._adelete_asset_based_resource(
            asset_id,
            self._get_required_element_from_response,
            purge_on_delete,
        )

    def _get_required_element_from_response(self, response_data: dict) -> dict:
        WMLResource._validate_type(response_data, "data assets response", dict)

        import copy

        new_el = {"metadata": copy.copy(response_data["metadata"])}

        try:
            new_el["metadata"]["guid"] = response_data["metadata"]["asset_id"]
            new_el["metadata"]["href"] = response_data["href"]

            new_el["metadata"]["asset_type"] = response_data["metadata"]["asset_type"]
            new_el["metadata"]["created_at"] = response_data["metadata"]["created_at"]
            new_el["metadata"]["last_updated_at"] = response_data["metadata"][
                "usage"
            ].get("last_updated_at")

            if self._client.default_space_id is not None:
                new_el["metadata"]["space_id"] = response_data["metadata"]["space_id"]

            elif self._client.default_project_id is not None:
                new_el["metadata"]["project_id"] = response_data["metadata"][
                    "project_id"
                ]

            if "entity" in response_data:
                new_el["entity"] = response_data["entity"]

            if "attachments" in response_data and response_data["attachments"]:
                req_attachment_props = {"id", "name"}
                attachment_details = response_data["attachments"][0]

                new_el["metadata"].update(
                    {
                        f"attachment_{prop}": attachment_details[prop]
                        for prop in req_attachment_props
                        if prop in attachment_details
                    }
                )

                new_el["attachments"] = response_data["attachments"]

            href_without_host = response_data["href"].split(".com")[-1]
            new_el["metadata"]["href"] = href_without_host

            return new_el
        except Exception:
            raise WMLClientError(
                Messages.get_message(
                    response_data,
                    message_id="failed_to_read_response_from_down_stream_service",
                )
            )
