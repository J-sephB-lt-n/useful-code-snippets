"""
TAGS: azure|blob|blob storage|cloud|upload|file|microsoft|s3|write|storage|unstructured
DESCRIPTION: Write (upload) contents of object in python memory to file on azure blob storage
REQUIREMENTS: pip install azure-storage-blob
"""

import os
from typing import Optional

from azure.storage.blob import (
    BlobServiceClient,
    BlobClient,
    ContainerClient,
    ContentSettings,
)


def write_blob(
    blob_container_name: str,
    blob_filepath: str,
    data: bytes | str,
    content_type: str,
) -> None:
    """
    Writes contents of python object to file on Azure blob storage

    Args:
        blob_container_name (str): Name of azure blob storage container to write to
        blob_filepath (str): Desired file name and path of blob to write to on azure blob storage
        data (bytes|str): Desired blob contents
        content_type (str): The MIME type of the file e.g. 'text/plain', 'application/json', 'application/xml'
    Example:
        >>> import json
        >>> write_blob(
        ...     blob_container_name="your-blob-storage-container-name",
        ...     blob_filepath="desired/path/to/blob/example.json",
        ...     data=json.dumps({"a":["little", "test"]}),
        ...     content_type="application/json",
        ... )
    """
    blob_storage_account_key: Optional[str] = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
    blob_endpoint: str = os.getenv(
        "AZURE_STORAGE_BLOB_ENDPOINT",
        "http://127.0.0.1:10000/devstoreaccount1/",
    )

    blob_service_client: BlobServiceClient = BlobServiceClient(
        account_url=blob_endpoint,
        credential=blob_storage_account_key,
    )
    blob_container_client: ContainerClient = blob_service_client.get_container_client(
        blob_container_name
    )
    blob_client: BlobClient = blob_container_client.get_blob_client(blob_filepath)
    blob_client.upload_blob(
        data,
        overwrite=True,
        content_settings=ContentSettings(content_type=content_type),
    )
