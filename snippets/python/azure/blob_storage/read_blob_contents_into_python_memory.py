"""
TAGS: azure|blob|blob storage|cloud|download|file|microsoft|s3|read|storage|unstructured
DESCRIPTION: Download contents of file stored on azure blob storage into python memory
REQUIREMENTS: pip install azure-storage-blob
"""

import os

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient


def read_blob(
    blob_container_name: str,
    blob_filepath: str,
) -> str:
    """
    Reads contents of file stored on Azure blob storage \
    into python memory
    """
    blob_storage_account_key: str = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
    blob_endpoint: str = os.getenv(
        "AZURE_STORAGE_BLOB_ENDPOINT",
        "http://127.0.0.1:10000/devstoreaccount1/",
    )

    blob_service_client = BlobServiceClient(
        account_url=blob_endpoint,
        credential=blob_storage_account_key,
    )
    blob_container_client: ContainerClient = blob_service_client.get_container_client(
        blob_container_name
    )
    blob_client: BlobClient = blob_container_client.get_blob_client(blob_filepath)
    download_stream = blob_client.download_blob()

    return download_stream.readall().decode("utf-8")
