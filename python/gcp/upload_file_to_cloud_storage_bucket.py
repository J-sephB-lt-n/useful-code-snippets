"""
TAGS: {bucket|cloud|gcp|google|google cloud|object|storage|upload|write}

Function which writes a file from the local filesystem to a google cloud storage bucket
"""

import google.cloud.storage

gcp_storage_client = google.cloud.storage.Client()
gcp_storage_bucket = gcp_storage_client.get_bucket("my-gcp-bucket-name")


def upload_file_to_cloud_storage(
    local_file_path: str,
    gcp_file_path: str,
    bucket: google.cloud.storage.bucket.Bucket = gcp_storage_bucket,
) -> None:
    """Uploads a file from the local filesystem to a GCP cloud storage bucket

    Args:
        local_file_path (str): Path to file on local filesystem
        gcp_file_path (str): Path to upload to on bucket
        bucket (google.cloud.storage.bucket.Bucket): Python object representation of the GCP bucket

    Returns:
        None: On successful upload, returns nothing
            (if unsuccessful, a python exception is raised)

    Example:
    >>> upload_file_to_cloud_storage(
    ...     local_file_path = "./output/summary_of_prev_period.pdf",
    ...     gcp_file_path = "daily_summary_pdfs/2069_07_24.pdf",
    ... )
    """
    blob = bucket.blob(gcp_file_path)
    blob.upload_from_filename(local_file_path)
