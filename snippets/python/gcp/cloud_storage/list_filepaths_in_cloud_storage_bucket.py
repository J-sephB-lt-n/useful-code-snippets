"""
TAGS: blob|blobs|bucket|cloud|cloud storage|filepath|filepaths|gcloud|gcp|google|list|path|paths|prefix|storage
DESCRIPTION: Lists all filepaths in the given google cloud storage bucket (possibly filtered by filepath prefix)
REQUIREMENTS: pip install google-cloud-storage
"""

from typing import Final, Optional

import google.cloud.storage

BUCKET_NAME: Final[str] = "your-bucket-name"
FILEPATH_PREFIX: Optional[str] = None  # e.g. "dir1/dir2/"

storage_client = google.cloud.storage.Client()
bucket = storage_client.get_bucket(BUCKET_NAME)
blobs = bucket.list_blobs(prefix=FILEPATH_PREFIX)
for blob in blobs:
    print(
        blob.name
    )  # blob.name is the full filepath on the bucket (e.g. "dir1/dir2/dir3/filename.json")
