"""
TAGS: bucket|cloud|cloud storage|download|fetch|gcp|get|gcloud|google|google cloud|local|memory|object|python|ram|storage
DESCRIPTION: Read a file from a google cloud storage bucket into python memory
REQUIREMENTS: pip install google-cloud-storage
"""

import json
from typing import Final

import google.cloud.storage

BUCKET_NAME: Final[str] = "your-bucket-name"
OBJECT_FILEPATH: Final[str] = "path/to/your/file.json"

gcp_storage_client = google.cloud.storage.Client()

bucket = gcp_storage_client.bucket(BUCKET_NAME)

with bucket.blob(OBJECT_FILEPATH).open("r") as file:
    file_contents: dict = json.load(file)
