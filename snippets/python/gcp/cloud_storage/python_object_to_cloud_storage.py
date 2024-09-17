"""
TAGS: cloud storage|gcp|google|google cloud|memory|python|ram|storage|upload|write
DESCRIPTION: Write an object (e.g. list or dict) from python memory into an object in a google cloud storage bucket 
REQUIREMENTS: pip install google-cloud-storage
"""

import json

import google.cloud.storage

gcp_storage_client = (
    google.cloud.storage.Client()
)  # credentials, projectId etc. is read from environment

gcp_storage_bucket = gcp_storage_client.bucket("your-bucket-name")

gcp_storage_bucket.blob("desired/output/path/myfile.json").upload_from_string(
    data=json.dumps({"joe": ["is", "the", "best"]}),
    content_type="application/json",
)
