"""
TAGS: auth|authentication|cloud|cloud storage|external|gcloud|gcp|google cloud|outside|storage|write 
DESCRIPTION: Write a file to a google cloud storage bucket from outside of the google cloud environment.
REQUIREMENTS: pip install google-auth google-cloud-storage
NOTES: A more secure approach if authenticating from AWS, Azure, GitHub, Okta, etc. is to use GCP Workload Identity Federation.
"""

import json

from google.cloud import storage
from google.oauth2 import service_account

auth_creds = service_account.Credentials.from_service_account_info(
    {
        # Create this key in IAM > Service Accounts in the gcloud UI #
        "type": "service_account",
        "project-id": "your-gcp-proj-id",
        "private_key_id": "237ghfj57g7h8694jgkbhfj5647fgahvbnfgjoes2",
        "private_key": "-----BEGIN PRIVATE KEY-----\nyouWillSeeAMassiveMultiLinePrivateKeyHere\n-----END PRIVATE KEY-----\n",
        "client_email": "service-acct-name@your-gcp-proj-id.iam.gserviceaccount.com",
        "client_id": "123869420932647",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": "https://cert.url/goes/here",
        "universe_domain": "googleapis.com",
    },
    scopes=["https://www.googleapis.com/auth/devstorage.read_write"],
)

storage_client = storage.Client(project="your-gcp-proj-id", credentials=auth_creds)

file_contents: str = json.dumps({"important": "data"})

bucket = storage_client.bucket("your-gcp-bucket-name")
blob = bucket.blob("folder1/folder2/filename_on_bucket.json")
with blob.open("w") as file:
    file.write(file_contents)
