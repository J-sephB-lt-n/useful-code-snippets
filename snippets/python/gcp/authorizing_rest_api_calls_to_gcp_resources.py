"""
TAGS: auth|gcp|http|rest
DESCRIPTION: How to authorize HTTP requests to GCP resources (e.g. from within a Virtual Machine)  
REQUIREMENTS: pip install google-auth requests 
"""

import google.auth
import google.auth.transport.requests
import requests

credentials, project_id = google.auth.default()
credentials.refresh(google.auth.transport.requests.Request())
auth_session = requests.Session()
auth_session.headers.update(
    {
        "Authorization": f"Bearer {credentials.token}",
        "Content-Type": "application/json",
    }
)

# now you can do the request #
delete_response = auth_session.delete(
    # example: deleting a VM
    (
        "https://compute.googleapis.com/compute/v1"
        "/projects/your-gcp-project-id"
        "/zones/europe-west12"
        "/instances/vm-name-here"
    )
)
print(delete_response)
