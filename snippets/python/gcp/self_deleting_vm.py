"""
TAGS: compute|compute engine|delete|deleting|gcp|google|google cloud|kill|killing|myself|self|suicide|virtual machine|vm
DESCRIPTION: Running this sript on a Google Cloud Virtual Machine (VM) will make it delete itself 
REQUIREMENTS: (if pip available) pip install google-auth requests
REQUIREMENTS: (on ubuntu, no pip available) apt install python3-pip python3-requests python3-google-auth -y
NOTES: The Virtual Machine (VM) must have Compute Engine read/write permission in order to delete itself
"""

import logging

import google.auth
import google.auth.transport.requests
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_vm_metadata(vm_attr: str):
    """Fetches desired VM attribute value from the internal metadata server
    (GCP project ID, zone, VM ID, VM name etc.)
    """
    metadata_url = f"http://metadata.google.internal/computeMetadata/v1/{vm_attr}"
    response = requests.get(metadata_url, headers={"Metadata-Flavor": "Google"})
    if response.status_code == 200:
        return response.text
    else:
        raise Exception(
            f"[{response.status_code}] Failed to get {vm_attr} from internal metadata server"
        )


zone: str = get_vm_metadata("instance/zone").split("/")[-1]
instance_name: str = get_vm_metadata("instance/name")

credentials, project_id = google.auth.default()
credentials.refresh(google.auth.transport.requests.Request())
auth_session = requests.Session()
auth_session.headers.update(
    {"Authorization": f"Bearer {credentials.token}", "Content-Type": "application/json"}
)

logger.info(
    "compute engine VM [%s] on project [%s] in zone [%s] requesting to delete itself",
    instance_name,
    zone,
    project_id,
)
delete_response = auth_session.delete(
    (
        "https://compute.googleapis.com/compute/v1"
        f"/projects/{project_id}"
        f"/zones/{zone}"
        f"/instances/{instance_name}"
    )
)
if delete_response.status_code == 200:
    logger.info(
        "compute engine VM [%s] on project [%s] in zone [%s] is being deleted",
        instance_name,
        zone,
        project_id,
    )
else:
    logger.warning(
        "compute engine VM [%s] on project [%s] in zone [%s] failed to delete itself (received response [%s])",
        instance_name,
        zone,
        project_id,
        delete_response.status_code,
    )
