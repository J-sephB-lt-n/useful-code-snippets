"""
TAGS: bucket|cloud|cloud storage|download|fetch|gcp|get|gcloud|google|google cloud|local|memory|object|python|ram|storage
DESCRIPTION: Download files from a google cloud storage bucket into python memory
REQUIREMENTS: pip install google-cloud-storage
"""

import json
import shutil
import time
from pathlib import Path
from typing import Any, Final

import google.cloud.storage
import google.cloud.storage.transfer_manager

gcp_storage_client = google.cloud.storage.Client()
bucket = gcp_storage_client.bucket("your-bucket-name")

#################
# A single file #
#################
OBJECT_FILEPATH: Final[str] = "path/to/your/file.json"
with bucket.blob(OBJECT_FILEPATH).open("r") as file:
    file_contents: dict = json.load(file)

#########################
# Many files (slow way) #
#########################
# I observed this to download 9 files per second (each file ~3KB) - tested in cloud shell
time_start: float = time.perf_counter()
downloaded_files: dict[str, Any] = {}
for blob in bucket.list_blobs():
    downloaded_files[blob.name] = json.loads(blob.download_as_bytes())
time_end: float = time.perf_counter()
print(
    f"""
Downloaded {len(downloaded_files):,} files in {(time_end-time_start):,.1f} seconds 
    """
)

###########################
# Many files (faster way) #
###########################
# I observed this to download 142 files per second (each file ~3KB) - tested in cloud shell
time_start: float = time.perf_counter()
downloaded_files: dict[str, Any] = {}
temp_storage_dir = Path("temp_storage")
temp_storage_dir.mkdir(exist_ok=True)
for file in temp_storage_dir.glob("*"):
    if file.is_file():
        file.unlink()
download_results = google.cloud.storage.transfer_manager.download_many_to_path(
    bucket,
    [blob.name for blob in bucket.list_blobs()],
    destination_directory=str(temp_storage_dir),
    max_workers=50,
)
for filepath in temp_storage_dir.glob("*.json"):
    with open(filepath, "r", encoding="utf-8") as file:
        downloaded_files[filepath.name] = json.load(file)
shutil.rmtree(temp_storage_dir)
time_end: float = time.perf_counter()
print(
    f"""
Processed {len(download_results):,} files in {(time_end-time_start):,.1f} seconds 
There were {sum([result is not None for result in download_results]):,} errors
    """
)
