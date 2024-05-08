"""
TAGS: bigquery|big query|cloud|database|download|file|gcp|google|google cloud|local|json|jsonl|read|sql
DESCRIPTION: Runs a query on Google BigQuery and saves the result to a local newline-delimited JSON file.
"""

# pip install google-cloud-bigquery

from typing import Final
from google.cloud import bigquery

GCP_PROJECT_NAME: Final[str] = "you fill this in"
GCP_LOCATION: Final[str] = "europe-west2"
BIGQUERY_DATASET_NAME: Final[str] = "you fill this in"
BIGQUERY_TABLE_NAME: Final[str] = "you fill this in"
OUTPUT_FILENAME: Final[str] = f"{BIGQUERY_TABLE_NAME}.jsonl" 
INTERIM_OUTPUT_GCP_BUCKET_NAME: Final[str] = "gs://put_your_bucket_name_here"
INTERIM_OUTPUT_GCP_BUCKET_FILEPATH: Final[str] = (
    f"{INTERIM_OUTPUT_GCP_BUCKET_NAME}/{OUTPUT_FILENAME}"
)

bigquery_client = bigquery.Client()

bigquery_job_config = bigquery.job.ExtractJobConfig()
bigquery_job_config.destination_format = (
    bigquery.DestinationFormat.NEWLINE_DELIMITED_JSON
)

bigquery_extract_job = bigquery_client.extract_table(
    bigquery.DatasetReference(
        GCP_PROJECT_NAME,
        BIGQUERY_DATASET_NAME,
    ).table(BIGQUERY_TABLE_NAME),
    INTERIM_OUTPUT_GCP_BUCKET_FILEPATH,
    job_config=bigquery_job_config,
    location=GCP_LOCATION,
)
bigquery_extract_job.result()

# TODO: file download using google cloud storage python API #
