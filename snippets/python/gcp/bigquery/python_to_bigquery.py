"""
TAGS: bigquery|cloud|gcloud|gcp|google|google cloud|python|write
DESCRIPTION: Write data (a list of dicts) from python memory into a BigQuery table (including code for creating or emptying the table) 
REQUIREMENTS: pip install google-cloud-bigquery
NOTE: inserts exceeding 10MB are rejected (break into multiple insert requests)
"""

import datetime
import warnings
from typing import Final

import google.api_core.exceptions
import google.cloud.bigquery

CREATE_TABLE: Final[bool] = True  # create table if it does not already exist
DELETE_EXISTING_TABLE_DATA: Final[bool] = True  # if table exists, delete all data in it

GCP_PROJECT_ID: Final[str] = "your-gcp-project-id"
BIGQUERY_DATASET_NAME: Final[str] = "your-dataset-name"
BIGQUERY_TABLE_NAME: Final[str] = "your-table-name"

BIGQUERY_TABLE_ID: Final[str] = (
    f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET_NAME}.{BIGQUERY_TABLE_NAME}"
)

bigquery_client = google.cloud.bigquery.Client()

schema = [
    google.cloud.bigquery.SchemaField("datetime_utc", "TIMESTAMP", mode="REQUIRED"),
    google.cloud.bigquery.SchemaField(
        "transaction_description", "STRING", mode="NULLABLE"
    ),
    google.cloud.bigquery.SchemaField("amount", "NUMERIC", mode="REQUIRED"),
]

bigquery_table = google.cloud.bigquery.Table(
    BIGQUERY_TABLE_ID,
    schema=schema,
)

if CREATE_TABLE:
    try:
        bigquery_client.create_table(bigquery_table)
        print(f"Created bigquery table '{BIGQUERY_TABLE_ID}'")
    except google.api_core.exceptions.Conflict:
        warnings.warn(f"Table already exists '{BIGQUERY_TABLE_ID}'", UserWarning)

if DELETE_EXISTING_TABLE_DATA:
    _ = bigquery_client.query(f"DELETE FROM `{BIGQUERY_TABLE_ID}` WHERE TRUE;").result()
    print(f"Deleted existing data from '{BIGQUERY_TABLE_ID}'")

rows_to_insert: list[dict] = [
    {
        "datetime_utc": datetime.datetime(
            year=2024,
            month=7,
            day=24,
            hour=10,
            minute=6,
            second=49,
            tzinfo=datetime.timezone.utc,
        ).isoformat(),
        "transaction_description": "the pub",
        "amount": "-420.69",
    },
    {
        "datetime_utc": datetime.datetime(
            year=2024,
            month=7,
            day=24,
            hour=11,
            minute=0,
            second=0,
            tzinfo=datetime.timezone.utc,
        ).isoformat(),
        "transaction_description": "petrol station",
        "amount": "-800.85",
    },
]

row_insert_errors: list = bigquery_client.insert_rows_json(
    BIGQUERY_TABLE_ID,
    rows_to_insert,
    row_ids=[None] * len(rows_to_insert),
)
if len(row_insert_errors) > 0:
    print("received the following insertion errors:")
    for error in row_insert_errors:
        print(error)
