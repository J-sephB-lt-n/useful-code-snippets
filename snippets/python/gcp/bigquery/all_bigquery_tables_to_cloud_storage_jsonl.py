"""
TAGS: all|bigquery|big query|cloud|cloud storage|database|dataset|download|dump|files|gcp|google|google cloud|json|jsonl|migrate|read|sql|storage|tables|transfer
DESCRIPTION: Dumps each table within a Google BigQuery dataset into a .jsonl file on cloud storage 
"""

# pip install google-cloud-bigquery
from typing import Final

from google.cloud import bigquery

GCP_PROJECT_NAME: Final[str] = "fill this in"
GCP_LOCATION: Final[str] = "europe-west2"
BIGQUERY_DATASET_NAME: Final[str] = "fill this in"
OUTPUT_BUCKET_PATH: Final[str] = "gs://your_bucket_name_here"

bigquery_client = bigquery.Client()

all_table_names: tuple[str] = tuple(
    [
        row.values()[0]
        for row in bigquery_client.query(
            f"""SELECT  DISTINCT table_name 
                FROM `{GCP_PROJECT_NAME}.{BIGQUERY_DATASET_NAME}.INFORMATION_SCHEMA.TABLES`
                 ;
            """
        ).result()
    ]
)

bigquery_job_config = bigquery.job.ExtractJobConfig()
bigquery_job_config.destination_format = (
    bigquery.DestinationFormat.NEWLINE_DELIMITED_JSON
)
errors_history = []
success_count: int = 0
for table_name in all_table_names:
    try:
        bigquery_extract_job = bigquery_client.extract_table(
            bigquery.DatasetReference(
                GCP_PROJECT_NAME,
                BIGQUERY_DATASET_NAME,
            ).table(table_name),
            f"{OUTPUT_BUCKET_PATH}/{table_name}.jsonl",
            job_config=bigquery_job_config,
            location=GCP_LOCATION,
        )
        bigquery_extract_job.result()
        print(f"Completed: [{table_name}]")
        success_count += 1
    except Exception as err:
        errors_history.append({"table_name": table_name, "error_string": str(err)})
        print(f"!ERROR! extracting table {table_name}. Error was:\n{err}")

print("\n --SUMMARY-- \n")
print(f"{success_count} tables successfully extracted")
print(f"{len(errors_history)} tables failed to extract")
