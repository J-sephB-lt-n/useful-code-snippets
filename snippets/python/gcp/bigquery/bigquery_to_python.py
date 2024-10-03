"""
TAGS: bigquery|client|local|dict|download|gcloud|gcp|get|google|google cloud|python|python client|query|read|sql
DESCRIPTION: Run a query on GCP BigQuery from python, getting the result in python memory as a list of dictionaries 
REQUIREMENTS: pip install google-cloud-bigquery 
"""

import google.cloud.bigquery

bigquery_client = google.cloud.bigquery.Client()

query_str: str = """
SELECT      *
FROM        `projectname.datasetname.tablename`
ORDER BY    RAND()
LIMIT       69
;
"""

result_rows: list[dict] = [
    dict(zip(row.keys(), row.values()))
    for row in bigquery_client.query(query_str).result()
]
