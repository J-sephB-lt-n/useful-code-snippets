```
TAGS: analytics|backend|bigquery|cloud|data|database|data warehouse|db|duckdb|gcloud|gcp|google|google cloud|google bigquery|local|parquet|query|sql|sqlite|storage
DESCRIPTION: Copy a BigQuery table into local parquet files and query the parquet files directly using DuckDB
REQUIREMENTS: pip install duckdb
NOTES: For my use-case (bigquery table with 8.5 million rows and 58 columns), DuckDB was 118x faster than BigQuery for the same query (0.00532 seconds vs 0.628 seconds)
```

```sql
--  this took 15 seconds and resulted in 27 parquet files
EXPORT DATA
  OPTIONS (
      uri = 'gs://your-storage-bucket-name/some-folder/*.parquet'
    , format = 'Parquet'
    , overwrite = true
)
AS (
  SELECT      *
  FROM        `gcp-project-name.bigquery-dataset-name.bigquery-table-name`
  ORDER BY    RecordSource
            , CompanyName
);
```

```shell
-- the 27 parquet files took 15 seconds to download (in cloud shell)
-- downloaded files came to 1.3GiB in total
mkdir data
start_time=$(date +%s)
gcloud storage cp 'gs://your-storage-bucket-name/some-folder/*.parquet' data/
end_time=$(date +%s)
echo "Finished copy in "$((end_time - start_time))" seconds"
```

```shell
-- in cloud shell
python -m venv venv
source venv/bin/activate
pip install duckdb
```

```python
# I ran this code in ipython in the cloud shell #
import time
import duckdb

start_time: float = time.perf_counter()
result = duckdb.sql(
  """
SELECT    --something--
FROM      'data/*.parquet'
WHERE     --some condition--
AND       --some other condition--
ORDER BY  --something--
;
  """
)
end_time: float = time.perf_counter()
print(f"query took {(end_time-start_time):,.5f} seconds querying parquet files directly from DuckDB")
# query took 0.00532 seconds seconds querying parquet files directly from DuckDB
# for reference, this same query took BigQuery 0.628 seconds (118x slower than DuckDB)
```
