LOAD DATA OVERWRITE `yourProjectName.yourDatasetName.yourTableName`
CLUSTER BY word
FROM FILES (
      format = 'PARQUET',
      uris = ['gs://your_bucket_name/output_data/*.parquet']
)
;
