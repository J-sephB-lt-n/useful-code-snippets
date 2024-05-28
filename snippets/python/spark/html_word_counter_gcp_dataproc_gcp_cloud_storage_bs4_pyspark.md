```bash
$ tree .
.
├── create_gcp_dataproc_cluster.sh
├── delete_gcp_dataproc_cluster.sh
├── main.py
└── write_output_to_gcp_bigquery_table.sql
```

```bash
SPARK_CLUST_NAME="html-word-counter"
GCP_PROJ_NAME="your_gcloud_project_name"
N_WORKERS=20

source create_gcp_dataproc_cluster.sh
gcloud dataproc jobs submit pyspark \
    main.py \
    --cluster=$SPARK_CLUST_NAME \
    --region='europe-west2' \
    -- \
    --data_input_bucket "gs://your_bucket_name" \
    --data_input_path "input_data" \
    --data_output_bucket "gs://your_bucket_name" \
    --data_output_path "output_data"
source delete_gcp_dataproc_cluster.sh
```
