```
NOTES: 10,000 html files were processed, each on average 1Mb (i.e. 10Gb of total input data)
NOTES: At time of writing (May 2024), the cost of the 50 worker VMs + 1 manager VM is $10.2 per hour ($0.17 per minute)   
NOTES: The spark job took 69 seconds to complete (i.e. processes 145 files per second = 145Mb of data processed per second)
NOTES: The majority of the job time (57 of the 69 seconds = 83%) was reading and writing data to/from GCP Cloud Storage (full breakdown later in this doc)
NOTES: The entire process of cluster creation, job run and cluster deletion took 4.94 minutes
NOTES: The result BigQuery table contained 10,520,000 rows
```

```bash
$ tree .
.
├── code_section_timer.py
├── create_gcp_dataproc_cluster.sh
├── delete_gcp_dataproc_cluster.sh
├── main.py
└── write_output_to_gcp_bigquery_table.sql
```

The final BigQuery table output looks like this:

| html_filepath                                       | word       | word_count |
|-----------------------------------------------------|------------|------------|
| gs://my_bucket_name/input_data/theguardian.com.html | email      | 2          |
| gs://my_bucket_name/input_data/theguardian.com.html | dimbleby   | 1          |
| gs://my_bucket_name/input_data/theguardian.com.html | stories    | 3          | 
| gs://my_bucket_name/input_data/theguardian.com.html | comments   | 12         |
| gs://my_bucket_name/input_data/theguardian.com.html | evidence   | 2          |
| gs://my_bucket_name/input_data/theguardian.com.html | invasion   | 2          |
| ...                                                 | ...        | ...        |
...


```bash
SPARK_CLUST_NAME="html-word-counter"
GCP_PROJ_NAME="your_gcloud_project_name"
N_WORKERS=50

source create_gcp_dataproc_cluster.sh
gcloud dataproc jobs submit pyspark \
    main.py \
    --cluster=$SPARK_CLUST_NAME \
    --region='europe-west2' \
    --py-files="code_section_timer.py" \
    -- \
    --data_input_bucket "gs://your_bucket_name" \
    --data_input_path "input_data" \
    --data_output_bucket "gs://your_bucket_name" \
    --data_output_path "output_data"
source delete_gcp_dataproc_cluster.sh
```

```
[Start Pyspark Script] 2024-05-29 09:14:42                                          
                0 seconds                                                           
[Count files to process] 2024-05-29 09:14:42                                        
                0 seconds                                                           
[Generate stopwords list] 2024-05-29 09:14:43                                       
                0 seconds                                                           
[Start spark session] 2024-05-29 09:14:43                                           
                8 seconds                                                           
[Read in input data] 2024-05-29 09:14:51                                            
                25 seconds                                                          
[Data processing] 2024-05-29 09:15:17                                               
                1 seconds                                                           
[Exporting results to cloud storage] 2024-05-29 09:15:18                            
                32 seconds                                                          
[Finished Pyspark Script] 2024-05-29 09:15:51
```

```python
# main.py

```

```bash
# create_gcp_dataproc_cluster.sh
```

```bash
# delete_gcp_dataproc_cluster.sh 
```

```sql
-- write_output_to_gcp_bigquery_table.sql

```

```python
# code_section_timer.py
```




