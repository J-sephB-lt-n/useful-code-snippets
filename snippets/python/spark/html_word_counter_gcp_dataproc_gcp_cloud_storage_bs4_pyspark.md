```
TAGS: apache|apache spark|big data|bigquery|cloud|cloud storage|cluster|data|dataproc|etl|google|google cloud|gcp|job|pyspark|spark|spark cluster|storage
DESCRIPTION: Runs a PySpark job on GCP Dataproc which parses HTML files and writes the wordcounts to BigQuery via google cloud storage
NOTES: I tested the code by processing 10,000 HTML files stored on google cloud storage
NOTES: Each HTML file was on average 1Mb (i.e. 10Gb of total input data)
NOTES: I used 1 cluster manager machine and 50 worker machines in the spark cluster
NOTES: At time of writing (May 2024), the cost of the 50 worker VMs + 1 manager VM was $10.20 per hour ($0.17 per minute)   
NOTES: The spark job took 69 seconds to complete (i.e. processed 145 files per second = 145Mb of data per second)
NOTES: The majority of the job time (57 of the 69 seconds = 83%) was reading and writing data to/from GCP Cloud Storage (full timing breakdown later in this doc)
NOTES: The entire process of cluster creation, job run and cluster deletion took 4.94 minutes
NOTES: The spark job wrote parquet files to cloud storage, which were then ingested into a BigQuery table (the writing to BigQuery took 4 seconds)
NOTES: The final result was a BigQuery table containing 10,520,000 rows
```

My code folder for this process looks like this:

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
| gs://your_bucket_name/input_data/theguardian.com.html | email      | 2          |
| gs://your_bucket_name/input_data/theguardian.com.html | dimbleby   | 1          |
| gs://your_bucket_name/input_data/theguardian.com.html | stories    | 3          | 
| gs://your_bucket_name/input_data/theguardian.com.html | comments   | 12         |
| gs://your_bucket_name/input_data/theguardian.com.html | evidence   | 2          |
| gs://your_bucket_name/input_data/theguardian.com.html | invasion   | 2          |
| ...                                                   | ...        | ...        |

Here is the terminal code for running the job:

```bash
$ SPARK_CLUST_NAME="html-word-counter"
$ GCP_PROJ_NAME="your_gcloud_project_name"
$ N_WORKERS=50
$ source create_gcp_dataproc_cluster.sh
...
$ gcloud dataproc jobs submit pyspark \
    main.py \
    --cluster=$SPARK_CLUST_NAME \
    --region='europe-west2' \
    --py-files="code_section_timer.py" \
    -- \
    --data_input_bucket "gs://your_bucket_name" \
    --data_input_path "input_data" \
    --data_output_bucket "gs://your_bucket_name" \
    --data_output_path "output_data"
...
$ source delete_gcp_dataproc_cluster.sh
...
```

Here are the specific timings which I observed for the different parts of the spark job:

```
[Start Pyspark Script] 2024-05-25 09:14:42                                          
                0 seconds                                                           
[Count files to process] 2024-05-25 09:14:42                                        
                0 seconds                                                           
[Generate stopwords list] 2024-05-25 09:14:43                                       
                0 seconds                                                           
[Start spark session] 2024-05-25 09:14:43                                           
                8 seconds                                                           
[Read in input data] 2024-05-25 09:14:51                                            
                25 seconds                                                          
[Data processing] 2024-05-25 09:15:17                                               
                1 seconds                                                           
[Exporting results to cloud storage] 2024-05-25 09:15:18                            
                32 seconds                                                          
[Finished Pyspark Script] 2024-05-25 09:15:51
Processed 10,000 files
```

Here are the contents of the various scripts: 

```python
# main.py
"""
Main entrypoint of the pyspark job
"""
import argparse
from typing import Final

import google.cloud.storage
import nltk
import pyspark.sql
import pyspark.sql.functions
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

from code_section_timer import CodeSectionTimer

timer = CodeSectionTimer()

timer.checkpoint("Start Pyspark Script")
nltk.download("stopwords")

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--data_input_bucket",
    help="Data will be imported from this cloud storage bucket",
    required=True,
)
parser.add_argument(
    "-I",
    "--data_input_path",
    help="Data will be imported from this location on the input bucket",
    required=True,
)
parser.add_argument(
    "-o",
    "--data_output_bucket",
    help="Data will be exported to this cloud storage bucket",
    required=True,
)
parser.add_argument(
    "-O",
    "--data_output_path",
    help="Data will be exported to this location on the output bucket",
    required=True,
)
args = parser.parse_args()

MIN_WORD_NCHARS: Final[int] = 3
timer.checkpoint("Count files to process")
N_FILES_TO_PROCESS: Final[int] = sum(
    1
    for _ in google.cloud.storage.Client().list_blobs(
        args.data_input_bucket.replace("gs://", ""), prefix=args.data_input_path
    )
)
timer.checkpoint("Generate stopwords list")
STOPWORDS: tuple[str] = tuple((w for w in stopwords.words("english") if "'" not in w))
STOPWORDS_STR: Final[str] = ", ".join([f"'{w}'" for w in STOPWORDS])

timer.checkpoint("Start spark session")
spark = pyspark.sql.SparkSession.builder.appName("html_word_counter").getOrCreate()

def text_from_html_str(html_str: str) -> str:
    """Extracts user-facing text from provided HTML

    Notes:
        Some code from here: https://stackoverflow.com/questions/328356/extracting-text-from-html-file-using-python
    """
    soup = BeautifulSoup(html_str, "html.parser")
    # remove script and hidden elements from the HTML #
    for element in soup(["style", "script"]):
        element.decompose()
    for element in soup.select('[style*="display:none"], [style*="visibility:hidden"]'):
        element.decompose()
    text = soup.get_text(separator=" ")
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in chunks if chunk)
    return text

timer.checkpoint("Read in input data")
df = spark.read.text(
    f"{args.data_input_bucket}/{args.data_input_path}/*.html", wholetext=True
)
timer.checkpoint("Data processing")
df = df.withColumn("html_filepath", pyspark.sql.functions.input_file_name())
df = df.selectExpr("html_filepath", "value AS html_raw")

udf_text_from_html_str = pyspark.sql.functions.udf(text_from_html_str)
df = df.withColumn("html_user_facing_text", udf_text_from_html_str(df.html_raw))

df.createOrReplaceTempView("df")
df = spark.sql(
    f"""
SELECT      html_filepath
        ,   word
        ,   COUNT(1) AS word_count
FROM        (
            SELECT  html_filepath
                ,   EXPLODE(
                        SPLIT(
                            REGEXP_REPLACE(
                                LOWER(html_user_facing_text),
                                '[^a-zA-Z]',
                                  ' '
                            ),
                            ' '
                        )
                    ) AS word
            FROM    df
            ) c
WHERE       LEN(word) >= {MIN_WORD_NCHARS}
AND         word NOT IN ({STOPWORDS_STR})
GROUP BY    html_filepath
        ,   word
;
"""
)

timer.checkpoint("Exporting results to cloud storage")
df.write.mode("overwrite").option("maxRecordsPerFile", 500_000).parquet(
    f"{args.data_output_bucket}/{args.data_output_path}"
)

timer.checkpoint("Finished Pyspark Script")
print(timer.summary_string())
print(f"Processed {N_FILES_TO_PROCESS:,} files")
spark.stop()
```

```bash
# create_gcp_dataproc_cluster.sh

#!/bin/bash
gcloud dataproc clusters create \
    $SPARK_CLUST_NAME \
    --region europe-west2 \
    --zone europe-west2-c \
    --enable-component-gateway \
    --master-machine-type n2-standard-4 \
    --master-boot-disk-type pd-balanced \
    --master-boot-disk-size 500 \
    --num-workers $N_WORKERS \
    --worker-machine-type n2-standard-4 \
    --worker-boot-disk-type pd-balanced \
    --worker-boot-disk-size 500 \
    --image-version 2.2-debian12 \
    --project $GCP_PROJ_NAME
```

```bash
# delete_gcp_dataproc_cluster.sh 

#!/bin/bash
gcloud dataproc clusters delete \
    $SPARK_CLUST_NAME \
    --region europe-west2 \
    --quiet
```

```sql
-- write_output_to_gcp_bigquery_table.sql

LOAD DATA OVERWRITE `yourProjectName.yourDatasetName.yourTableName`
CLUSTER BY word
FROM FILES (
      format = 'PARQUET',
      uris = ['gs://your_bucket_name/output_data/*.parquet']
)
;
```

```python
# code_section_timer.py

"""Defines class CodeSectionTimer()"""
import datetime
import time

class CodeSectionTimer:
    """Convenient interface for timing a script which contains multiple sections of interest"""

    def __init__(self) -> None:
        """docstring TODO"""
        self.history = []

    def checkpoint(self, name: str) -> None:
        """docstring TODO"""
        self.history.append((name, time.perf_counter(), datetime.datetime.now()))

    def summary_string(self) -> str:
        """docstring TODO"""
        build_str = ""
        for idx, chkpnt in enumerate(self.history):
            if idx > 0:
                secs_elapsed = chkpnt[1] - self.history[idx - 1][1]
                tot_hours = int(secs_elapsed / 3600)
                tot_minutes = int(secs_elapsed / 60) - (60 * tot_hours)
                tot_seconds = (
                    int(secs_elapsed) - (60 * tot_minutes) - (3600 * tot_hours)
                )
                build_str += f"\t\t"
                if tot_hours > 0:
                    build_str += f"{tot_hours} hours "
                if tot_minutes > 0:
                    build_str += f"{tot_minutes} minutes "
                build_str += f"{tot_seconds} seconds\n"
            build_str += (
                f"[{chkpnt[0]}] " + chkpnt[2].strftime("%Y-%m-%d %H:%M:%S") + "\n"
            )
        return build_str
```




