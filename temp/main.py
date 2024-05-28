import argparse
import time
from typing import Final

import google.cloud.storage
import nltk
import pyspark.sql
import pyspark.sql.functions
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

start_time: float = time.perf_counter()
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
N_FILES_TO_PROCESS: Final[int] = sum(
    1
    for _ in google.cloud.storage.Client().list_blobs(
        args.data_input_bucket.replace("gs://", ""), prefix=args.data_input_path
    )
)
STOPWORDS: tuple[str] = tuple((w for w in stopwords.words("english") if "'" not in w))
STOPWORDS_STR: Final[str] = ", ".join([f"'{w}'" for w in STOPWORDS])

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


df = spark.read.text(
    f"{args.data_input_bucket}/{args.data_input_path}/*.html", wholetext=True
)
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

df.write.mode("overwrite").option("maxRecordsPerFile", 500_000).parquet(
    f"{args.data_output_bucket}/{args.data_output_path}"
)

print(
    f"FINISHED: Processed {N_FILES_TO_PROCESS:,} files in {(time.perf_counter()-start_time)/60:,.2f} minutes."
)

spark.stop()
