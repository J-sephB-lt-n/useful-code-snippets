import pyspark.sql
import pyspark.sql.functions
from bs4 import BeautifulSoup

spark = pyspark.sql.SparkSession.builder.appName("html_word_counter").getOrCreate()

df = spark.read.text("gs://mybucketname/*.html", wholetext=True)
df = df.withColumn("html_filepath", pyspark.sql.functions.input_file_name())
df = df.selectExpr("value AS html_raw")


def text_from_html_str(html_str: str) -> str:
    """docstring TODO"""
    soup = BeautifulSoup(html_str, "html.parser")
    return soup.get_text()


udf_text_from_html_str = pyspark.sql.functions.udf(text_from_html_str)
df = df.withColumn("html_user_facing_text", udf_text_from_html_str(df.html_raw))
df.show()
