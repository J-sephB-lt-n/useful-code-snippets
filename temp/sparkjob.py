from typing import Final

import pyspark.sql
import pyspark.sql.functions
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

MIN_WORD_NCHAR: Final[int] = 3
STOPWORDS: tuple[str] = tuple([w for w in stopwords.words("english") if "'" not in w])
STOPWORDS_STR: Final[str] = ", ".join([f"'{w}'" for w in STOPWORDS])

spark = pyspark.sql.SparkSession.builder.appName("html_word_counter").getOrCreate()


def text_fr(html_str: str) -> str:
    """docstring TODO

    Notes:
        Took some code from here: https://stackoverflow.com/questions/328356/extracting-text-from-html-file-using-python
    """
    soup = BeautifulSoup(html_str, "html.parser")
    # remove script and hidden elements from the HTML #
    for element in soup(["style", "script"]):
        element.decompose()
    for element in soup.select('[style*="display:none"], [style*="visibility:hidden"]'):
        element.decompose()
    text = soup.get_text(separator=" ")
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = "\n".join(chunk for chunk in chunks if chunk)
    return text

df = spark.read.text("gs://mybucketname/*.html", wholetext=True)
df = df.withColumn("html_filepath", pyspark.sql.functions.input_file_name())
df = df.selectExpr("value AS html_raw")

udf_text_from_html_str = pyspark.sql.functions.udf(text_from_html_str)
df = df.withColumn("html_user_facing_text", udf_text_from_html_str(df.html_raw))

df.createOrReplaceTempView("df")
spark.sql(
    f"""
SELECT      html_filepath
        ,   word
        ,   COUNT(1) AS word_count
FROM        (
            SELECT      html_filepath
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
            FROM        df
            ) c
WHERE       LEN(word) >= {MIN_WORD_NCHAR}
AND         word NOT IN ({STOPWORDS_STR})
GROUP BY    html_filepath
        ,   word
;
"""
).show()
df.show()
