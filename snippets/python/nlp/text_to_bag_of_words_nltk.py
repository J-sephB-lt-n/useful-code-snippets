"""
TAGS: bag|bag of words|bow|language|of|natural|nlp|natural language|natural language processing|nltk|token|tokens|tokenisation|tokenization|word|words
DESCRIPTION: Converts a given html file into a bag-of-words representation (i.e. a count of words by frequency of occurrence)
NOTES: This process might benefit from adding stemming and n-grams
"""

# pip install beautifulsoup4 (I used version pip install beautifulsoup4==4.12.3)
# pip install nltk (I used version pip install nltk==3.8.1)

import re
from collections import defaultdict
from typing import Final

from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords

MIN_WORD_LEN_NCHAR: Final[int] = 3
MIN_WORD_FREQ_COUNT: Final[int] = 1

nltk.download("stopwords")  # this only needs to be run once

with open("some_webpage.html", "r", encoding="utf-8") as file:
    raw_html: str = file.read()

soup = BeautifulSoup(raw_html, "html.parser")

for element in soup(["style", "script"]):
    element.decompose()
for element in soup.select('[style*="display:none"], [style*="visibility:hidden"]'):
    element.decompose()

text: str = soup.get_text(separator=" ", strip=True)
text = text.lower()
text = re.sub(r"\n", " ", text)
text = re.sub(r"\s+", " ", text)
text = re.sub("[^a-zA-Z]", " ", text)

nltk_stopwords = set(stopwords.words("english"))

tokens: list[str] = nltk.word_tokenize(text)
tokens = [t for t in tokens if len(t) >= MIN_WORD_LEN_NCHAR and t not in nltk_stopwords]

word_counts_ddict: defaultdict[str, int] = defaultdict(int)
for token in tokens:
    word_counts_ddict[token] += 1

word_counts: dict[str, int] = dict(
    sorted(word_counts_ddict.items(), key=lambda x: -x[1])
)
word_counts = {k: v for k, v in word_counts.items() if v >= MIN_WORD_FREQ_COUNT}

print("Total number of words found:", len(word_counts))
print("Top 20 most common words are:")
for idx, (word, count) in enumerate(word_counts.items()):
    print(f"{idx}. {word} ({count:,})")
    if idx > 19:
        break
