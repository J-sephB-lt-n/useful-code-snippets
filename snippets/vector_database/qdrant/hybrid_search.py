"""

pip install qdrant-client requests scikit-learn sentence-transformers

https://qdrant.tech/documentation/concepts/hybrid-queries/#hybrid-search
"""

import itertools
import re
import subprocess
import time

import numpy as np
import requests
import scipy.sparse
import qdrant_client as qd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


text: str = requests.get(
    url="https://www.gutenberg.org/cache/epub/35830/pg35830.txt",
).text

docs: list[str] = [
    paragraph.strip()
    for paragraph in re.split(r"\r?\n\r?\n", text)
    if len(paragraph.strip()) > 0
]

dense_embed_model: SentenceTransformer = SentenceTransformer(
    # "sentence-transformers/multi-qa-MiniLM-L6-dot-v1",
    "../../../models/language_embedding_models/multi-qa-MiniLM-L6-dot-v1/",
    local_files_only=True,
)

tfidf_vectorizer = TfidfVectorizer(
    lowercase=True,  # text converted to lowercase before tokenisation
    analyzer="word",  # split text by word
    stop_words=None,  # can supply a list of stopwords (ignored words) here
    # token_pattern=r"(?u)\b\w\w+\b", # used to split text into words
    ngram_range=(1, 1),  # (min_n, max_n). Lengths of n-grams to create
    max_df=0.5,  # if float, words appearing in more than 100(max_df)% of docs are ignored
    min_df=2,  # if int, words must appear in at least this many documents to be included
    norm="l2",  # L2 makes the dot prod the same as the cosine similarity
    smooth_idf=True,
    sublinear_tf=False,  # replaces term frequency (TF) with 1+log(tf)
)

start_time: float = time.perf_counter()
dense_doc_embeddings: np.ndarray = dense_embed_model.encode(
    docs,
    batch_size=50,
    show_progress_bar=True,
)
end_time: float = time.perf_counter()
print(
    f"Finished dense embeddings of {len(docs):,} docs in {(end_time-start_time)/60:,.1f} minutes"
)

start_time: float = time.perf_counter()
tfidf_doc_vectors: scipy.sparse._csr.csr_matrix = tfidf_vectorizer.fit_transform(docs)
end_time: float = time.perf_counter()
print(
    f"Finished training tf-idf vectoriser on {len(docs):,} docs in {(end_time-start_time)/60:,.1f} minutes"
)

# Start QDrant database server in background (docker) #
_ = subprocess.run(
    [
        "docker",
        "run",
        "--detach",
        "-p",
        "6333:6333",
        "-p",
        "6334:6334",
        "qdrant/qdrant",
    ],
    capture_output=True,
    text=True,
)

qd_client = qd.QdrantClient(url="http://localhost:6333")

_: bool = qd_client.create_collection(
    # docs: https://qdrant.tech/documentation/concepts/collections/
    collection_name="test-collection",
    vectors_config={
        "dense_embedding": qd.models.VectorParams(
            size=384,  # vector dimension
            distance=qd.models.Distance.DOT,  # {"COSINE", "DOT", "EUCLID", "MANHATTAN"}
        )
    },
    sparse_vectors_config={
        "tf_idf": qd.models.SparseVectorParams(),
    },
)

for batch in itertools.batched(
    iterable=zip(
        range(len(docs)),
        dense_doc_embeddings.tolist(),
        list(tfidf_doc_vectors),
        docs,
    ),
    n=50,  # number per batch
):
    qd_client.upsert(
        # docs: https://qdrant.tech/documentation/concepts/points/#upload-points:~:text=Sparse%20vectors%20must%20be%20named%20and%20can%20be%20uploaded%20in%20the%20same%20way%20as%20dense%20vectors.
        collection_name="test-collection",
        wait=True,
        points=[
            qd.models.PointStruct(
                id=doc_index,
                vector={
                    "dense_embedding": dense_embedding,
                    "tf_idf": qd.models.SparseVector(
                        indices=sparse_tfidf_vector.indices,
                        values=sparse_tfidf_vector.data,
                    ),
                },
                payload={
                    "doc_text": (
                        f"{doc_text[:99]}..<truncated>"
                        if len(doc_text) > 99
                        else doc_text
                    ),
                },
            )
            for doc_index, dense_embedding, sparse_tfidf_vector, doc_text in zip(
                range(len(docs)),
                dense_doc_embeddings.tolist(),
                list(tfidf_doc_vectors),
                docs,
            )
        ],
    )
