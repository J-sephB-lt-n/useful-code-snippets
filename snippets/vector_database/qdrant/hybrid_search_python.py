"""
TAGS: database|db|docker|embed|embedding|hybrid|qdrant|query|search|sentence|sentence-transformers|store|tfidf|transformers|vector
DESCRIPTION: Code for implementing hybrid search (i.e. combined dense and sparse vector search) in Qdrant (using the python client)
REQUIREMENTS: pip install model2vec qdrant-client requests scikit-learn
NOTES: https://qdrant.tech/documentation/concepts/hybrid-queries/#hybrid-search
"""

import itertools
import random
import re
import subprocess
import time
from typing import Final

import numpy as np
import requests
import scipy.sparse
import qdrant_client as qd
import model2vec
from sklearn.feature_extraction.text import TfidfVectorizer

# fetch input text #
text: str = requests.get(
    url="https://www.gutenberg.org/cache/epub/35830/pg35830.txt",
).text

# split input text by paragraph #
docs: list[str] = [
    paragraph.strip()
    for paragraph in re.split(r"\r?\n\r?\n", text)
    if len(paragraph.strip()) > 0
]
docs = docs[:999]  # so the code runs fast

# load dense embedding model #
dense_embed_model = model2vec.StaticModel.from_pretrained("minishlab/potion-base-8M")

# if you have cloned the model repo, you can load the model like this:
# dense_embed_model: SentenceTransformer = SentenceTransformer(
#     "../../../../../models/language_embedding_models/multi-qa-MiniLM-L6-dot-v1/",
#     local_files_only=True,
# )

# load tf-idf model #
tfidf_vectorizer = TfidfVectorizer(
    lowercase=True,  # text converted to lowercase before tokenisation
    analyzer="word",  # split text by word
    stop_words=None,  # can supply a list of stopwords (ignored words) here
    # token_pattern=r"(?u)\b\w\w+\b", # used to split text into words
    ngram_range=(1, 1),  # (min_n, max_n). Lengths of n-grams to create
    max_df=1.0,  # if float, words appearing in strictly more than 100(max_df)% of docs are ignored
    min_df=1,  # if int, words must appear in at least this many documents to be included
    norm="l2",  # L2 makes the dot prod the same as the cosine similarity
    smooth_idf=True,
    sublinear_tf=False,  # replaces term frequency (TF) with 1+log(tf)
)

# calculate dense embedding for each doc (paragraph) #
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

# calculate tf-idf vectors for each doc (paragraph) #
start_time: float = time.perf_counter()
tfidf_doc_vectors: scipy.sparse._csr.csr_matrix = tfidf_vectorizer.fit_transform(docs)
end_time: float = time.perf_counter()
print(
    f"Finished training tf-idf vectoriser on {len(docs):,} docs in {(end_time-start_time)/60:,.1f} minutes"
)

# Start vector database (QDrant) #
_ = subprocess.run(
    [
        "docker",
        "run",
        "--name",
        "qdrant_vector_db",
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

# vector database python client #
qd_client = qd.QdrantClient(host="localhost", port=6333)

# create collection in vector database #
COLLECTION_NAME: Final[str] = "my-temp-collection"
if not qd_client.collection_exists(
    collection_name=COLLECTION_NAME,
):
    _ = qd_client.create_collection(
        # docs: https://qdrant.tech/documentation/concepts/collections/
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense_embedding": qd.models.VectorParams(
                size=dense_doc_embeddings.shape[1],  # vector dimension
                distance=qd.models.Distance.DOT,  # {"COSINE", "DOT", "EUCLID", "MANHATTAN"}
                # use DOT instead of COSINE since vectors in minishlab/potion-base-8M are unit length
            )
        },
        sparse_vectors_config={
            "tf_idf": qd.models.SparseVectorParams(),
        },
    )

# add all vectors into the vector database #
for batch_num, batch in enumerate(
    itertools.batched(
        iterable=zip(
            range(len(docs)),
            dense_doc_embeddings.tolist(),
            tfidf_doc_vectors,
            docs,
        ),
        n=50,  # number per batch
    ),
    start=1,
):
    print("started batch", batch_num)
    qd_client.upsert(
        # docs: https://qdrant.tech/documentation/concepts/points/#upload-points:~:text=Sparse%20vectors%20must%20be%20named%20and%20can%20be%20uploaded%20in%20the%20same%20way%20as%20dense%20vectors.
        collection_name=COLLECTION_NAME,
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
            for doc_index, dense_embedding, sparse_tfidf_vector, doc_text in batch
        ],
    )

# query vector database using different methods #
query_text: str = "wartime economic statistics"

print(":: Dense embeddings only query ::")
search_result = qd_client.query_points(
    collection_name=COLLECTION_NAME,
    query=dense_embed_model.encode([query_text])[0].tolist(),
    with_payload=True,
    limit=5,
    using="dense_embedding",
)
for result in search_result.points:
    print(result.payload["doc_text"], "\n-----")

print(":: TF-IDF only query ::")
query_tfidf_vector = tfidf_vectorizer.transform([query_text])[0]
search_result = qd_client.query_points(
    collection_name=COLLECTION_NAME,
    query=qd.models.SparseVector(
        indices=query_tfidf_vector.indices,
        values=query_tfidf_vector.data,
    ),
    with_payload=True,
    limit=5,
    using="tf_idf",
)
for result in search_result.points:
    print(result.payload["doc_text"], "\n-----")

print(":: Hybrid search using RRF (dense+TF-IDF) ::")
query_tfidf_vector = tfidf_vectorizer.transform([query_text])[0]
search_result = qd_client.query_points(
    collection_name=COLLECTION_NAME,
    prefetch=[
        qd.models.Prefetch(
            query=dense_embed_model.encode([query_text])[0].tolist(),
            using="dense_embedding",
            limit=50,
        ),
        qd.models.Prefetch(
            query=qd.models.SparseVector(
                indices=query_tfidf_vector.indices,
                values=query_tfidf_vector.data,
            ),
            using="tf_idf",
            limit=50,
        ),
    ],
    query=qd.models.FusionQuery(fusion=qd.models.Fusion.RRF),
    limit=5,
)
for result in search_result.points:
    print(result.payload["doc_text"], "\n-----")

# double check my understanding of sparse matrices (used by TF-IDF) #
random_doc_index: int = random.randint(0, len(docs) - 1)
random_doc: str = docs[random_doc_index]
query_tfidf_vector = tfidf_vectorizer.transform(
    [random_doc]
)  # don't need to index if a 1xn sparse array
search_result = qd_client.query_points(
    collection_name=COLLECTION_NAME,
    query=qd.models.SparseVector(
        indices=query_tfidf_vector.indices,
        values=query_tfidf_vector.data,
    ),
    with_payload=True,
    with_vectors=True,
    limit=1,
    using="tf_idf",
)
result_text: str = search_result.points[0].payload["doc_text"]
assert result_text == random_doc, f"'{result_text}'\n!=\n'{random_doc}'"
assert (
    tfidf_doc_vectors.todense()[random_doc_index]
    == tfidf_doc_vectors[random_doc_index].todense()
).all()
true_doc_embed_dense: np.ndarray = tfidf_doc_vectors[random_doc_index].todense()
non_zero_indices_true_doc_embed: list[int] = true_doc_embed_dense.nonzero()[1].tolist()
assert (
    non_zero_indices_true_doc_embed == search_result.points[0].vector["tf_idf"].indices
)
assert (
    np.sum(
        np.array(true_doc_embed_dense[0, non_zero_indices_true_doc_embed].tolist()[0])
        - np.array(search_result.points[0].vector["tf_idf"].values)
    )
    < 0.000001
)


# shut down the vector database (qdrant) container #
_ = subprocess.run(
    [
        "docker",
        "stop",
        "qdrant_vector_db",
    ],
    capture_output=True,
    text=True,
)

# delete the vector database (qdrant) container #
_ = subprocess.run(
    [
        "docker",
        "rm",
        "qdrant_vector_db",
    ],
    capture_output=True,
    text=True,
)
