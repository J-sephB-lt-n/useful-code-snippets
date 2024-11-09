```
TAGS: database|db|embed|embedding|qdrant|vector|vector database|vector db
DESCRIPTION: Example of basic usage of QDrant vector database using the python API
REQUIREMENTS: pip install numpy qdrant-client
NOTES: A good starting point for hybrid search is https://qdrant.tech/articles/hybrid-search/
```

```shell
# pull the latest QDrant image
docker pull qdrant/qdrant
```

```shell
# start the Qdrant server 
# (note that there is no auth or encryption)
docker run \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage:z \
  qdrant/qdrant
```

```shell
# install python dependencies
uv add numpy
uv add qdrant-client # or pip install qdrant-client 
```

```python
import json

import qdrant_client as qd
import numpy as np

qd_client = qd.QdrantClient(url="http://localhost:6333")

# create a collection #
collection_is_created: bool = qd_client.create_collection(
    # docs: https://qdrant.tech/documentation/concepts/collections/
    collection_name="test_collection",
    vectors_config=qd.models.VectorParams(
        size=4,  # vector dimension
        distance=qd.models.Distance.COSINE,  # {"COSINE", "DOT", "EUCLID", "MANHATTAN"}
    ),
    sparse_vectors_config={
        "text": qd.models.SparseVectorParams(),
    },
)

# insert points (vectors) into the database #
insert_operation_info: qd.http.models.UpdateResult = qd_client.upsert(
    collection_name="test_collection",
    wait=True,
    points=[
        qd.models.PointStruct(
            id=1,
            payload={"point": {"metadata": ["goes", "here"]}},
            vector=np.random.random(4).tolist(),
        ),
        qd.models.PointStruct(
            id=2, payload={"word": "joe"}, vector=np.random.random(4).tolist()
        ),
        qd.models.PointStruct(
            id=3, payload={"word": "is"}, vector=np.random.random(4).tolist()
        ),
        qd.models.PointStruct(
            id=4, payload={"word": "the"}, vector=np.random.random(4).tolist()
        ),
        qd.models.PointStruct(
            id=5, payload={"word": "best"}, vector=np.random.random(4).tolist()
        ),
    ],
)
print(
    json.dumps(
        insert_operation_info.dict(),
        # indent=4,
    ),
)
# {"operation_id": 0, "status": "completed"}

# check if a collection exists #
collection_exists: bool = qd_client.collection_exists(collection_name="test_collection")

# get collection info #
collection_info: qd.http.models.CollectionInfo = qd_client.get_collection(
    collection_name="test_collection"
)
print(
    json.dumps(
        collection_info.dict(),
        indent=4,
    ),
)

# query the collection #
search_result: qd.http.models.QueryResponse = qd_client.query_points(
    collection_name="test_collection",
    query=[0.2, 0.1, 0.9, 0.7],
    with_payload=True,
    limit=2,
)
print(
    json.dumps(
        search_result.dict(),
        indent=4,
    )
)
# {
#     "points": [
#         {
#             "id": 5,
#             "version": 1,
#             "score": 0.9131471,
#             "payload": {
#                 "word": "best"
#             },
#             "vector": null,
#             "shard_key": null,
#             "order_value": null
#         },
#         {
#             "id": 1,
#             "version": 1,
#             "score": 0.8711503,
#             "payload": {
#                 "point": {
#                     "metadata": [
#                         "goes",
#                         "here"
#                     ]
#                 }
#             },
#             "vector": null,
#             "shard_key": null,
#             "order_value": null
#         }
#     ]
# }

# delete collection #
delete_collection_response: bool = qd_client.delete_collection(
    collection_name="test_collection"
)
```

