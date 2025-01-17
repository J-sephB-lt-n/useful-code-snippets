"""
TAGS: database|db|multitenant|multitenancy|qdrant|partition|separation|vector|vector db|vector database
DESCRIPTION: Experiments in QDrant multitenancy - I thought that prefetch queries would not obey the query_filter on the second query, but they actually do
REQUIREMENTS:   pip install qdrant-client
NOTES: https://qdrant.tech/documentation/quickstart/
NOTES: https://qdrant.tech/documentation/guides/multiple-partitions/
NOTES: https://qdrant.tech/documentation/concepts/filtering/
NOTES: https://qdrant.tech/documentation/concepts/hybrid-queries/
"""

import qdrant_client as qd

qdrant_db_client = qd.QdrantClient(host="localhost", port="6333")

qdrant_db_client.create_collection(
    collection_name="temp-collection",
    vectors_config=qd.models.VectorParams(
        size=4,
        distance=qd.models.Distance.COSINE,
    ),
)

# separate clients using point payload #
qdrant_db_client.upsert(
    collection_name="temp-collection",
    wait=True,
    points=[
        qd.models.PointStruct(
            id=1,
            vector=[1, 2, 3, 4],
            payload={"tenant_id": "1"}, # Tenant 1
        ),
        qd.models.PointStruct(
            id=2,
            vector=[101, 102, 103, 104],
            payload={"tenant_id": "2"}, # Tenant 2
        ),
    ],
)

def safe_query_points(tenant_id, collection_name, query, *args, **kwargs):
    """
    Toy example wrapper achieving tenant separation using an enforced MUST filter on the client side
    (obviously more complex logic is required to combine this with user's own query_filter)
    """
    return qdrant_db_client.query_points(
        collection_name,
        query,
        query_filter=qd.models.Filter(
            must=[
                qd.models.FieldCondition(
                    key="tenant_id",
                    match=qd.models.MatchValue(
                        value=tenant_id,
                    ),
                )
            ]
        ),
        *args,
        **kwargs,
    )

tenant1_query = safe_query_points(
    tenant_id="1",
    collection_name="temp-collection",
    query=[101, 102, 103, 104],
    limit=2,
)
assert len(tenant1_query.points) == 1
assert tenant1_query.points[0].payload["tenant_id"] == "1"

tenant2_query = safe_query_points(
    tenant_id="2",
    collection_name="temp-collection",
    query=[1, 2, 3, 4],
    limit=2,
)
assert len(tenant2_query.points) == 1
assert tenant2_query.points[0].payload["tenant_id"] == "2"

# I thought that a 2-stage search would cause tenants to leak 
# points to each other, but the prefetch actually obeys the 
# query_filter outside of the prefetch
hybrid_search = safe_query_points(
    tenant_id="2",
    collection_name="temp-collection",
    prefetch=[
        qd.models.Prefetch(
            query=[1, 2, 3, 4],
            limit=1,
        ),
        qd.models.Prefetch(
            query=[11, 12, 13, 14],
            limit=1,
        ),
    ],
    query=qd.models.FusionQuery(fusion=qd.models.Fusion.RRF),
    limit=1,
)
assert len(hybrid_search.points) == 1 
assert hybrid_search.points[0].payload["tenant_id"]=="2"
