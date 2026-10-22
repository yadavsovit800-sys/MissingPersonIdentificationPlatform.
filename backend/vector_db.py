from pathlib import Path
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

ROOT = Path(__file__).resolve().parent.parent
QDRANT_PATH = ROOT / "data" / "qdrant"

COLLECTION_NAME = "missing_persons"
VECTOR_SIZE = 512

QDRANT_PATH.mkdir(parents=True, exist_ok=True)

client = QdrantClient(path=str(QDRANT_PATH))


def ensure_collection():
    collections = client.get_collections().collections
    names = [collection.name for collection in collections]

    if COLLECTION_NAME not in names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )


def insert_vector(embedding):
    ensure_collection()

    vector_id = str(uuid.uuid4())

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=vector_id,
                vector=embedding,
                payload={}
            )
        ]
    )

    return vector_id


def search_vectors(embedding, limit=5):
    ensure_collection()

    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        limit=limit
    )

    return result.points


if __name__ == "__main__":
    ensure_collection()
    print("Qdrant collection ready")