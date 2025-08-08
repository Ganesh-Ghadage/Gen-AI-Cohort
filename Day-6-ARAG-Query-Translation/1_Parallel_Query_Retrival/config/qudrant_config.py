from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

COLLECTION_NAME = "nodejs_document"

def get_qdrant_client():
  client = QdrantClient(url="http://localhost:6333")
  
  if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
    client.create_collection(
      collection_name=COLLECTION_NAME,
      vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
    )
    
  return client

get_qdrant_client()