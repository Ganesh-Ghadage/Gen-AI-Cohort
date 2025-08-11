from langchain_qdrant import QdrantVectorStore

from .qudrant_config import get_qdrant_client, COLLECTION_NAME
from .embedder import embeddings

def get_vector_store():
    client = get_qdrant_client()
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
