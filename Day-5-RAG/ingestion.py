import time
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore

from qdrant_client.http.models import Distance, VectorParams
from langchain_google_genai._common import GoogleGenerativeAIError

from embedder import embeddings
from qdrant import qdrant_client

print("-------- Starting Ingestion ------------")

pdf_path = Path(__file__).parent / "nodejs.pdf"
print("File Path:", pdf_path)

loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()
print("Doc length : ", len(docs))

splitter = RecursiveCharacterTextSplitter(
  chunk_size=1000,
  chunk_overlap=200
)
chunks = splitter.split_documents(documents=docs)
print("Chunks length : ", len(chunks))



# qdrant = QdrantVectorStore.from_documents(
#   documents=[],
#   embedding=embeddings,
#   url="http://localhost:6333",
#   collection_name="nodejs_document",
# )

# qdrant.add_documents(documents=chunks)

qdrant_client.create_collection(
  collection_name="nodejs_document",
  vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
)

qdrant = QdrantVectorStore(
  client=qdrant_client,
  collection_name="nodejs_document",
  embedding=embeddings
)


def embed_in_batch(qdrant, chunks, batch_size=100, delay_seconds=60):
  """
  Adds documents to a Qdrant vector store in batches with a retry mechanism
  to handle rate limiting errors.

  Args:
    qdrant: An initialized Qdrant client instance.
    chunks: A list of documents to be added.
    batch_size: The number of documents to process in each batch.
    delay_seconds: The number of seconds to wait after a rate limit error.
  """
  total_chunks = len(chunks)
  start_index = 0
  retry = 0

  while start_index < total_chunks:
    end_index = min(start_index + batch_size, total_chunks)
    batch = chunks[start_index:end_index]

    try:
      print(f"Processing batch from index {start_index} to {end_index-1}...")
      qdrant.add_documents(documents=batch)
      print(f"Successfully added batch from index {start_index} to {end_index-1}.")
      start_index += batch_size
      
      print(f"Batch completed, waiting for {delay_seconds} s")
      time.sleep(delay_seconds)

    except GoogleGenerativeAIError as e:
      if "429" in str(e):
        print(f"Rate limit exceeded. Waiting for {delay_seconds} seconds...")
        time.sleep(delay_seconds)
        print("Resuming...")
        retry += 1
        # The loop will automatically retry the current batch
        # because start_index has not been incremented.
      else:
        # Handle other potential errors
        print(f"An unexpected error occurred: {e}")
        raise  # Re-raise the exception if it's not a rate limit error
    
    if retry > 3 :
      print("Retry exceed, Aborting embeddings")
      break
      
embed_in_batch(qdrant=qdrant, chunks=chunks, batch_size=60)

print("----------- Ingestion completed ------------")