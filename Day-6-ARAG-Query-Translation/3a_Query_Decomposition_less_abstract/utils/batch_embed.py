import time
from langchain_google_genai._common import GoogleGenerativeAIError

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
      
      if(start_index > total_chunks):
        print(f"Batch completed, waiting for {delay_seconds} s")
        time.sleep(delay_seconds)

    except GoogleGenerativeAIError as e:
      if "429" in str(e):
        print(f"Rate limit exceeded. Waiting for {delay_seconds} seconds...")
        time.sleep(delay_seconds)
        print("Resuming...")
        retry += 1
      else:
        print(f"An unexpected error occurred: {e}")
        raise
    
    if retry > 3 :
      print("Retry exceeded, Aborting embeddings")
      break