from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.vector_store import get_vector_store
from utils.batch_embed import embed_in_batch

qdrant = get_vector_store()

def index_documents():
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

  # As Gemini has Rate limiting per min, we are batching chunks and embedding them
  embed_in_batch(qdrant=qdrant, chunks=chunks, batch_size=60)

  print("----------- Ingestion completed ------------")