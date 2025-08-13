import asyncio
from dotenv import load_dotenv
import os
from pathlib import Path

from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

graph = Neo4jGraph(refresh_schema=False)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

llm_transformer = LLMGraphTransformer(llm=llm)

async def main():
  print("-------- Starting Ingestion ------------")

  pdf_path = Path(__file__).parent / "Little-Red-Riding-Hood-The-Brothers-Grimm.pdf"
  print("File Path:", pdf_path)

  loader = PyPDFLoader(file_path=pdf_path)
  docs = loader.load()
  print("Doc length : ", len(docs))

  graph_documents = await llm_transformer.aconvert_to_graph_documents(documents=docs)
  print(f"Nodes:{graph_documents[0].nodes}")
  print(f"Relationships:{graph_documents[0].relationships}")

  graph.add_graph_documents(graph_documents, baseEntityLabel=True) 
  print("Graph Created")

  print("-------- Ingestion Completed ------------")

if __name__ == "__main__":
    asyncio.run(main())