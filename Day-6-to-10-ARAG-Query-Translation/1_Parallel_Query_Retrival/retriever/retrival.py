import os
from dotenv import load_dotenv

from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import BaseOutputParser
from typing import List, Dict, Set
import concurrent.futures
from langchain_core.documents import Document

from utils.output_parser import output_parser
from config.vector_store import get_vector_store
from llm.prompt_templates import QUERY_REWRITE_PROMPT

# ----- setup --------

qdrant = get_vector_store()

llm = ChatGoogleGenerativeAI(
  model="gemini-2.5-flash",
  google_api_key=os.getenv("GEMINI_API_KEY"),
  temperature=0,
  max_tokens=None,
  timeout=None,
  max_retries=2,
)

llm_chain = QUERY_REWRITE_PROMPT | llm | output_parser

def parallel_query_retriver(
  user_query: str, 
  llm_chain=llm_chain, 
  retriever=qdrant.as_retriever()
) -> List[Document]:
  """
  Generate multiple queries using llm_chain, fetch documents using retriever,
  and return deduplicated list of documents.
  
  Args:
    user_query (str): The input user query.
    llm_chain: A Runnable (PromptTemplate | LLM | OutputParser) for generating queries.
    retriever: A LangChain retriever (e.g., qdrant.as_retriever()).

  Returns:
    List[Document]: Deduplicated retrieved documents across all queries.
  """
  # Step 1: Generate multiple queries
  generated_queries = llm_chain.invoke(user_query)

  # Step 2: Fetch documents in parallel
  def fetch_docs(query: str):
    return retriever.invoke(query)

  all_docs: List[Document] = []
  with concurrent.futures.ThreadPoolExecutor() as executor:
    results = executor.map(fetch_docs, generated_queries)
    for docs in results:
      all_docs.extend(docs)

  # Step 3: Deduplicate docs
  seen: Set[str] = set()
  unique_docs: List[Document] = []

  for doc in all_docs:
    doc_id = doc.metadata.get("id") or hash(doc.page_content.strip())
    if doc_id not in seen:
      seen.add(doc_id)
      unique_docs.append(doc)

  return unique_docs