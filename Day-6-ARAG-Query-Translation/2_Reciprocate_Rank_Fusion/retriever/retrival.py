import os

from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Set
import concurrent.futures
from langchain_core.documents import Document

from utils.output_parser import output_parser
from config.vector_store import get_vector_store
from llm.prompt_templates import QUERY_REWRITE_PROMPT
from utils.rank_docs import reciprocal_rank_fusion

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

  # Step 3: Rank docs    
  sorted_docs = reciprocal_rank_fusion(all_docs)
  
  # print("--------- sorted docs ------")
  # print(f"\nTotal sorted documents: {len(sorted_docs)}")
  # print("===============================")
  # for i, doc in enumerate(sorted_docs, 1):
  #   print(f"Doc {i} ID: {doc}")
  # print("===============================")

  return sorted_docs