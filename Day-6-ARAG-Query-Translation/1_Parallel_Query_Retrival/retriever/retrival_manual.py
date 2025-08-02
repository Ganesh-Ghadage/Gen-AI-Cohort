import os

from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from typing import List, Dict
import concurrent.futures

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

# ----- Step 1: Generate Multiple Queries -----
user_query = "What is FS Module?"
generated_queries = llm_chain.invoke(user_query)

print("--------- Generated Queries -----------")
for i, query in enumerate(generated_queries, 1):
  print(f"{i}. {query}")
print("------------------------")

## The below code will fetch the documents correctly but run in sync not in parallel
# for query in generated_queries:
#   docs = qdrant.as_retriever().invoke(query)
#   query_to_docs[query] = docs
  
# ----- Step 2: Fetch Documents in Parallel -----
def fetch_docs(query):
  docs = qdrant.as_retriever().invoke(query)
  return (query, docs)

query_to_docs: Dict[str, List] = {}

# Use ThreadPoolExecutor to run in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
  results = list(executor.map(fetch_docs, generated_queries))

# Map results
for query, docs in results:
  query_to_docs[query] = docs

for query, docs in query_to_docs.items():
  print(f"\n===== Results for Query: \"{query}\" =====")
  for i, doc in enumerate(docs, 1):
    print(f"Doc {i} ID: {doc.metadata["_id"]}")
  print("===============================")

# ----- Step 3: Deduplicate Documents -----
# Strategy: use doc.metadata['id'] if available, else page_content hash
def get_doc_id(doc):
  return doc.metadata.get("id") or hash(doc.page_content.strip())

seen = {}
for docs in query_to_docs.values():
  for doc in docs:
    doc_id = get_doc_id(doc)
    if doc_id not in seen:
      seen[doc_id] = doc

unique_docs = list(seen.values())

print(f"\nTotal Unique documents: {len(unique_docs)}")
print("===============================")
for i, doc in enumerate(unique_docs, 1):
  print(f"Doc {i} ID: {doc.metadata["_id"]}")
print("===============================")