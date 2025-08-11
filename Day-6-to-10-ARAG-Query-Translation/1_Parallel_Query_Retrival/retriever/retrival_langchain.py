import os

from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers.multi_query import MultiQueryRetriever

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

## This is Langchain abstract way:
## Eberything will be taken care by MultiQueryRetriever

retriver =  MultiQueryRetriever(
  retriever=qdrant.as_retriever(),
  llm_chain=llm_chain
)

unique_docs = retriver.invoke("What is FS Module?")
print(len(unique_docs))