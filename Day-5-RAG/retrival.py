import os
from dotenv import load_dotenv

from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from qdrant import qdrant_client
from embedder import embeddings

load_dotenv()

qdrant = QdrantVectorStore(
  client=qdrant_client,
  collection_name="nodejs_document",
  embedding=embeddings
)

llm = ChatGoogleGenerativeAI(
  model="gemini-2.5-flash",
  google_api_key=os.getenv("GEMINI_API_KEY"),
  temperature=0,
  max_tokens=None,
  timeout=None,
  max_retries=2,
)

prompt = ChatPromptTemplate.from_messages(
  [
    (
      "system",
      """You are an helpfull AI assistant, you answwr user query based on revlevant context data provided to you.
      Relevant Context data :
      {relavent_chunks}""",
    ),
    ("human", "{input}"),
  ]
)

chain = prompt | llm

query = input("> ")
relavent_chunks = qdrant.similarity_search(query=query)

ai_msg = chain.invoke({
  "relavent_chunks": relavent_chunks,
  "input": query
})

print(ai_msg.content)


  
        