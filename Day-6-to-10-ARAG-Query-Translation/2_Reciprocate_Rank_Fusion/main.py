import os
from dotenv import load_dotenv

from ingestion.index_document import index_documents
from generator.chat_llm import llm_chat

load_dotenv()

def main():
  # index_documents()

  query = "what is fs module?"
  result = llm_chat(query=query)

  print(result)

if __name__ == "__main__":
  main()