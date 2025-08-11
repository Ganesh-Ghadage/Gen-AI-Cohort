from dotenv import load_dotenv

from ingestion.index_document import index_documents
from generator.chat_llm import llm_chat

load_dotenv()

def main():
  # Indexing needs to do only for the first time,
  # Consider writing diffirent script for injestion that will only run for first time or When dataset is updated
  # index_documents()

  query = "How to import file in node.js?"
  result = llm_chat(query=query)

  print(result)


if __name__ == "__main__":
  main()