import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
  model="models/gemini-embedding-001",
  google_api_key=os.getenv("GEMINI_API_KEY")
)