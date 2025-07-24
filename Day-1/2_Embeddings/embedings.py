from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

text="The cat sat on the mat"

result = client.models.embed_content(
  model="gemini-embedding-001",
  contents=text
)

print("Embeded Text: ", result.embeddings)