from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
  api_key=os.getenv("GEMINI_API_KEY"),
  base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# In zero shot prompting we directly ask question or dictly send query to LLM
# In this technique we didn't provied any example or don't set any context

responce = client.chat.completions.create(
  model="gemini-2.5-flash",
  messages=[
   { "role": "user", "content": "What is 2 + 2?"}
  ]
)

print(responce.choices[0].message.content)