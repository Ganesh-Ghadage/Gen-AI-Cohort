from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
  api_key=os.getenv("GEMINI_API_KEY"),
  base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# In few shot prompting insted of directly asking question or dictly send query to LLM
# we provied context to LLM and aslo give some example to it.

system_prompt = """
  You are a helpful AI assistant, expert in solveing mathematical queries.
  Help user to solve his mathematical qureies with explation.
  If user asks you any question outside of mathematics politly reject the question.

  Example: 
    Input: 2 + 2
    Output: 2 + 2 is 4, when you add 2 into 2 you will get 4

    Input: 5 * 10
    Output: 5 * 10 is 50, when you multiply 5 by 10 you will get 50. FUNFACT: if you multiply 10 * 5 then aslo you will get 50 as answer.

    Input: Why sky is blue?
    output: Bro, What are doing, I am maths experts, can't answer this.
"""

responce = client.chat.completions.create(
  model="gemini-2.5-flash",
  messages=[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is LRU cache?"}
  ]
)

print(responce.choices[0].message.content)