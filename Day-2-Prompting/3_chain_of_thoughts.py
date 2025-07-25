from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

client = OpenAI(
  api_key=os.getenv("GEMINI_API_KEY"),
  base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# chain of thoughts prompting method involves breaking down user query,
# analyzing the query deeply, thinking on the query over 2-3 times, 
# and the providing the revelant answer to the user

system_prompt = """
  You are an AI assistant who is expert in breaking down complex problems and then resolve the user query.

  For the given user input, analyse the input and break down the problem step by step.
  Atleast think 5-6 steps on how to solve the problem before solving it down.

  The steps are you get a user input, you analyse, you think, you again think for several times and then return an output with explanation and then finally you validate the output as well before giving final result.

  Follow the steps in sequence that is "analyse", "think", "output", "validate" and finally "result".

  Rules:
    1. Follow the strict JSON output as per Output schema.
    2. Always perform one step at a time and wait for next input
    3. Carefully analyse the user query

  Output Format:
    {{ step: "string", content: "string" }}

  Example:
    Input: What is 2 + 2.
    Output: {{ step: "analyse", content: "Alright! The user is intersted in maths query and he is asking a basic arthermatic operation" }}
    Output: {{ step: "think", content: "To perform the addition i must go from left to right and add all the operands" }}
    Output: {{ step: "output", content: "4" }}
    Output: {{ step: "validate", content: "seems like 4 is correct ans for 2 + 2" }}
    Output: {{ step: "result", content: "2 + 2 = 4 and that is calculated by adding all numbers" }}

"""

messages = [
  {"role": "system", "content": system_prompt}
]

query = input("> ")
messages.append({"role": "user", "content": query})

while True:
  responce = client.chat.completions.create(
    model="gemini-2.5-flash",
    response_format={"type": "json_object"},
    messages=messages
  )

  parsed_responce = json.loads(responce.choices[0].message.content)
  
  # In this while loop we are feeding assistant responce to LLM, for next step
  messages.append({"role": "assistant", "content": json.dumps(parsed_responce)})

  if parsed_responce.get("step") == "result" :
    print(f"🤖 : {parsed_responce.get("content")}")
    break

  print(f"🧠 : {parsed_responce.get("content")}")