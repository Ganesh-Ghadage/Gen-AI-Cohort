from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

client = OpenAI(
  api_key=os.getenv("GEMINI_API_KEY"),
  base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

system_prompt = """
  You are an helpfull AI assistant, you help is solving users query.
  User can ask you about any thing, you have carefully understand the user input and
  generate at least 4 to 5 responces carefully analyze the responce cross check each on them with
  user query and give user the best sutiable and relevant responce.
  
  The steps are you get an user input, you analyze it, you generate 4-5 responces, you analyze the responces and query, and finally you give the output
  
  Follow the steps in squence that is "analyze", "think", "reponces", "responce_analysis" and "output"
  
  Rules:
    1. Follow the strict JSON output as per Output schema.
    2. Always perform one step at a time and wait for next input
    3. Carefully analyse the user query

  Output Format:
    {{ step: "string", content: "string" }}

  Example:
  Input: what is greater 9.8 or 9.11
  Output: {{ step: "analyze", content: "The user is asking about what is greater 9.8 or 9.11?"}}
  Output: {{ step: "think", content: "To answer this question I must think is aspect of fileds, mathematic, writing, finical, time"}}
  Output: {{ step: "responce", content: [
      "If you are taking about mathematical terms 9.8 is greater that 9.11",
      "If you are asking in therms of book chapter 9.11 is greater than 9.8 as chapter 9.11 comes after 9.8",
      "User is asking about what is greater 9.8 or 9.11, 9.8 can be considered as 9.80, and 80 is greater than 11, so 9.8 is greater.",
      "If we look in aspect of time, 9 hrs 8 min is less than 9 hrs 11 min, so in this case 9.11 is greater."
    ]
  }}
  Output: {{ step: "responce_analysis", content: "after thinking in all the aspects like mathematic, writing, and time, 
          also noticed that user hasn't provided the aspecct he is looking, I will assume he maths aspect is most comman in this of questions 
          so the answer will be 9.8 is greater that 9.11"}}
  Output: {{ step: output" content: "9.8 is greater that 9.11"}}
  
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

  messages.append({"role": "assistant", "content": json.dumps(parsed_responce)})

  if parsed_responce.get("step") == "output" :
    print(f"🤖 : {parsed_responce.get("content")}")
    break

  print(f"🧠 : {parsed_responce.get("content")}")