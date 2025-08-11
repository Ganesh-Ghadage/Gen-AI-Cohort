from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

client = OpenAI(
  api_key=os.getenv("GEMINI_API_KEY"),
  base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

SYSTEM_PROMPT = """
  You are an AI Routing Assistant, You help logically chooing best data source for information retervial
  for user query based on the below parameters.
  
  you work in stepped manner and follow steps as per sequence.
  user input -> analyze -> think -> result -> analyze -> output
  
  Parameters:
    1. Carefully analyze the user query.
    2. Think logically which would be the best data source for it, from the avaliable_datasource list.
    3. Check if choosen data source is correct or not.
    
  avaliable_datasources: [
    {{
      "name": "finance_data",
      "description": "This data set contains the information about all the finicial of orginization."
    }},
    {{
      "name": "employee_data",
      "description": "This data set contains the information about all the employees in orginization."
    }},
    {{
      "name": "technical_doc_data",
      "description": "This data set contains the information technical documentation of the orginizations."
    }},
    {{
      "name": "products_data",
      "description": "This data set contains the information about products that we sell."
    }},
  ]
  
  Rules:
    - follow strict JSON output format.
    - perform one step at a time and wait for next step
    - carefully analyze the user query
    
  Output JSON Format:
    {{
      "step": "string",
      "content": "string",
      "data_source": "string"
    }}
    
  Example:
    Input: "What is price of xyz product?"
    Output: {{"step": "analyze", "content":"User is asking about the price of product"}}
    Output: {{"step": "think", "content":"As per avaliable_datasources I think best choise will be products_data data source, as it might contain information about product price"}}
    Output: {{"step": "result", "content":"You should choose products_data as data source"}}
    Output: {{"step": "analyze", "content":"user asked for product information, to get this information we choose the correct data source i.e. products_data"}}
    Output: {{"step": "output", "data_source":"products_data", "content":"Best data source is: 'products_data'"}}
  
"""

messages = []
messages.append({"role": "system", "content": SYSTEM_PROMPT})

print("Welcome to App 👋🏼")
while True:
  print("Ask somethin or enter Quit to exit")
  query = input("> ")
  
  if(query.lower() == "quit" or query.lower() == 'exit'):
    print("Bye 🙋🏼‍♂️")
    break
  
  messages.append({"role": "user", "content": query})
  
  while True:
    responce = client.chat.completions.create(
      model="gemini-2.5-flash",
      response_format={"type": "json_object"},
      messages=messages
    )
    
    parsed_responce = json.loads(responce.choices[0].message.content)
    messages.append({"role":"assistant", "content": json.dumps(parsed_responce)})
    
    if parsed_responce.get("step") == "output":
      print(f"🤖: {parsed_responce.get("content")}")
      break
    
    print(f"🧠 : {parsed_responce.get("content")}")
  
  