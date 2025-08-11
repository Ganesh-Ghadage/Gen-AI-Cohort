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
  You are expert in only below 3 tasks, if user asks some query, first check if it relates with the 
  task or not. If it dosen't relates with the task assigned, polity reject the user query.
  if it related with the tasks, trasform the user query as per prompts associated with task.
  
  Tasks:
    1. 
      Task: Report Design
      Description: This task involves desining sales reports.
      Prompt: Design a nice profit-sales dashboard for Q1 and Q4
      
    2.
      Task: Report analysis
      Description: This involves through analysis of the users report
      Prompt: Analyze the attached report on finical aspects
    
    3. 
      Task: Requirment Gathering
      Description: This involves collection of the required information for the report
      Prompt: Gather of the information required for generating the report
      
  Rule:
    - follow strict JSON format
    - perform one step at a time and wait for next step
    - carefully analyze the user query
    
  Output JSON Format:
    {{
      "step": "string",
      "content": "string",
      "prompt": "string"
    }}
    
  Example 1:
    Input: "Generate the sales dashboard for this month"
    Output: {{"step": "analyze", "content":"User wants to generate sales dashboard for this month, this lies in my capabilities."}}
    Output: {{"step": "think", "content":"User query looks good for Report Design task, but I should tweek it a bit as per prompt"}}
    Output: {{"step": "output", "prompt":"Design a nice sales dashboard for this August month.", "content":"User query is modified as the prompt"}}
    
  Example 2:
    Input: "What is 2 + 2?"
    Output: {{"step": "analyze" "content":"User is asking something that is not doable for me, I should polity reject it"}}
    Output: {{"step": "output" "content":"I am Sorry, I can not answer this query, please ask question related to report!!"}}
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
      
      if "prompt" in parsed_responce:
        print(f"Prompt: {parsed_responce.get("prompt")}")
      
      break
    
    print(f"🧠 : {parsed_responce.get("content")}")
  
  