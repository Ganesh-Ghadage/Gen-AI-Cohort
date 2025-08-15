# from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import requests
from langfuse.openai import openai
from langfuse import observe


load_dotenv()

client = openai.OpenAI(
  api_key=os.getenv("GEMINI_API_KEY"),
  base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

@observe()
def get_weather(city: str):
  print("🔨: get_weather tool called, input : ", city)
  
  url = f"https://wttr.in/{city}?format=%C+%t"
  response = requests.get(url)
  
  if response.status_code == 200:
    return response.text

  return "Something went wrong"

avaliable_tools = {
  "get_weather": {
    "fn": get_weather,
    "description": "This function takes a city as a input and retuns the weather infomation for it."
  }
}

tool_descriptions = "\n".join(
  f"- {tool_name}: {avaliable_tools[tool_name]['description']}" for tool_name in avaliable_tools
)

system_prompt = f"""
  You are an helpfull weather AI assistant, you help in user to finding the current weather of their location.
  If user asks anything besids weather info, kindly reject the question.
  
  once user asks his query, you break down the query, analyze it, you plan how to answer it,
  use the approprate tool from the list to get required data, wait for it, and once you get data from tool
  use it to answer user query. include some funy message in output
  
  Rules:
    - follow strict JSON output format.
    - perform one step at a time and wait for next step
    - carefully analyze the user query
    
  Output JSON Format:
    {{
      "step": "string",
      "content": "string",
      "function": "string", # Name of the function in action step
      "input": "string", # Input paramenter for the function
      "result": string" # Output of the function call
    }}
    
  Avaliable tools:
    {tool_descriptions}
  
  Example:
    Input: What is weather in Pune?
    Output: {{"step": "analyze", "content": "User is instreated in knowing the weather information of pune"}}
    Output: {{"step": "plan", "content": "In order to get weather information I need to call get_weather tool with pune as parameter"}}
    Output: {{"step": "action", "function": "get_weather", "input": "pune"}}
    Output: {{"step": "observe", "result": "raniy, 20 degree Celcuis"}}
    Output: {{"step": "output", "content": "Curent weather of pune is raniy with 20 degree Celcuis, don't forget to take your umbrella!!"}}
"""

messages = []
messages.append({"role": "system", "content": system_prompt})

print("Welcome 🙋🏼‍♂️, Know the weather of your location!⛅")
while True:
  query = input("> ")
  messages.append({"role": "user", "content": query})
  
  if query.lower() in ['exit', 'quit'] :
    print("Good Bye!! 👋🏼")
    break
  
  while True:
    response = client.chat.completions.create(
      model="gemini-2.5-flash",
      response_format={"type": "json_object"},
      messages=messages
    )
    
    parsed_response = json.loads(response.choices[0].message.content)
    messages.append({"role": "assistant", "content": json.dumps(parsed_response)})
    
    if parsed_response.get("step") == "output":
      print(f"🤖: {parsed_response.get("content")}")
      break
    
    if parsed_response.get("step") == "action":
      function_name = parsed_response.get("function")
      function_input = parsed_response.get("input")
      
      if function_name in avaliable_tools:
        result = avaliable_tools[function_name]["fn"](function_input)
        messages.append({"role": "assistant", "content": json.dumps({"step": "observe", "result": result})})
        continue
      else:
        print(f"❌ Invalid function name")
        
    print(f"🧠: {parsed_response.get("content")}")
        