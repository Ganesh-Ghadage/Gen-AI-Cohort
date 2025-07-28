from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import platform

load_dotenv()

os_name = platform.system().lower()

client = OpenAI(
  api_key=os.getenv("GEMINI_API_KEY"),
  base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def run_command(cmd: str):
  print(f"🔨 run_command tool called, cmd: {cmd}")
  
  if cmd.find('sudo') != -1 or cmd.find('dzdo') != -1:
    permission = input(f"are you sure? you wants to run sudo command : {cmd} y/n? : ")
    
    if permission.lower in ["n", "no"]:
      print("Command terminated")
      return "Command terminated"
  
  result = os.system(cmd)
  print(f"📢: tool call result: {result}")
  
  if result == 0:
    return "commnad executed successfully"
  
  return "command exection failed"

def create_or_write_to_file(file_path, content, mode='w'):
  print(f"🔨 write_to_file tool called, file: {file_path}")
  try:
    with open(file_path, mode) as file:
      file.write(content)
    print(f"Sucessfully added content in {file_path} file")
    return f"Sucessfully added content in {file_path} file"
  except IOError as e:
    print(f"Error writing to file '{file_path}': {e}")
    return f"Error writing to file '{file_path}': {e}"
  


avaliable_tools = {
  "run_command": {
    "fn": run_command,
    "description": "This tool helps in running command in the system, it takes command as input and run it on system"
  },
  "create_or_write_to_file": {
    "fn": create_or_write_to_file,
    "description": """
      Creates a new file or overwrites an existing one with specified content.

      Args:
        file_path (str): The path to the file to be created.
        content (str): The content to write to the file. Defaults to an empty string.
        mode (str): The file opening mode. "w" for write (overwrites), "x" for exclusive creation (errors if exists),
                    "a" for append (adds to end if exists). Defaults to "w".
    """
  }
}

tool_descriptions = "\n".join(
  f"- {tool_name}: {avaliable_tools[tool_name]['description']}" for tool_name in avaliable_tools
)

system_prompt = f"""
  You are an helpfull coding AI assistant, you help in user in writing clean and efficient code.
  Write a production ready code with best practices implemations. 
  
  User may ask complex codes, analyze the user query carefully,first break down the user query in small tasks 
  plan each task acrefully and perfrom the each task at a time. wait for a task to complete and then move to next task.
  Note: Planning is crucial step so make sure you plan it correctly, think over it twice and if you feel plan can be 
  improved, improve it. 
  
  you are running on {os_name} operating system, while generating commnands keep the OS in mind and generate OS compatible commands
  Before running any commnd if it is OS compatible and check if its prerequiests are fullfill or not? if not full fill it.
  if you want execute command please open a new terminal and run all the command in that terminal
  
  follow the steps in sequence as per below:
    user query > analyze > plan > think > action > observe > result
  
  Rules:
    - strictly follow the setps sequence
    - follow strict JSON output format.
    - perform one step at a time and wait for next step
    - carefully analyze the user query
    - carefully break the query into small tasks and perfrom one task at a time.
    - generate {os_name} operating system compatible commands
    - Before running any commnd check it its prerequiests are fullfilled or not
    - before creating or writing into any file, always make sure you are in correct working directory
    
  Output JSON Format:
    {{
      "step": "string",
      "content": "string",
      "tasks": "string[]", # List of all sub tasks
      "function": "string", # Name of the function in action step
      "input": "string", # Input paramenter for the function
      "result": string" # Output of the function call
    }}
    
  Avaliable tools:
    {tool_descriptions}
  
  Example:
    Input: Write a code to add two numbers in python.
    Output: {{"step": "analyze", "content": "User is asking me to write a code which will add to number in python"}}
    Output: {{"step": "plan", "content": "to fullfill user requirment I need to plan the small tasks first", 
                  "tasks" [
                    "I need to created sum.py in user current directory",
                    "I need to generate code which will take two numbers as input and returns sum of them",
                    "I need to add this code in sum.py file"
                  ]}}
    Output: {{"step": "think", "content": "to fullfill requiremnt I have 3 steps, 1. create file, 2. write code, 3. add code in file and save it.,
                  all steps seems to be accurate and necessary to complete the requirment.}}
    Output: {{"step": "action", "function": "create_or_write_to_file", "input": {{"file_path":"sum.py", "mode": "x"}} }}
    Output: {{"step": "observe", "result": "file created sucessfully"}}
    Output: {{"step": "code", "content": "def sum(x, y):
                          return x + y}}
    Output: {{"step": "think", "content": "In order to add code in sum.py file, I need to check if file exists or not?"}}
    Output: {{"step": "action", "function": "run_cmd", "input": "ls -l"}}
    Output: {{"step": "observe", "result": "command executed sucessfully"}}
    Output: {{"step": "think", "content": "I got the sum.py file, I can add code into file"}}
    Output: {{"step": "action", "function": "create_or_write_to_file", "input": {{"file_path":"sum.py", "content": code}} }}
    Output: {{"step": "observe", "result": "Sucessfully added content in sum.py file"}}
    Output: {{"step": "output", "content": "All tasks completed, code generated successfully"}}
"""

messages = []
messages.append({"role": "system", "content": system_prompt})

print("Welcome 🙋🏼")
while True:
  query = input("> ")
  messages.append({"role": "user", "content": query})
  
  if query.lower() in ['exit', 'quit'] :
    print("Good Bye!! 👋🏼")
    break
  
  while True:
    response = client.chat.completions.create(
      model="gemini-2.0-flash",
      response_format={"type": "json_object"},
      messages=messages
    )
    
    parsed_response = json.loads(response.choices[0].message.content)
    messages.append({"role": "assistant", "content": json.dumps(parsed_response)})
    
    if parsed_response.get("step") == "output":
      print(f"🤖: {parsed_response.get("content")}")
      break
    
    if parsed_response.get("step") == "plan":
      print(f"📃: {parsed_response.get("tasks")}")
      continue
    
    if parsed_response.get("step") == "action":
      function_name = parsed_response.get("function")
      function_input = parsed_response.get("input")
      
      if function_name in avaliable_tools:
        
        if function_name == 'create_or_write_to_file':
          file_path = function_input.get("file_path")
          content = function_input.get("content") or ""
          mode = function_input.get("mode") or "w"
          
          result = avaliable_tools[function_name]["fn"](file_path, content, mode)
          messages.append({"role": "assistant", "content": json.dumps({"step": "observe", "result": result})})
          continue
        else: 
          result = avaliable_tools[function_name]["fn"](function_input)
          messages.append({"role": "assistant", "content": json.dumps({"step": "observe", "result": result})})
          continue
      else:
        print(f"❌ Invalid function name")
        
    print(f"🧠: {parsed_response.get("content")}")
      