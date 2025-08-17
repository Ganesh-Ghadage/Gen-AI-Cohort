from openai import OpenAI
from dotenv import load_dotenv
import os
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langsmith.wrappers import wrap_openai
from pydantic import BaseModel
from typing import Literal, Union
import platform
import subprocess
import json
from langsmith import traceable
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

os_name = platform.system().lower()

client = wrap_openai(OpenAI(
  api_key=os.getenv("GEMINI_API_KEY"),
  base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
))

parser = JsonOutputParser()

#schema
class DetectQuerySchema(BaseModel):
  is_coding_que: bool
  
class DetectCodeTypeSchema(BaseModel):
  is_que_to_write_code: bool

class LLMResponceSchema(BaseModel):
  answer: str

class State(TypedDict):
  user_message: str
  ai_message: str
  is_coding_que: bool
  is_que_to_write_code: bool
  step: str
  function: str
  tool_input: dict
  tool_result: str
  messages: list[str]

# Tools
@traceable
def run_command(cmd: str):
  # print(f"\n🔨 run_command tool called, cmd: {cmd}\n")
  if "sudo" in cmd or "dzdo" in cmd:
    permission = input(f"are you sure? you want to run sudo command : {cmd} y/n? : ")
    if permission.lower() in ["n", "no"]:
      return "Command terminated"

    try:
      result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
      if result.returncode == 0:
        return f"✅ Success:\n{result.stdout.strip()}"
      else:
        return f"❌ Error:\n{result.stderr.strip()}"
    except Exception as e:
      return f"⚠️ Exception: {e}"

@traceable
def create_or_write_to_file(file_path, content, mode='w'):
  # print(f"\n🔨 write_to_file tool called, file: {file_path}\n")
  try:
    with open(file_path, mode) as file:
      file.write(content)
    # print(f"\nSucessfully added content in {file_path} file\n")
    return f"Sucessfully added content in {file_path} file"
  except IOError as e:
    # print(f"\nError writing to file '{file_path}': {e}\n")
    return f"Error writing to file '{file_path}': {e}"
  
# Nodes
def analyze_query(state: State):
  user_message = state.get("user_message")
  
  SYSTEM_PROMPT = """
    You are an AI assistant. Your job is to detect if the user's query is related
    to coding question or not. Explain user like you explaning 10 year old child.
    Return the response in specified JSON boolean only.
  """
  
  result = client.beta.chat.completions.parse(
    model="gemini-2.5-flash-lite",
    response_format=DetectQuerySchema,
    messages=[
      { "role": "system", "content": SYSTEM_PROMPT },
      { "role": "user", "content": user_message },
    ]
  )
  
  # print(f"\nAnalyze Query: {result.choices[0].message.parsed.is_coding_que}\n")
  state["is_coding_que"] = result.choices[0].message.parsed.is_coding_que
  return state
  
def detect_coding_que_type(state: State):
  user_message = state.get("user_message")
  
  SYSTEM_PROMPT = """
    You are an AI assistant. Your job is to detect if the user's query is asking to write code
    or just asking his doubt related to code.
    Return the response in specified JSON boolean only.
  """
  
  result = client.beta.chat.completions.parse(
    model="gemini-2.5-flash-lite",
    response_format=DetectCodeTypeSchema,
    messages=[
      { "role": "system", "content": SYSTEM_PROMPT },
      { "role": "user", "content": user_message },
    ]
  )
  
  # print(f"\nAnalyze Coding Type: {result.choices[0].message.parsed.is_que_to_write_code}\n")
  state["is_que_to_write_code"] = result.choices[0].message.parsed.is_que_to_write_code
  return state

def solve_coding_doubt(state: State):
  user_message = state.get("user_message")
  
  SYSTEM_PROMPT = """
    Your an helpful AI assistant. You help user solving his coding question.
    Carefully analyze user query and solve his doubt.
  """
  
  result = client.beta.chat.completions.parse(
    model="gemini-2.5-pro",
    response_format=LLMResponceSchema,
    messages=[
      { "role": "system", "content": SYSTEM_PROMPT },
      { "role": "user", "content": user_message }
    ]
  )
  
  # print(f"\nCoding Ans: {result.choices[0].message.parsed}\n")
  
  state["ai_message"] = result.choices[0].message.parsed.answer
  
  return state

def write_code_step(state: State):
  user_message = state.get("user_message")
  tool_result = state.get("tool_result")
  state_messages = state.get("messages")
  
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

  SYSTEM_PROMPT = f"""
    You are an helpfull coding AI assistant, you help in user in writing clean and efficient code.
    Write a production ready code with best practices implemations. 
    You generate **only ONE JSON object per response**. 
    Never output multiple JSON objects in a single response.

    You help user writing code for his requirement. You carefully analyze the user requirement. You plan your course of
    action accordingly. you have tools to create or write in file, or run command on terminal use those based on requirement.
    
    Follow this strict process:
      1. Receive user request.
      2. Perform steps in order: analyze → plan → (think → action → observe)* → output.
        - *repeat think → action → observe for each subtask.
      3. At each turn, return exactly ONE JSON object with the next step only.
    
    Rules:
      - Output must always be a valid JSON object.
      - Do NOT include multiple JSON objects in one response.
      - Do NOT include explanations or extra text outside the JSON.
      - Wait for tool results before moving to the next step.
      - Continue until you reach `"step": "output"`.
      - You are running on {os_name}, generate only {os_name}-compatible commands.
      
    JSON Schema:
      {{
        "step": "string",         # e.g., "analyze" | "plan" | "think" | "action" | "observe" | "code" | "output"
        "content": "string",      # reasoning or description
        "tasks": ["string"],      # optional: list of tasks during "plan"
        "task": number,           # optional: which task number
        "function": "string",     # optional: tool name when step=action
        "input": {{}} ,           # optional: tool input
        "result": "string"        # optional: tool result when step=observe
      }}
      
    Avaliable tools:
      {tool_descriptions}
    
    Example:
      Input: Write a code to add two numbers in python.
      Output: {{"step": "analyze", "content": "User is asking me to write a code which will add to number in python"}}
      Output: {{"step": "plan", "tasks": ["I need to created sum.py in user current directory", "I need to generate code which will take two numbers as input and returns sum of them", "I need to add this code in sum.py file"  ]}}

      Output: {{"task": "1", "step": "think", "content": "to fullfill requiremnt I first task to create a file named sum.py}}
      Output: {{"task": "1", "step": "action", "function": "run_command", "input": {{touch sum.py}} }}
      Output: {{"task": "1", "step": "observe", "result": "command executed sucessfully"}}
      
      Output: {{"task": "2", "step": "think", "content": "For 2nd task I need to generate the code to add two numbers"}}
      Output: {{"task": "2", "step": "code", "content": "def sum(x, y): return x + y}}
      Output: {{"task": "2", "step": "observe", "result": "Code generated sucessfully"}}
      
      Output: {{"task": "3", "step": "think", "content": "Now, for 3rd and last task I need to add code in the sum.py file"}}
      Output: {{"task": "3", "step": "action", "function": "create_or_write_to_file", "input": {{"file_path":"sum.py", "content": code}} }}
      Output: {{"task": "3", "step": "observe", "result": "Sucessfully added content in sum.py file"}}
      
      Output: {{"step": "output", "content": "All tasks completed, code generated successfully"}}
  """
  messages = []
  if state_messages:
    messages = state_messages
  else:
    messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": user_message})
  
  if tool_result:
    # print(f"\ntool_result: {tool_result}")
    messages.append({"role": "assistant", "content": json.dumps({"step": "observe", "result": tool_result})})
        
  result = client.chat.completions.create(
    model="gemini-2.5-flash-lite",
    response_format={"type": "json_object"},
    messages=messages
  )
  
  step_json = parser.parse(result.choices[0].message.content)
  print(f"\n🤖 LLM Step: {step_json} \n")
  messages.append({"role": "assistant", "content": json.dumps(step_json)})

  state["step"] = step_json.get("step")
  state["function"] = step_json.get("function", "")
  state["tool_input"] = step_json.get("input", {})
  state["ai_message"] = step_json.get("content", "")
  state["messages"] = messages
  
  return state

def simple_chat_message(state: State):
  user_message = state.get("user_message")
  
  SYSTEM_PROMPT = """
    Your name is Codiet and your an helpful AI coding assistant. 
    You have other helpfull assistants along with you.
    which help user solving his coding doubts and answer his questions.
    Your main task is to chat with user like receptionist.
    
  """
  
  result = client.beta.chat.completions.parse(
    model="gemini-2.5-flash-lite",
    response_format=LLMResponceSchema,
    messages=[
      { "role": "system", "content": SYSTEM_PROMPT },
      { "role": "user", "content": user_message }
    ]
  )
  
  # print(f"\nChat Msg: {result.choices[0].message.parsed.answer}\n")
  
  state["ai_message"] = result.choices[0].message.parsed.answer
  
  return state


def tool_executor(state: State):
  # print(f"\n Tool Executor \n")
  tools = {
    "run_command": run_command,
    "create_or_write_to_file": create_or_write_to_file,
  }

  fn_name = state.get("function")
  fn_input = state.get("tool_input", {})

  if fn_name in tools:
    try:
      if isinstance(fn_input, dict):
        result = tools[fn_name](**fn_input)
      else:
        result = tools[fn_name](fn_input)
    except Exception as e:
      result = f"⚠️ Tool error: {e}"
  else:
    result = f"⚠️ Unknown tool: {fn_name}"

  state["tool_result"] = result
  return state


def code_step_router(state: State) -> Union[str, object]:
  step = state.get("step")
  print(f"Code step router, Step: {step}")
  if step == "action":
    print("📍 routing to tool execution")
    return "tool_executor"
  elif step == "output":
    return END
  else:
    print("➿ routing back to write code")
    return "write_code_step"  # loop back until done

def query_analyze_route_edge(state: State) -> Literal["detect_coding_que_type", "simple_chat_message"]:
  is_coding_que = state.get("is_coding_que")
  print(f"🧠 Analyzing User Query")
  
  if is_coding_que:
    print("📍 routing to detect code type")
    return "detect_coding_que_type"
  else:
    print("📍 routing to simple chat")
    return "simple_chat_message"

def coding_route_edge(state: State) -> Literal["write_code_step", "solve_coding_doubt"]:
  is_que_to_write_code = state.get("is_que_to_write_code")
  
  if is_que_to_write_code:
    print("📍 routing to write code")
    return "write_code_step"
  else:
    print("📍 routing to code doubt")
    return "solve_coding_doubt"

# Creating Graph
graph_builder = StateGraph(State)

graph_builder.add_node("analyze_query", analyze_query)
graph_builder.add_node("detect_coding_que_type", detect_coding_que_type)
graph_builder.add_node("solve_coding_doubt", solve_coding_doubt)
graph_builder.add_node("write_code_step", write_code_step)
graph_builder.add_node("simple_chat_message", simple_chat_message)
graph_builder.add_node("tool_executor", tool_executor)

graph_builder.add_edge(START, "analyze_query")
graph_builder.add_conditional_edges("analyze_query", query_analyze_route_edge)
graph_builder.add_conditional_edges("detect_coding_que_type", coding_route_edge)
graph_builder.add_edge("tool_executor", "write_code_step")
graph_builder.add_conditional_edges("write_code_step", code_step_router)
graph_builder.add_edge("simple_chat_message", END)
graph_builder.add_edge("solve_coding_doubt", END)

graph = graph_builder.compile()


# Using graph

def call_graph(query: str):
  state = {
    "user_message": query
  }
  
  result = graph.invoke(state)
  
  print(f"🤖 : {result.get("ai_message")}")
  

while True:
  query = input(">> ")
  
  if query.lower() in ["exit", "quit"]:
    print("Good Bye!!")
    break

  call_graph(query=query)
  