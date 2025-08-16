from openai import OpenAI
from dotenv import load_dotenv
import os
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langsmith.wrappers import wrap_openai
from pydantic import BaseModel
from typing import Literal

load_dotenv()

#schema
class DetectQuerySchema(BaseModel):
  is_coding_que: bool

class LLMResponceSchema(BaseModel):
  answer: str

class State(TypedDict):
  user_message: str
  ai_message: str
  is_coding_que: bool
  

client = wrap_openai(OpenAI(
  api_key=os.getenv("GEMINI_API_KEY"),
  base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
))

def analyze_query(state: State):
  user_message = state.get("user_message")
  
  SYSTEM_PROMPT = """
    You are an AI assistant. Your job is to detect if the user's query is related
    to coding question or not.
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
  
  state["is_coding_que"] = result.choices[0].message.parsed.is_coding_que
  return state
  
  
def solve_coding_que(state: State):
  user_message = state.get("user_message")
  
  SYSTEM_PROMT = """
    Your an helpful AI assistant. You help user solving his coding question.
    Carefully analyze user query and solve his doubt.
  """
  
  result = client.beta.chat.completions.parse(
    model="gemini-2.5-pro",
    response_format=LLMResponceSchema,
    messages=[
      { "role": "system", "content": SYSTEM_PROMT },
      { "role": "user", "content": user_message }
    ]
  )
  
  print(f"Coding Ans: {result.choices[0].message.parsed}")
  
  state["ai_message"] = result.choices[0].message.parsed.answer
  
  return state

def simple_chat_message(state: State):
  user_message = state.get("user_message")
  
  SYSTEM_PROMT = """
    Your an helpful AI assistant. You chat with user.
  """
  
  result = client.beta.chat.completions.parse(
    model="gemini-2.5-flash-lite",
    response_format=LLMResponceSchema,
    messages=[
      { "role": "system", "content": SYSTEM_PROMT },
      { "role": "user", "content": user_message }
    ]
  )
  
  print(f"Chat Msg: {result.choices[0].message.parsed}")
  
  state["ai_message"] = result.choices[0].message.parsed.answer
  
  return state

def route_edge(state: State) -> Literal["solve_coding_que", "simple_chat_message"]:
  is_coding_que = state.get("is_coding_que")
  
  if is_coding_que:
    return "solve_coding_que"
  else:
    return "simple_chat_message"

graph_builder = StateGraph(State)

graph_builder.add_node("analyze_query", analyze_query)
graph_builder.add_node("solve_coding_que", solve_coding_que)
graph_builder.add_node("simple_chat_message", simple_chat_message)
graph_builder.add_node("route_edge", route_edge)

graph_builder.add_edge(START, "analyze_query")
graph_builder.add_conditional_edges("analyze_query", route_edge)
graph_builder.add_edge("solve_coding_que", END)
graph_builder.add_edge("simple_chat_message", END)

graph = graph_builder.compile()


# Using graph

def call_graph():
  state = {
    "user_message": "What is pydantic in python",
    "ai_message": "",
    "is_coding_que": False
  }
  
  result = graph.invoke(state)
  
  print(f"🤖 : {result}")
  

call_graph()
  