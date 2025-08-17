import os
from dotenv import load_dotenv
from typing import Annotated
from langchain.chat_models import init_chat_model

from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

load_dotenv()

class State(TypedDict):
  messages: Annotated[list, add_messages]
  
llm = init_chat_model(model_provider="google_genai", model="gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY"))

def chatbot(state: State):
  return {"messages": [llm.invoke(state["messages"])]}

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# graph = graph_builder.compile()

def create_checkpointer_graph(checkpointer):
  return graph_builder.compile(checkpointer=checkpointer)