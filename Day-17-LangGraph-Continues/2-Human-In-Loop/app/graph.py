import os
from dotenv import load_dotenv

from typing import Annotated
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.prebuilt import ToolNode, tools_condition


load_dotenv()

class State(TypedDict):
  messages: Annotated[list, add_messages]
  
llm = init_chat_model(model_provider="google_genai", model="gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY"))

@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]
  
tools = [human_assistance]
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
  message = llm_with_tools.invoke(state["messages"])
  assert len(message.tool_calls) <= 1
  return {"messages": [message]}

graph_builder = StateGraph(State)

tool_node = ToolNode(tools=tools)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)

# graph = graph_builder.compile()

def create_checkpointer_graph(checkpointer):
  return graph_builder.compile(checkpointer=checkpointer)