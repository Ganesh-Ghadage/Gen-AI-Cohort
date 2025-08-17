from .graph import create_checkpointer_graph

import json

from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.types import Command

config = {"configurable": {"thread_id": "1"}}
DB_URI = "mongodb://admin:admin@localhost:27017"

def main():
  with MongoDBSaver.from_conn_string(DB_URI) as checkpointer:
    
    while True:
      graph = create_checkpointer_graph(checkpointer=checkpointer)
      
      state = graph.get_state(config=config)
      last_messsage = state.values["messages"][-1]
      
      tool_called = last_messsage.tool_calls

      user_query = None
      
      for call in tool_called:
        if call.get("name") == "human_assistance":
          user_query = call.get("args", {}).get("query", "")
            
      print("User is Tying to Ask:", user_query)
      ans = input("Resolution > ")
      
      resume_command = Command(resume={"data": ans})
            
      for event in graph.stream(resume_command, config, stream_mode="values"):
        if "messages" in event:
          event["messages"][-1].pretty_print()
      
if __name__ == "__main__":
  main()