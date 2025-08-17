from .graph import create_checkpointer_graph

from langgraph.checkpoint.mongodb import MongoDBSaver

config = {"configurable": {"thread_id": "1"}}
DB_URI = "mongodb://admin:admin@localhost:27017"

def main():
  with MongoDBSaver.from_conn_string(DB_URI) as checkpointer:
    
    while True:
      user_query = input(">> ")
      
      if user_query.lower() in ["exit", "quit"]:
        print("Good Bye!!")
        break
      
      graph = create_checkpointer_graph(checkpointer=checkpointer)
      
      for event in graph.stream({"messages": [{"role": "user", "content": user_query}]}, config, stream_mode="values"):
        if "messages" in event:
          print("Event", event)
          event["messages"][-1].pretty_print()
      
if __name__ == "__main__":
  main()