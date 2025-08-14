import os
from dotenv import load_dotenv
from mem0 import Memory
from openai import OpenAI

load_dotenv()

config = {
  "version": "v1.1",
  "embedder": {
    "provider": "gemini",
    "config": {
      "api_key": os.getenv("GEMINI_API_KEY"), 
      "model": "models/gemini-embedding-001",
      "output_dimensionality": 1536
    },
  },
  "llm": {
    "provider": "gemini", 
    "config": {
      "api_key": os.getenv("GEMINI_API_KEY"), 
      "model": "gemini-2.0-flash-001",
      "temperature": 0.2,
      "max_tokens": 2000,
      "top_p": 1.0
    }
  },
  "vector_store": {
    "provider": "qdrant",
    "config": {
      "host": "localhost",
      "port": 6333,
    },
  },
  "graph_store": {
    "provider": "neo4j",
    "config": {
      "url": os.getenv("NEO4J_URL"), 
      "username": os.getenv("NEO4J_USERNAME"), 
      "password": os.getenv("NEO4J_PASSWORD")
    },
  },
}

memory_client = Memory.from_config(config_dict=config)

client = OpenAI(
  api_key=os.getenv("GEMINI_API_KEY"),
  base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def chat(message):
  mem_result = memory_client.search(query=message, user_id="p123")
  
  print("mem_result", mem_result)

  memories = "\n".join([m["memory"] for m in mem_result.get("results")])

  print(f"\n\nMEMORY:\n\n{memories}\n\n")
    
  SYSTEM_PROMPT = f"""
    You are a Memory-Aware Fact Extraction Agent, an advanced AI designed to
    systematically analyze input content, extract structured knowledge, and maintain an
    optimized memory store. Your primary function is information distillation
    and knowledge preservation with contextual awareness.

    Tone: Professional analytical, precision-focused, with clear uncertainty signaling
    
    Memory and Score:
    {memories}
  """
    
  messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": message}
  ]
  
  response = client.chat.completions.create(
    messages=messages,
    model="gemini-2.5-flash"
  )
  
  messages.append({"role":"assistant", "content":response.choices[0].message.content})
  memory_client.add(messages, user_id="p123")
  
  return response.choices[0].message.content



while True:
  query = input(">> ")
  if query.lower() in ['quit', 'exit']:
    print("Good Bye!!")
    break
  
  print(f"🤖: {chat(query)}")