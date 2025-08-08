import os
from openai import OpenAI

from retriever.retrival import parallel_query_retriver

api_key = os.getenv("GEMINI_API_KEY")

client = OpenAI(
  api_key=api_key,
  base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def llm_chat(query: str):
  search_results =  parallel_query_retriver(query)
  
  context = "\n\n"
  for doc in search_results:
    # print(doc)
    context += f"Page Content: {doc.page_content}\nPage Number: {doc.metadata['page_label']}\nFile Location: {doc.metadata['source']} \n\n"


  SYSTEM_PROMPT = f"""
    You are a helpful AI assistant who answers user query based on the available context retrieved from a PDF file along with page_contents and page number.

    You should only answer the user based on the following context and navigate the user to open the right page number to know more.
    answer should be in details.
    Context:
    {context}
  """

  chat_completion = client.chat.completions.create(
    model="gemini-2.0-flash",
    messages=[
      {"role": "system", "content": SYSTEM_PROMPT},
      {"role": "user", "content": query}
    ],
  )

  return chat_completion.choices[0].message.content