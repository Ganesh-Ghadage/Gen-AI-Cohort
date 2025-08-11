from retriever.retrival import generate_hyde

from llm.prompt_templates import FINAL_PROMPT
from llm.client import llm
from utils.output_parser import output_parser

def llm_chat(query: str):
  context = generate_hyde(question=query)
  
  final_chain = FINAL_PROMPT | llm | output_parser
  
  answer = final_chain.invoke(
    {
      "context": context,
      "question": query
    }
  )
  
  return answer