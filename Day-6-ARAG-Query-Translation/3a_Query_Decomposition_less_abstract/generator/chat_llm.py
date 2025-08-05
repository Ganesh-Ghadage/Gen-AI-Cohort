from retriever.retrival import recursively_ask

from llm.prompt_templates import FINAL_PROMPT
from llm.client import llm
from utils.output_parser import output_parser

def llm_chat(query: str):
  qa_pairs = recursively_ask(question=query)
  
  final_chain = FINAL_PROMPT | llm | output_parser
  
  answer = final_chain.invoke(
    {
      "question": query,
      "qa_pairs": qa_pairs
    }
  )
  
  return answer