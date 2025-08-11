from utils.output_parser import output_parser
from config.vector_store import get_vector_store
from llm.prompt_templates import HYDE_PROMPT
from llm.client import llm

qdrant = get_vector_store()

hyde_chain = HYDE_PROMPT | llm | output_parser

retriver = qdrant.as_retriever()

def generate_hyde(question):
  doc = hyde_chain.invoke(question)
  
  context = retriver.invoke(doc)
  
  # print(doc)
  # print(len(context))
  
  return context