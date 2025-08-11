from utils.output_parser import output_parser
from config.vector_store import get_vector_store
from llm.prompt_templates import STEP_BACK_PROMPT
from llm.client import llm

qdrant = get_vector_store()

step_back_chain = STEP_BACK_PROMPT | llm | output_parser

retriver = qdrant.as_retriever()

def get_broder_context(question):
  step_back_que = step_back_chain.invoke(question)
  
  context = retriver.invoke(step_back_que)
  
  # print(context)
  
  return context