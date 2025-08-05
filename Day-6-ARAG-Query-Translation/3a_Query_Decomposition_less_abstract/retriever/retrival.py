from utils.output_parser import DictOutputParser
from config.vector_store import get_vector_store
from llm.prompt_templates import ANSWER_AND_FOLLOUP_PROMPT
from llm.client import llm

qdrant = get_vector_store()

answer_and_followup_chain = ANSWER_AND_FOLLOUP_PROMPT | llm | DictOutputParser()

retriver = qdrant.as_retriever()

def recursively_ask(question, prior_qa="", n=3):
  context = retriver.invoke(question)
  
  response = answer_and_followup_chain.invoke(
    {
      "context": context,
      "question": question,
      "prior_qa": prior_qa,
    }
  )
  
  answer, followup = response["Answer"], response["Followup"]
  
  print(f"Que: {question} \n Ans: {answer} \n followup: {followup} \n\n")
  
  prior_qa += f"Q:{question}nA:{answer}nn"
  n -= 1
  
  if n == 0:
    return prior_qa
  else:
    return recursively_ask(question=followup, prior_qa=prior_qa, n=n)
  
# question = "What is diffrence between require and import?"
# question_answer = recursively_ask(question=question, n=3)

# print(question_answer)