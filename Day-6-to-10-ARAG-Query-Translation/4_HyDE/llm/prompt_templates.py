from langchain_core.prompts import PromptTemplate

HYDE_PROMPT = PromptTemplate(
  input_variables=["question"],
  template="""
    You're an AI language assistant. 
    Your task is to generate hypothetical document answering users question.
    Generate document which will anser the user query include all the fancy keyword related to topic in it.
    
    Question: {question}
    Output:
  """
)


FINAL_PROMPT = PromptTemplate(
  input_variables=["context", "question"],
  template="""
    Answer the question in the below context.
    Your response should be comprehensive and not contradicted with the following context.
    If the context is not relevant to the question, say "I don't know":
    {context}

    Question: {question}
  """
)
