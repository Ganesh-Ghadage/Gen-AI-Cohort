from langchain_core.prompts import PromptTemplate

ANSWER_AND_FOLLOUP_PROMPT = PromptTemplate(
  input_variables=["context", "question", "prior_qa"],
  template="""
    You are an AI assistant who is expert in breaking down the user query and asking the follow up question on it.
    based on the context given to you. you need to break down the user query and ask question for each part of the query.
    
    Context: {context}
    Question: {question}

    Also, consider any prior questions and answers you've generated:
    Prior questions and answers: {prior_qa}

    You must respond in this exact JSON format, and only output the JSON — no explanation or markdown:
    {{
      "Answer": "<your answer>",
      "Followup": "<your follow-up question>"
    }}
  """
)


FINAL_PROMPT = PromptTemplate(
  input_variables=["question", "qa_pairs"],
  template="""
    Provide a comprehensive answer to the following question based on the subquestions you answered.
    Question: {question}

    Here are the subquestions and answers you provided:
    {qa_pairs}

    Answer: 
  """
)
