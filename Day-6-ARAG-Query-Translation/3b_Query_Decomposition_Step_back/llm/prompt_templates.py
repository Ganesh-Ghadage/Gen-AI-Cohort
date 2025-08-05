from langchain_core.prompts import PromptTemplate

STEP_BACK_PROMPT = PromptTemplate(
  input_variables=["question"],
  template="""
    You are an expert software engineer. 
    Your task is to rephrase the given question into a more general form that is easier to answer.

    # Example 1
    Question: How to create file in node.js?
    Output: What is FS module in node.js?

    # Example 2
    Question: How to optimize browser cache in Node.js?
    Output: What are the different caching options?

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
