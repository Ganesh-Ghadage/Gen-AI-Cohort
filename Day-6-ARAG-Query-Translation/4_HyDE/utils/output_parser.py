from langchain_core.output_parsers import StrOutputParser

def output_parser(x):
  "This helper function parses the LLM output, prints it, and returns it."
  parsed_output = StrOutputParser().invoke(x)
  # print("n" + parsed_output + "n")

  return parsed_output