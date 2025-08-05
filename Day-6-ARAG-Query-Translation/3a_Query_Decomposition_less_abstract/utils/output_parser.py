from langchain_core.output_parsers import StrOutputParser
import json
import re

class DictOutputParser(StrOutputParser):
  def parse(self, text: str) -> dict:
    # Remove code block markers like ```json ... ```
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    
    try:
      return json.loads(cleaned)
    except json.JSONDecodeError:
      raise ValueError(f"Failed to parse JSON: {cleaned}")


def output_parser(x):
  "This helper function parses the LLM output, prints it, and returns it."
  parsed_output = StrOutputParser().invoke(x)
  # print("n" + parsed_output + "n")

  return parsed_output