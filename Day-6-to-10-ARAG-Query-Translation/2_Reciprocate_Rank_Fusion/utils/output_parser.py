from langchain_core.output_parsers import BaseOutputParser
from typing import List

class LineListOutputParser(BaseOutputParser[List[str]]):
  """Output parser for a list of lines."""

  def parse(self, text: str) -> List[str]:
    lines = text.strip().split("\n")
    return list(filter(None, lines)) 
  
output_parser = LineListOutputParser()