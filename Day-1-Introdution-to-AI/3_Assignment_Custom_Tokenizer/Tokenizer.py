class Tokenizer:
  def encode(self, text):
    wordToken = []
    
    for char in text:
      wordToken.append(ord(char))
    
    return wordToken
  
  def decode(self, tokens):
    chars = []
    for token in tokens:
      chars.append(chr(token))
      
    return "".join(chars)
  
tokenizer = Tokenizer()