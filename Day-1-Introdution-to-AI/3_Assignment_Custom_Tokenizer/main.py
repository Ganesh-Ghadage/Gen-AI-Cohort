from Tokenizer import tokenizer

text = "The cat sat on the mat"

encoded = tokenizer.encode(text)

print("Encoded : " , encoded)

decoded = tokenizer.decode(encoded)

print("Decoded : ", decoded)