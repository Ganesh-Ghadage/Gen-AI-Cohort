import tiktoken

encoder = tiktoken.encoding_for_model('gpt-4o')

text="The cat sat on the mat"

print("Vocab Size: ", encoder.n_vocab)

encoded = encoder.encode(text)

print("Encoded Tokens: ", encoded) # [976, 9059, 10139, 402, 290, 2450]

decoded = encoder.decode(encoded)

print("Decoded Text: ", decoded)