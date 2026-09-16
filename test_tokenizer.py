from tokenizer.tokenizer import CharacterTokenizer


text = "cat is meowing!"

tokenizer = CharacterTokenizer(text)

print("Vocabulary:")
print(tokenizer.chars)

print("\nVocabulary size:")
print(tokenizer.vocab_size)

encoded = tokenizer.encode(text)

print("\nEncoded:")
print(encoded)

decoded = tokenizer.decode(encoded)

print("\nDecoded:")
print(decoded)