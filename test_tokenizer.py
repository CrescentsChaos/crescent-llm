from tokenizer.tokenizer import CharacterTokenizer


text = "The cat is sitting."


tokenizer = CharacterTokenizer(text)


print("Vocabulary size:", tokenizer.vocab_size)

print("\nVocabulary:")
for token_id, token in tokenizer.id_to_char.items():
    print(token_id, repr(token))


encoded = tokenizer.encode(text)

print("\nOriginal text:")
print(text)

print("\nEncoded tokens:")
print(encoded)

print("\nDecoded text:")
print(tokenizer.decode(encoded))


test_text = "The cat 🐱"

encoded_unknown = tokenizer.encode(test_text)

print("\nText with unknown character:")
print(test_text)

print("\nEncoded:")
print(encoded_unknown)

print("\nDecoded:")
print(tokenizer.decode(encoded_unknown))