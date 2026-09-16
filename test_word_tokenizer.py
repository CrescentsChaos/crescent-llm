from tokenizer.word_tokenizer import WordTokenizer


text = "The cat is sitting on the grass."


tokenizer = WordTokenizer(text)


print("Vocabulary size:", tokenizer.vocab_size)

print("\nVocabulary:")

for token_id, word in tokenizer.id_to_word.items():
    print(token_id, repr(word))


encoded = tokenizer.encode(text)


print("\nOriginal text:")
print(text)

print("\nEncoded tokens:")
print(encoded)

print("\nNumber of tokens:")
print(len(encoded))

print("\nDecoded text:")
print(tokenizer.decode(encoded))


unknown_text = "The elephant is running"


unknown_encoded = tokenizer.encode(
    unknown_text
)


print("\nUnknown-word example:")
print(unknown_text)

print("\nEncoded:")
print(unknown_encoded)

print("\nDecoded:")
print(tokenizer.decode(unknown_encoded))