from tokenizer.bpe_tokenizer import BPETokenizer


with open(
    "data/raw/training.txt",
    "r",
    encoding="utf-8"
) as file:
    text = file.read()


tokenizer = BPETokenizer(
    text,
    num_merges=10
)


print("BPE Tokenizer Test")
print("==================")

print(
    "Vocabulary size:",
    tokenizer.vocab_size
)


test_text = "The cat"

encoded = tokenizer.encode(
    test_text
)

decoded = tokenizer.decode(
    encoded
)


print("\nOriginal:")
print(test_text)

print("\nEncoded IDs:")
print(encoded)

print("\nDecoded:")
print(decoded)

# =========================
# Unknown text test
# =========================

unknown_text = "The dinosaur 🦖"

unknown_encoded = tokenizer.encode(
    unknown_text
)

unknown_decoded = tokenizer.decode(
    unknown_encoded
)


print("\nUnknown text test")
print("=================")

print("Original:")
print(unknown_text)

print("\nEncoded IDs:")
print(unknown_encoded)

print("\nDecoded:")
print(unknown_decoded)

print("\nLearned merges:")

for number, pair in enumerate(
    tokenizer.merge_rules,
    start=1
):

    print(
        number,
        repr(pair[0]),
        "+",
        repr(pair[1])
    )