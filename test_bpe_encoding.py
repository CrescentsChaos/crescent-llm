from tokenizer.bpe_tokenizer import BPETokenizer


# Load training text
with open(
    "data/raw/training.txt",
    "r",
    encoding="utf-8"
) as file:
    text = file.read()


# Create tokenizer
tokenizer = BPETokenizer(
    text,
    num_merges=50
)


test_text = (
    "The cat watches the rabbit."
)


tokens = tokenizer.encode(
    test_text
)


print("BPE Encoding Test")
print("=================")

print("\nOriginal:")
print(test_text)

print("\nToken IDs:")
print(tokens)

print("\nToken strings:")

for token_id in tokens:

    print(
        token_id,
        repr(
            tokenizer.id_to_token[
                token_id
            ]
        )
    )

print("\nNumber of characters:")
print(len(test_text))

print("\nNumber of BPE tokens:")
print(len(tokens))

print("\nDecoded:")
print(
    tokenizer.decode(tokens)
)