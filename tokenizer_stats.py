from tokenizer.bpe_tokenizer import BPETokenizer


# Load training text
with open(
    "data/raw/training.txt",
    "r",
    encoding="utf-8"
) as file:
    text = file.read()


# Train BPE tokenizer
tokenizer = BPETokenizer(
    text,
    num_merges=50
)


print("BPE Vocabulary")
print("==============")

print(
    "Vocabulary size:",
    tokenizer.vocab_size
)


print("\nTokens:")

for token_id, token in tokenizer.id_to_token.items():

    print(
        f"{token_id:3d}  {repr(token)}"
    )


print("\nLearned Merge Rules")
print("===================")

for number, pair in enumerate(
    tokenizer.merge_rules,
    start=1
):

    print(
        f"{number:2d}. "
        f"{repr(pair[0])} + "
        f"{repr(pair[1])}"
        f" → "
        f"{repr(pair[0] + pair[1])}"
    )