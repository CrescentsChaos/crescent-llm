from pathlib import Path

from tokenizer.bpe_tokenizer import BPETokenizer


text = Path(
    "data/raw/training.txt"
).read_text(
    encoding="utf-8"
)

tokenizer = BPETokenizer(
    text,
    num_merges=100
)

words = [
    "Python",
    "programming",
    "language",
    "computer",
    "software",
    "development",
    "machine",
    "learning",
]

print("Vocabulary size:")
print(tokenizer.vocab_size)

print("\nTokenization:\n")

for word in words:

    token_ids = tokenizer.encode(word)

    token_strings = [
        tokenizer.id_to_token[token_id]
        for token_id in token_ids
    ]

    print(
        f"{word:15} -> "
        f"{token_strings}"
    )