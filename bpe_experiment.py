from collections import Counter

from tokenizer.tokenizer import CharacterTokenizer


# =========================
# Load training text
# =========================

with open(
    "data/raw/training.txt",
    "r",
    encoding="utf-8"
) as file:
    text = file.read()


# =========================
# Character tokenizer
# =========================

tokenizer = CharacterTokenizer(text)

tokens = tokenizer.encode(text)


# =========================
# Count adjacent pairs
# =========================

pairs = Counter(
    zip(tokens, tokens[1:])
)


# =========================
# Find most common pair
# =========================

most_common_pair, count = pairs.most_common(1)[0]

first, second = most_common_pair

print("Most common pair:")
print(
    repr(tokenizer.id_to_char[first]),
    "+",
    repr(tokenizer.id_to_char[second])
)

print("Frequency:", count)


# =========================
# Create merged token
# =========================

merged_token = (
    tokenizer.id_to_char[first]
    +
    tokenizer.id_to_char[second]
)

print("\nNew merged token:")
print(repr(merged_token))