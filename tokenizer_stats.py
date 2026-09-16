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
# Create tokenizer
# =========================

tokenizer = CharacterTokenizer(text)

tokens = tokenizer.encode(text)


# =========================
# Statistics
# =========================

character_count = len(text)

word_count = len(text.split())

token_count = len(tokens)

unique_characters = len(
    set(text)
)


print("Tokenizer Statistics")
print("====================")

print(
    f"Characters:          {character_count}"
)

print(
    f"Words:               {word_count}"
)

print(
    f"Tokens:              {token_count}"
)

print(
    f"Unique characters:   {unique_characters}"
)

print(
    f"Vocabulary size:     {tokenizer.vocab_size}"
)


# =========================
# Tokenization examples
# =========================

examples = [
    "The cat",
    "transformer",
    "machine learning",
    "The bird flies"
]


print("\nTokenization Examples")
print("=====================")

for example in examples:

    encoded = tokenizer.encode(example)

    print(f"\nText:   {example}")
    print(f"Tokens: {len(encoded)}")
    print(f"IDs:    {encoded}")