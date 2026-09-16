from tokenizer.tokenizer import CharacterTokenizer
from tokenizer.bpe_tokenizer import BPETokenizer


# Load training text
with open(
    "data/raw/training.txt",
    "r",
    encoding="utf-8"
) as file:
    text = file.read()


# Character tokenizer
character_tokenizer = CharacterTokenizer(text)

character_tokens = character_tokenizer.encode(text)


# BPE tokenizer
bpe_tokenizer = BPETokenizer(
    text,
    num_merges=50
)

bpe_tokens = bpe_tokenizer.encode(text)


# Calculate statistics
character_count = len(character_tokens)
bpe_count = len(bpe_tokens)

tokens_saved = character_count - bpe_count

reduction_percentage = (
    tokens_saved / character_count
) * 100


print("Tokenizer Comparison")
print("====================")

print(
    "Characters:",
    len(text)
)

print(
    "Character tokens:",
    character_count
)

print(
    "BPE tokens:",
    bpe_count
)

print(
    "Tokens saved:",
    tokens_saved
)

print(
    "Token reduction:",
    f"{reduction_percentage:.2f}%"
)

print(
    "Character vocabulary:",
    character_tokenizer.vocab_size
)

print(
    "BPE vocabulary:",
    bpe_tokenizer.vocab_size
)


print("\nBPE Merge Rules")
print("================")

for number, pair in enumerate(
    bpe_tokenizer.merge_rules,
    start=1
):

    print(
        f"{number:2d}. "
        f"{repr(pair[0])} + "
        f"{repr(pair[1])} → "
        f"{repr(pair[0] + pair[1])}"
    )