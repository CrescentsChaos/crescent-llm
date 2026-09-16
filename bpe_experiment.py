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

tokens = [
    tokenizer.id_to_char[token_id]
    for token_id in tokenizer.encode(text)
]


# =========================
# BPE merge function
# =========================

def merge_pair(tokens, pair):

    first, second = pair

    merged_token = first + second

    new_tokens = []

    i = 0

    while i < len(tokens):

        if (
            i < len(tokens) - 1
            and tokens[i] == first
            and tokens[i + 1] == second
        ):
            new_tokens.append(
                merged_token
            )

            i += 2

        else:
            new_tokens.append(
                tokens[i]
            )

            i += 1

    return new_tokens


# =========================
# Perform BPE merges
# =========================

num_merges = 10
vocabulary = set(tokens)
merge_rules = []
print("BPE Training")
print("============")

for merge_number in range(num_merges):

    pairs = Counter(
        zip(
            tokens,
            tokens[1:]
        )
    )

    if not pairs:
        break

    most_common_pair, count = pairs.most_common(1)[0]
    merge_rules.append(
    most_common_pair
)
    first, second = most_common_pair

    merged_token = first + second

    print(
        f"\nMerge {merge_number + 1}:"
    )

    print(
        repr(first),
        "+",
        repr(second),
        "→",
        repr(merged_token)
    )

    print(
        "Frequency:",
        count
    )

    tokens = merge_pair(
        tokens,
        most_common_pair
    )
    vocabulary.add(
    merged_token
)

    print(
        "Token count:",
        len(tokens)
    )
print("\nLearned vocabulary")
print("==================")

print(
    "Vocabulary size:",
    len(vocabulary)
)

for token in sorted(
    vocabulary,
    key=lambda x: (len(x), x)
):
    print(repr(token))
# =========================
# Assign token IDs
# =========================

token_to_id = {
    token: token_id
    for token_id, token in enumerate(
        sorted(
            vocabulary,
            key=lambda x: (len(x), x)
        )
    )
}

id_to_token = {
    token_id: token
    for token, token_id in token_to_id.items()
}


print("\nToken IDs")
print("=========")

for token, token_id in token_to_id.items():
    print(
        token_id,
        repr(token)
    )
print("\nLearned Merge Rules")
print("===================")

for number, pair in enumerate(
    merge_rules,
    start=1
):

    print(
        number,
        repr(pair[0]),
        "+",
        repr(pair[1])
    )
# =========================
# BPE Encoder
# =========================

def encode(text, vocabulary):

    tokens = list(text)

    # Try longer tokens first
    sorted_vocabulary = sorted(
        vocabulary,
        key=len,
        reverse=True
    )

    result = []

    i = 0

    while i < len(tokens):

        matched = False

        for token in sorted_vocabulary:

            token_length = len(token)

            if (
                "".join(
                    tokens[i:i + token_length]
                )
                == token
            ):
                result.append(token)

                i += token_length

                matched = True

                break

        if not matched:

            result.append(tokens[i])

            i += 1

    return result
# =========================
# Test BPE encoding
# =========================

test_text = "The cat"

encoded_tokens = encode(
    test_text,
    vocabulary
)

encoded_ids = [
    token_to_id[token]
    for token in encoded_tokens
]

print("\nBPE Encoding Test")
print("=================")

print("Text:")
print(test_text)

print("\nTokens:")
print(encoded_tokens)

print("\nToken IDs:")
print(encoded_ids)

# =========================
# Proper BPE Encoder
# =========================

def bpe_encode(text, merge_rules):

    tokens = list(text)

    for pair in merge_rules:

        tokens = merge_pair(
            tokens,
            pair
        )

    return tokens
# =========================
# Test proper BPE
# =========================

proper_tokens = bpe_encode(
    "The cat",
    merge_rules
)

proper_ids = [
    token_to_id[token]
    for token in proper_tokens
]

print("\nProper BPE Encoding")
print("===================")

print("Text:")
print("The cat")

print("\nTokens:")
print(proper_tokens)

print("\nToken IDs:")
print(proper_ids)

# =========================
# BPE Decoder
# =========================

def bpe_decode(token_ids, id_to_token):

    tokens = [
        id_to_token[token_id]
        for token_id in token_ids
    ]

    return "".join(tokens)

# =========================
# Test BPE decoding
# =========================

decoded_text = bpe_decode(
    proper_ids,
    id_to_token
)

print("\nBPE Decoding")
print("============")

print("Token IDs:")
print(proper_ids)

print("\nDecoded text:")
print(decoded_text)