import torch

from config import ModelConfig
from model.transformer import Transformer, LanguageModelHead


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================
# Load checkpoint
# =========================

checkpoint = torch.load(
    "model.pt",
    map_location=device
)

vocab_size = checkpoint["vocab_size"]

tokenizer_tokens = checkpoint[
    "tokenizer_tokens"
]

merge_rules = [
    tuple(pair)
    for pair in checkpoint[
        "tokenizer_merge_rules"
    ]
]


# =========================
# Tokenizer
# =========================

token_to_id = {
    token: index
    for index, token
    in enumerate(tokenizer_tokens)
}

id_to_token = {
    index: token
    for index, token
    in enumerate(tokenizer_tokens)
}

special_tokens = [
    "<UNK>",
    "<USER>",
    "<ASSISTANT>",
    "<END>"
]

unk_id = token_to_id["<UNK>"]


def merge_pair(tokens, pair):

    first, second = pair

    merged = first + second

    result = []

    i = 0

    while i < len(tokens):

        if (
            i < len(tokens) - 1
            and tokens[i] == first
            and tokens[i + 1] == second
        ):

            result.append(merged)

            i += 2

        else:

            result.append(tokens[i])

            i += 1

    return result


def encode(text):

    tokens = []

    i = 0

    while i < len(text):

        matched_special = None

        for special_token in special_tokens[1:]:

            if text.startswith(
                special_token,
                i
            ):

                matched_special = special_token

                break

        if matched_special:

            tokens.append(
                matched_special
            )

            i += len(
                matched_special
            )

        else:

            tokens.append(
                text[i]
            )

            i += 1

    for pair in merge_rules:

        tokens = merge_pair(
            tokens,
            pair
        )

    result = []

    for token in tokens:

        if token in token_to_id:

            result.append(
                token_to_id[token]
            )

        else:

            for character in token:

                result.append(
                    token_to_id.get(
                        character,
                        unk_id
                    )
                )

    return result


# =========================
# Model
# =========================

transformer = Transformer(
    vocab_size=vocab_size,
    context_length=ModelConfig.context_length,
    embedding_dim=ModelConfig.embedding_dim,
    num_heads=ModelConfig.num_heads,
    num_layers=ModelConfig.num_layers
).to(device)


lm_head = LanguageModelHead(
    transformer.embedding
    .token_embedding
    .embedding
    .weight
).to(device)


transformer.load_state_dict(
    checkpoint["transformer"]
)

lm_head.load_state_dict(
    checkpoint["lm_head"]
)

transformer.eval()
lm_head.eval()


# =========================
# Test prompt
# =========================

prompt = (
    "<USER> What is Python? "
    "<ASSISTANT> P"
)

tokens = encode(prompt)

print("Prompt:")
print(prompt)

print("\nTokens:")
print(tokens)

print("\nToken strings:")

for token_id in tokens:

    print(
        token_id,
        repr(id_to_token[token_id])
    )


# =========================
# Prediction
# =========================

with torch.no_grad():

    x = torch.tensor(
        [tokens],
        dtype=torch.long,
        device=device
    )

    output = transformer(x)

    logits = lm_head(output)

    next_logits = logits[
        0,
        -1
    ]

    probabilities = torch.softmax(
        next_logits,
        dim=-1
    )

    values, indices = torch.topk(
        probabilities,
        20
    )


print("\nTop 20 predictions:")

for probability, token_id in zip(
    values,
    indices
):

    token = id_to_token[
        token_id.item()
    ]

    print(
        f"{repr(token):20} "
        f"{probability.item() * 100:.2f}%"
    )