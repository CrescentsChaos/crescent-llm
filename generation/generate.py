import torch
import torch.nn.functional as F

from config import ModelConfig
from model.transformer import Transformer, LanguageModelHead


# =========================
# Device
# =========================

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


vocab_size = checkpoint[
    "vocab_size"
]

tokenizer_tokens = checkpoint[
    "tokenizer_tokens"
]

merge_rules = checkpoint[
    "tokenizer_merge_rules"
]


# =========================
# Rebuild tokenizer
# =========================

token_to_id = {
    token: token_id
    for token_id, token
    in enumerate(tokenizer_tokens)
}

id_to_token = {
    token_id: token
    for token_id, token
    in enumerate(tokenizer_tokens)
}


unk_token = "<UNK>"

unk_id = token_to_id[
    unk_token
]


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


def encode(text):

    words = text.split()

    result = []

    for word in words:

        tokens = list(word)

        # Apply BPE merge rules
        for pair in merge_rules:

            tokens = merge_pair(
                tokens,
                pair
            )

        for token in tokens:

            result.append(
                token_to_id.get(
                    token,
                    unk_id
                )
            )

        # Preserve space
        result.append(
            token_to_id[" "]
        )

    # Remove final space
    if result:

        result.pop()

    return result


def decode(token_ids):

    return "".join(
        id_to_token.get(
            token_id,
            unk_token
        )
        for token_id in token_ids
    )


# =========================
# Build model
# =========================

transformer = Transformer(
    vocab_size=vocab_size,
    context_length=ModelConfig.context_length,
    embedding_dim=ModelConfig.embedding_dim,
    num_heads=ModelConfig.num_heads,
    num_layers=ModelConfig.num_layers
).to(device)


lm_head = LanguageModelHead(
    embedding_dim=ModelConfig.embedding_dim,
    vocab_size=vocab_size
).to(device)


# =========================
# Load model weights
# =========================

transformer.load_state_dict(
    checkpoint["transformer"]
)

lm_head.load_state_dict(
    checkpoint["lm_head"]
)


transformer.eval()
lm_head.eval()


print("BPE model loaded.")
print("Device:", device)
print("Vocabulary size:", vocab_size)


# =========================
# Prompt
# =========================

prompt = input(
    "Enter a prompt: "
)


tokens = encode(prompt)


print("\nPrompt tokens:")
print(tokens)


print("\nPrompt decoded:")
print(decode(tokens))


# =========================
# Generation
# =========================

max_new_tokens = 100

with torch.no_grad():

    for _ in range(
        max_new_tokens
    ):

        input_tokens = tokens[
            -ModelConfig.context_length:
        ]

        x = torch.tensor(
            [input_tokens],
            dtype=torch.long,
            device=device
        )

        transformer_output = transformer(
            x
        )

        logits = lm_head(
            transformer_output
        )

        next_token_logits = (
            logits[:, -1, :]
        )

        temperature = 0.8

        top_k = 5

        scaled_logits = (
            next_token_logits
            / temperature
        )

        top_k_values, top_k_indices = (
            torch.topk(
                scaled_logits,
                top_k,
                dim=-1
            )
        )

        probabilities = F.softmax(
            top_k_values,
            dim=-1
        )

        sampled_index = torch.multinomial(
            probabilities,
            num_samples=1
        )

        next_token = (
            top_k_indices[
                0,
                sampled_index
            ].item()
        )

        tokens.append(
            next_token
        )


# =========================
# Output
# =========================

generated_text = decode(
    tokens
)

print("\nGenerated text:")
print(generated_text)