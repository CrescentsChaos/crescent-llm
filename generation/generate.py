import torch

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

tokenizer_tokens = checkpoint[
    "tokenizer_tokens"
]

merge_rules = [
    tuple(pair)
    for pair in checkpoint[
        "tokenizer_merge_rules"
    ]
]

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

unk_token = "<UNK>"

unk_id = token_to_id[
    unk_token
]


def merge_pair(tokens, pair):

    first, second = pair

    merged_token = first + second

    result = []

    i = 0

    while i < len(tokens):

        if (
            i < len(tokens) - 1
            and tokens[i] == first
            and tokens[i + 1] == second
        ):

            result.append(
                merged_token
            )

            i += 2

        else:

            result.append(
                tokens[i]
            )

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

                if character in token_to_id:

                    result.append(
                        token_to_id[
                            character
                        ]
                    )

                else:

                    result.append(
                        unk_id
                    )

    return result


def decode(token_ids):

    result = ""

    for token_id in token_ids:

        token = id_to_token.get(
            token_id,
            unk_token
        )

        result += token

    return result

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
    transformer.embedding.token_embedding.embedding.weight
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

formatted_prompt = (
    "<USER> "
    + prompt
    + " <ASSISTANT>"
)

tokens = encode(
    formatted_prompt
)
prompt_length = len(tokens)



print("\nPrompt tokens:")
print(tokens)


print("\nPrompt decoded:")
print(decode(tokens))


# =========================
# Generation
# =========================

max_new_tokens = 150

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

        next_token = torch.argmax(
    next_token_logits,
    dim=-1
).item()

        tokens.append(
            next_token
        )

        if next_token == token_to_id["<END>"]:
            break


# =========================
# Output
# =========================

generated_tokens = tokens[
    prompt_length:
]

generated_text = decode(
    generated_tokens
)

# Remove everything after <END>
if "<END>" in generated_text:

    generated_text = generated_text.split(
        "<END>",
        1
    )[0]

# Remove conversation markers if generated
generated_text = generated_text.replace(
    "<ASSISTANT>",
    ""
)

generated_text = generated_text.replace(
    "<USER>",
    ""
)

print("\nAssistant:")
print(
    generated_text.strip()
)