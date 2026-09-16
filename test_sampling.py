import torch
import torch.nn.functional as F

from model.transformer import Transformer, LanguageModelHead
from tokenizer.bpe_tokenizer import BPETokenizer
from config import ModelConfig


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# Load checkpoint
# --------------------------------------------------

checkpoint = torch.load(
    "model.pt",
    map_location=device,
    weights_only=False
)

vocab_size = checkpoint["vocab_size"]

print("Vocabulary size:", vocab_size)


# --------------------------------------------------
# Reconstruct BPE tokenizer
# --------------------------------------------------

tokenizer = BPETokenizer.__new__(
    BPETokenizer
)

tokenizer.special_tokens = [
    "<UNK>",
    "<USER>",
    "<ASSISTANT>",
    "<END>"
]

tokenizer.unk_token = "<UNK>"

tokenizer.tokens = checkpoint[
    "tokenizer_tokens"
]

tokenizer.merge_rules = checkpoint[
    "tokenizer_merge_rules"
]

tokenizer.token_to_id = {
    token: index
    for index, token in enumerate(
        tokenizer.tokens
    )
}

tokenizer.id_to_token = {
    index: token
    for token, index in tokenizer.token_to_id.items()
}

tokenizer.unk_id = tokenizer.token_to_id[
    tokenizer.unk_token
]

tokenizer.vocab_size = len(
    tokenizer.tokens
)


# --------------------------------------------------
# Create model
# --------------------------------------------------

transformer = Transformer(
    vocab_size=vocab_size,
    context_length=ModelConfig.context_length,
    embedding_dim=ModelConfig.embedding_dim,
    num_heads=ModelConfig.num_heads,
    num_layers=ModelConfig.num_layers
).to(device)

transformer.load_state_dict(
    checkpoint["transformer"]
)


lm_head = LanguageModelHead(
    transformer.embedding.token_embedding.embedding.weight
).to(device)

lm_head.load_state_dict(
    checkpoint["lm_head"]
)

transformer.eval()
lm_head.eval()


# --------------------------------------------------
# Sampling function
# --------------------------------------------------

def sample_next_token(
    logits,
    temperature=0.8,
    top_k=10
):

    logits = logits / temperature

    if top_k is not None:

        values, indices = torch.topk(
            logits,
            min(
                top_k,
                logits.size(-1)
            )
        )

        filtered_logits = torch.full_like(
            logits,
            float("-inf")
        )

        filtered_logits.scatter_(
            -1,
            indices,
            values
        )

        logits = filtered_logits

    probabilities = F.softmax(
        logits,
        dim=-1
    )

    token = torch.multinomial(
        probabilities,
        num_samples=1
    )

    return token.item()


# --------------------------------------------------
# Generate
# --------------------------------------------------

prompt = (
    "<USER> What is Python? "
    "<ASSISTANT> Python is a programming language"
)

tokens = tokenizer.encode(
    prompt
)

print("\nPrompt:")
print(prompt)

print("\nInitial tokens:")
print(tokens)

print("\nInitial token strings:")

for token_id in tokens:

    print(
        token_id,
        repr(
            tokenizer.id_to_token.get(
                token_id,
                "<UNK>"
            )
        )
    )


print("\nGenerating...\n")


with torch.no_grad():

    for _ in range(30):

        input_tokens = tokens[
            -ModelConfig.context_length:
        ]

        x = torch.tensor(
            [input_tokens],
            dtype=torch.long,
            device=device
        )

        hidden = transformer(x)

        logits = lm_head(hidden)

        next_logits = logits[
            0,
            -1
        ]

        next_token = torch.argmax(
    next_logits
).item()

        tokens.append(
            next_token
        )

        if next_token == tokenizer.token_to_id[
            "<END>"
        ]:

            break


print("Generated:")

print(
    tokenizer.decode(tokens)
)