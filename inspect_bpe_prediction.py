import torch
import torch.nn.functional as F

from config import ModelConfig
from model.transformer import Transformer, LanguageModelHead
from tokenizer.bpe_tokenizer import BPETokenizer


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Load checkpoint
checkpoint = torch.load(
    "model.pt",
    map_location=device
)

vocab_size = checkpoint["vocab_size"]


# Reconstruct the exact tokenizer used during training
tokenizer = BPETokenizer(
    "",
    num_merges=0
)

tokenizer.tokens = checkpoint["tokenizer_tokens"]
tokenizer.merge_rules = [
    tuple(pair)
    for pair in checkpoint["tokenizer_merge_rules"]
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

tokenizer.vocab_size = len(
    tokenizer.tokens
)

tokenizer.unk_token = "<UNK>"
tokenizer.unk_id = tokenizer.token_to_id[
    tokenizer.unk_token
]

tokenizer.special_tokens = [
    "<UNK>",
    "<USER>",
    "<ASSISTANT>",
    "<END>"
]


# Create model
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


# Load trained weights
transformer.load_state_dict(
    checkpoint["transformer"]
)

lm_head.load_state_dict(
    checkpoint["lm_head"]
)


transformer.eval()
lm_head.eval()


# Prompt
prompt = "<USER> What is Python? <ASSISTANT> Python is"

tokens = tokenizer.encode(prompt)


print("BPE Prediction Inspection")
print("=========================")

print("\nCheckpoint vocabulary size:")
print(vocab_size)

print("\nTokenizer vocabulary size:")
print(tokenizer.vocab_size)

print("\nPrompt:")
print(prompt)

print("\nPrompt tokens:")
print(tokens)

print("\nPrompt token strings:")

for token_id in tokens:
    print(
        token_id,
        repr(
            tokenizer.id_to_token.get(
                token_id,
                "<UNKNOWN>"
            )
        )
    )


# Convert to tensor
x = torch.tensor(
    [tokens],
    dtype=torch.long,
    device=device
)


# Model prediction
with torch.no_grad():

    transformer_output = transformer(x)

    logits = lm_head(
        transformer_output
    )

    next_token_logits = (
        logits[:, -1, :]
    )

    probabilities = F.softmax(
        next_token_logits,
        dim=-1
    )


# Get top 10
top_probabilities, top_indices = torch.topk(
    probabilities,
    10,
    dim=-1
)


print("\nTop 10 next-token predictions:")
print("--------------------------------")

for probability, token_id in zip(
    top_probabilities[0],
    top_indices[0]
):

    token_id = token_id.item()
    probability = probability.item()

    token = tokenizer.id_to_token.get(
        token_id,
        "<UNKNOWN>"
    )

    print(
        f"{token_id:3d} "
        f"{repr(token):15s} "
        f"{probability * 100:6.2f}%"
    )