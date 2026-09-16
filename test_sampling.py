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

vocab_size = checkpoint["vocab_size"]
chars = checkpoint["tokenizer_chars"]

char_to_id = {
    char: i
    for i, char in enumerate(chars)
}

id_to_char = {
    i: char
    for i, char in enumerate(chars)
}


# =========================
# Create model
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
# Load weights
# =========================

transformer.load_state_dict(
    checkpoint["transformer"]
)

lm_head.load_state_dict(
    checkpoint["lm_head"]
)

transformer.eval()
lm_head.eval()


# =========================
# Prompt
# =========================

prompt = "The "

tokens = [
    char_to_id.get(
        char,
        char_to_id["<UNK>"]
    )
    for char in prompt
]

x = torch.tensor(
    [tokens],
    dtype=torch.long,
    device=device
)


# =========================
# Get probabilities
# =========================

with torch.no_grad():

    transformer_output = transformer(x)

    logits = lm_head(transformer_output)

    next_token_logits = logits[:, -1, :]

    probabilities = F.softmax(
        next_token_logits,
        dim=-1
    )[0]


# =========================
# Sample 1000 times
# =========================

samples = torch.multinomial(
    probabilities,
    num_samples=1000,
    replacement=True
)


# =========================
# Count samples
# =========================

counts = torch.bincount(
    samples,
    minlength=vocab_size
)


# =========================
# Display
# =========================

print("Prompt:", repr(prompt))

print("\nSampling 1000 times:\n")

for token_id in torch.argsort(
    counts,
    descending=True
):

    token_id = token_id.item()

    count = counts[token_id].item()

    if count == 0:
        continue

    model_probability = (
        probabilities[token_id].item() * 100
    )

    actual_percentage = (
        count / 1000 * 100
    )

    token = id_to_char[token_id]

    print(
        f"{repr(token):8s} "
        f"Model: {model_probability:6.2f}% "
        f"Samples: {actual_percentage:6.2f}% "
        f"({count})"
    )