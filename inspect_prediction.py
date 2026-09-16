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

prompt = input("Enter text: ")


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
# Model prediction
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
# Show predictions
# =========================

top_k = min(10, vocab_size)

values, indices = torch.topk(
    probabilities,
    top_k
)


print("\nPrompt:")
print(prompt)

print("\nTop predictions for the NEXT character:")

for probability, token_id in zip(values, indices):

    token = id_to_char[token_id.item()]
    logit = next_token_logits[0, token_id].item()

    print(
        f"{repr(token):8s} "
        f"Logit: {logit:8.4f} "
        f"Probability: {probability.item() * 100:6.2f}%"
    )