import torch

from config import ModelConfig
from model.transformer import Transformer, LanguageModelHead


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

vocab_size = 80


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


x = torch.randint(
    0,
    vocab_size,
    (
        2,
        8
    ),
    device=device
)


transformer_output = transformer(x)

logits = lm_head(
    transformer_output
)


print("Weight-Tied Forward Test")
print("========================")

print(
    "Input shape:",
    x.shape
)

print(
    "Transformer output shape:",
    transformer_output.shape
)

print(
    "Logits shape:",
    logits.shape
)

print(
    "Device:",
    logits.device
)