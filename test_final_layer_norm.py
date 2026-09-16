import torch

from config import ModelConfig
from model.transformer import Transformer


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = Transformer(
    vocab_size=80,
    context_length=ModelConfig.context_length,
    embedding_dim=ModelConfig.embedding_dim,
    num_heads=ModelConfig.num_heads,
    num_layers=ModelConfig.num_layers
).to(device)


x = torch.randint(
    0,
    80,
    (
        2,
        4
    ),
    device=device
)


output = model(x)


print("Final LayerNorm Test")
print("====================")
print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Output device:", output.device)

print(
    "Output mean:",
    output.mean().item()
)

print(
    "Output standard deviation:",
    output.std().item()
)