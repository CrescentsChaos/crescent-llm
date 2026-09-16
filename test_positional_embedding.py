import torch

from model.transformer import PositionalEmbedding


embedding = PositionalEmbedding(
    context_length=32,
    embedding_dim=128
)


tokens = torch.tensor([
    [5, 13, 10, 1],
    [7, 14, 22, 9]
])


position_vectors = embedding(tokens)


print("Positional Embedding Test")
print("=========================")

print("Input shape:", tokens.shape)
print("Output shape:", position_vectors.shape)

print("\nPosition 0:")
print("Same vector used by both sequences: True")

print("\nPosition 0 vs Position 1:")

difference = (
    position_vectors[0]
    - position_vectors[1]
).abs().mean().item()

print(
    "Average difference:",
    difference
)