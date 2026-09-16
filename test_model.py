import torch

from config import ModelConfig
from model.transformer import (
    InputEmbedding,
    SelfAttention
)


embedding = InputEmbedding(
    vocab_size=ModelConfig.vocab_size,
    context_length=ModelConfig.context_length,
    embedding_dim=ModelConfig.embedding_dim
)

attention = SelfAttention(
    ModelConfig.embedding_dim,
    ModelConfig.num_heads,
    ModelConfig.context_length
)


tokens = torch.tensor([
    [6, 14, 11, 1, 9, 7],
    [9, 7, 25, 1, 6, 14]
])


x = embedding(tokens)

Q, K, V, scores = attention(x)


print("Input:")
print(x.shape)

print("\nQuery:")
print(Q.shape)

print("\nKey:")
print(K.shape)

print("\nValue:")
print(V.shape)

print("\nAttention Scores:")
print(scores.shape)