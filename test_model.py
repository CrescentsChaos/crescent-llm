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

(Q, K, V, scores, attention_weights, attention_output) = attention(x)
residual_output = x + attention_output
normalized_output = attention.layer_norm(
    residual_output
)
ffn_hidden = attention.fc1(
    normalized_output
)
ffn_activated = attention.gelu(
    ffn_hidden
)

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

print("\nMasked scores - Batch 0, Head 0:")
print(scores[0, 0])

print("\nAttention Weights - Batch 0, Head 0:")
print(attention_weights[0, 0])

print("\nAttention Output:")
print(attention_output.shape)

print("\nResidual Output:")
print(residual_output.shape)

print("\nNormalized Output:")
print(normalized_output.shape)

print("\nNormalized Mean:")
print(normalized_output.mean())

print("\nNormalized Standard Deviation:")
print(normalized_output.std())

print("\nFFN Hidden:")
print(ffn_hidden.shape)

print("\nFFN After GELU:")
print(ffn_activated.shape)