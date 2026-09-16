import torch

from config import ModelConfig
from model.transformer import InputEmbedding


embedding = InputEmbedding(
    vocab_size=ModelConfig.vocab_size,
    context_length=ModelConfig.context_length,
    embedding_dim=ModelConfig.embedding_dim
)


tokens = torch.tensor([
    [6, 14, 11, 1],
    [9, 7, 25, 1]
])


output = embedding(tokens)


print("Input shape:")
print(tokens.shape)

print("\nOutput shape:")
print(output.shape)

print("\nNumber of parameters:")

print(
    sum(
        parameter.numel()
        for parameter in embedding.parameters()
    )
)