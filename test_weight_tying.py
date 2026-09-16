import torch

from model.transformer import Transformer, LanguageModelHead
from config import ModelConfig


vocab_size = 80

transformer = Transformer(
    vocab_size=vocab_size,
    context_length=ModelConfig.context_length,
    embedding_dim=ModelConfig.embedding_dim,
    num_heads=ModelConfig.num_heads,
    num_layers=ModelConfig.num_layers
)

lm_head = LanguageModelHead(
    transformer.embedding.token_embedding.embedding.weight
)

input_weight = (
    transformer.embedding
    .token_embedding
    .embedding
    .weight
)

output_weight = lm_head.embedding_weights


print("Weight Tying Test")
print("=================")

print(
    "Input embedding shape:",
    input_weight.shape
)

print(
    "LM head weight shape:",
    output_weight.shape
)

print(
    "Same object:",
    input_weight is output_weight
)

print(
    "Same memory:",
    input_weight.data_ptr() == output_weight.data_ptr()
)