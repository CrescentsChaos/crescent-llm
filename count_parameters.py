from config import ModelConfig
from model.transformer import Transformer, LanguageModelHead


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


all_parameters = list(transformer.parameters()) + list(
    lm_head.parameters()
)

unique_parameters = {
    id(parameter): parameter
    for parameter in all_parameters
}

total_parameters = sum(
    parameter.numel()
    for parameter in unique_parameters.values()
)


print("Unique Parameter Count")
print("======================")

print(
    "Unique parameters:",
    total_parameters
)

print(
    "Unique parameters (K):",
    total_parameters / 1_000
)

print(
    "Unique parameters (M):",
    total_parameters / 1_000_000
)