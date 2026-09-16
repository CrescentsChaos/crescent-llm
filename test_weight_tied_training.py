import torch
import torch.nn.functional as F
import torch.optim as optim

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


optimizer = optim.AdamW(
    transformer.parameters(),
    lr=ModelConfig.learning_rate
)


x = torch.randint(
    0,
    vocab_size,
    (2, 8),
    device=device
)

y = torch.randint(
    0,
    vocab_size,
    (2, 8),
    device=device
)


shared_weight = (
    transformer.embedding
    .token_embedding
    .embedding
    .weight
)


weight_before = shared_weight.detach().clone()


optimizer.zero_grad()

transformer_output = transformer(x)

logits = lm_head(
    transformer_output
)

loss = F.cross_entropy(
    logits.view(-1, vocab_size),
    y.view(-1)
)

loss.backward()

gradient_norm = shared_weight.grad.norm().item()

optimizer.step()


weight_after = shared_weight.detach().clone()

weight_change = (
    weight_after - weight_before
).abs().sum().item()


print("Weight-Tied Training Test")
print("=========================")

print(
    "Loss:",
    loss.item()
)

print(
    "Shared weight gradient norm:",
    gradient_norm
)

print(
    "Shared weight change:",
    weight_change
)

print(
    "Weights updated:",
    weight_change > 0
)