import torch
import torch.nn.functional as F
import torch.optim as optim
from training.dataset import TextDataset
from config import ModelConfig
from model.transformer import (
    InputEmbedding,
    SelfAttention,
    FeedForward,
    TransformerBlock,
    Transformer,
    LanguageModelHead
)
text = """
The cat is sitting on the grass.
The dog is running through the field.
The bird is flying above the trees.
The fox is walking through the forest.
The rabbit is eating a small green leaf.
"""
dataset = TextDataset(
    text,
    ModelConfig.context_length
)

vocab_size = dataset.tokenizer.vocab_size

print("\nDataset Vocabulary Size:")
print(vocab_size)


transformer = Transformer(
    vocab_size=vocab_size,
    context_length=ModelConfig.context_length,
    embedding_dim=ModelConfig.embedding_dim,
    num_heads=ModelConfig.num_heads,
    num_layers=ModelConfig.num_layers
)

lm_head = LanguageModelHead(
    embedding_dim=ModelConfig.embedding_dim,
    vocab_size=vocab_size
)
optimizer = optim.AdamW(
    list(transformer.parameters()) +
    list(lm_head.parameters()),
    lr=ModelConfig.learning_rate
)


x, y = dataset[0]

x = x.unsqueeze(0)
y = y.unsqueeze(0)

optimizer.zero_grad()

transformer_output = transformer(x)

print("\nTransformer Output:")
print(transformer_output.shape)

logits = lm_head(transformer_output)

print("\nLogits:")
print(logits.shape)

loss = F.cross_entropy(
    logits.view(-1, vocab_size),
    y.view(-1)
)

print("\nReal Dataset Loss:")
print(loss)

loss.backward()

print("\nBackpropagation completed.")

optimizer.step()

print("\nOptimizer step completed.")
