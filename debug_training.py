import torch
import torch.nn.functional as F
import torch.optim as optim

from config import ModelConfig
from tokenizer.bpe_tokenizer import BPETokenizer
from training.dataset import TextDataset, split_conversations
from model.transformer import Transformer, LanguageModelHead


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# Load data
# --------------------------------------------------

with open(
    "data/raw/training.txt",
    "r",
    encoding="utf-8"
) as file:

    text = file.read()


train_conversations, _ = split_conversations(
    text,
    validation_ratio=0.10,
    seed=42
)

train_conversations = train_conversations[:20]

train_text = "\n".join(
    train_conversations
)


# --------------------------------------------------
# Tokenizer
# --------------------------------------------------

tokenizer = BPETokenizer(
    train_text,
    num_merges=100
)

print(
    "Vocabulary size:",
    tokenizer.vocab_size
)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

dataset = TextDataset(
    train_conversations,
    tokenizer,
    ModelConfig.context_length,
    stride=8
)

print(
    "Dataset samples:",
    len(dataset)
)


# --------------------------------------------------
# One sample
# --------------------------------------------------

x, y, loss_mask = dataset[0]

print("\nFirst sample:")
print("x shape:", x.shape)
print("y shape:", y.shape)
print("mask:", loss_mask)
print("mask sum:", loss_mask.sum().item())


# --------------------------------------------------
# Model
# --------------------------------------------------

model = Transformer(
    vocab_size=tokenizer.vocab_size,
    context_length=ModelConfig.context_length,
    embedding_dim=ModelConfig.embedding_dim,
    num_heads=ModelConfig.num_heads,
    num_layers=ModelConfig.num_layers
).to(device)


lm_head = LanguageModelHead(
    model.embedding.token_embedding.embedding.weight
).to(device)


optimizer = optim.AdamW(
    model.parameters(),
    lr=ModelConfig.learning_rate
)


# --------------------------------------------------
# One batch
# --------------------------------------------------

x = x.unsqueeze(0).to(device)
y = y.unsqueeze(0).to(device)
loss_mask = loss_mask.unsqueeze(0).to(device)


# --------------------------------------------------
# Before training
# --------------------------------------------------

model.train()
lm_head.train()

with torch.no_grad():

    hidden = model(x)

    logits = lm_head(hidden)

    losses = F.cross_entropy(
        logits.view(-1, tokenizer.vocab_size),
        y.view(-1),
        reduction="none"
    )

    losses = losses.view(y.shape)

    loss = (
        losses * loss_mask
    ).sum() / loss_mask.sum()


print("\nBefore training:")
print("Loss:", loss.item())
print(
    "Logit max:",
    logits.max().item()
)
print(
    "Logit min:",
    logits.min().item()
)
print(
    "Logit mean:",
    logits.mean().item()
)


# --------------------------------------------------
# One optimization step
# --------------------------------------------------

optimizer.zero_grad()

hidden = model(x)

logits = lm_head(hidden)

losses = F.cross_entropy(
    logits.view(-1, tokenizer.vocab_size),
    y.view(-1),
    reduction="none"
)

losses = losses.view(y.shape)

loss = (
    losses * loss_mask
).sum() / loss_mask.sum()


print("\nLoss before backward:")
print(loss.item())


loss.backward()


# --------------------------------------------------
# Gradient information
# --------------------------------------------------

embedding_gradient = (
    model.embedding
    .token_embedding
    .embedding
    .weight
    .grad
)

print("\nGradient:")
print(
    "Gradient mean:",
    embedding_gradient.mean().item()
)
print(
    "Gradient abs mean:",
    embedding_gradient.abs().mean().item()
)
print(
    "Gradient max:",
    embedding_gradient.abs().max().item()
)


optimizer.step()


# --------------------------------------------------
# After one step
# --------------------------------------------------

with torch.no_grad():

    hidden = model(x)

    logits = lm_head(hidden)

    losses = F.cross_entropy(
        logits.view(-1, tokenizer.vocab_size),
        y.view(-1),
        reduction="none"
    )

    losses = losses.view(y.shape)

    new_loss = (
        losses * loss_mask
    ).sum() / loss_mask.sum()


print("\nAfter one optimization step:")
print(
    "Loss:",
    new_loss.item()
)