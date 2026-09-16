import torch
import torch.nn.functional as F
import torch.optim as optim
from training.dataset import (
    TextDataset,
    split_conversations
)
import time
from torch.utils.data import DataLoader
from tokenizer.bpe_tokenizer import BPETokenizer
from config import ModelConfig
from model.transformer import Transformer, LanguageModelHead


# =========================
# Device
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# =========================
# Dataset
# =========================

with open(
    "data/raw/training.txt",
    "r",
    encoding="utf-8"
) as file:

    text = file.read()


# =========================
# Split conversations first
# =========================

train_conversations, validation_conversations = split_conversations(
    text,
    validation_ratio=0.10,
    seed=42
)

print("\nTraining conversations:")
print(len(train_conversations))

print("Validation conversations:")
print(len(validation_conversations))

train_text = "\n".join(
    train_conversations
)

tokenizer = BPETokenizer(
    train_text,
    num_merges=100
)

vocab_size = tokenizer.vocab_size

print("\nBPE tokenizer created.")
print("Vocabulary size:", vocab_size)

train_dataset = TextDataset(
    train_conversations,
    tokenizer,
    ModelConfig.context_length,
    stride=8
)

val_dataset = TextDataset(
    validation_conversations,
    tokenizer,
    ModelConfig.context_length,
    stride=8
)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


# =========================
# DataLoaders
# =========================

train_dataloader = DataLoader(
    train_dataset,
    batch_size=ModelConfig.batch_size,
    shuffle=True
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=ModelConfig.batch_size,
    shuffle=False
)


# =========================
# Model
# =========================

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


print("\nModel device:")
print("Transformer:", next(transformer.parameters()).device)
print("LM Head:", next(lm_head.parameters()).device)


# =========================
# Optimizer
# =========================

optimizer = optim.AdamW(
    transformer.parameters(),
    lr=ModelConfig.learning_rate,
    weight_decay=0.0
)


# =========================
# Training
# =========================

epochs = 30


print("\nStarting training...")
training_start = time.perf_counter()
best_validation_loss = float("inf")
for epoch in range(epochs):
    if device.type == "cuda":
        torch.cuda.synchronize()
    epoch_start = time.perf_counter()
    total_loss = 0.0

    for x, y, loss_mask in train_dataloader:

        x = x.to(device)
        y = y.to(device)
        loss_mask = loss_mask.to(device)

        optimizer.zero_grad()

        transformer_output = transformer(x)

        logits = lm_head(transformer_output)

        losses = F.cross_entropy(
            logits.view(-1, vocab_size),
            y.view(-1),
            reduction="none"
        )

        losses = losses.view(y.shape)

        loss = (
            losses * loss_mask
        ).sum() / loss_mask.sum()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            transformer.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()


    average_loss = total_loss / len(train_dataloader)
    # Validation
    transformer.eval()
    lm_head.eval()

    validation_loss = 0.0

    with torch.no_grad():

        for x, y, loss_mask in val_dataloader:

            x = x.to(device)
            y = y.to(device)
            loss_mask = loss_mask.to(device)
            transformer_output = transformer(x)
            logits = lm_head(transformer_output)

            losses = F.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1),
                reduction="none"
            )

            losses = losses.view(
                y.shape
            )

            loss = (
                losses * loss_mask
            ).sum() / loss_mask.sum()

            validation_loss += loss.item()


    average_validation_loss = (
        validation_loss / len(val_dataloader)
    )


    # Save best model
    if average_validation_loss < best_validation_loss:

        best_validation_loss = average_validation_loss

        torch.save(
            {
                "transformer": transformer.state_dict(),
                "lm_head": lm_head.state_dict(),
                "vocab_size": vocab_size,
                "tokenizer_tokens": tokenizer.tokens,
                "tokenizer_merge_rules": tokenizer.merge_rules
            },
            "model.pt"
        )

        print("Best model saved!")


    # Update learning rate

    transformer.train()
    lm_head.train()
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    epoch_time = time.perf_counter() - epoch_start

    print(
        f"Epoch {epoch + 1:3d}/{epochs} "
        f"Train Loss: {average_loss:.4f} "
    f"Val Loss: {average_validation_loss:.4f} "
        f"Time: {epoch_time:.3f}s"
    )
if device.type == "cuda":
    torch.cuda.synchronize()    
total_time = time.perf_counter() - training_start
print(f"\nTotal training time: {total_time:.2f}s")

print("\nModel saved to model.pt")