import torch
import torch.nn.functional as F
import torch.optim as optim
import time
from torch.utils.data import DataLoader

from training.dataset import TextDataset
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

with open("data/raw/training.txt", "r", encoding="utf-8") as file:
    text = file.read()


dataset = TextDataset(
    text,
    ModelConfig.context_length
)

vocab_size = dataset.tokenizer.vocab_size

print("\nDataset created.")
print("Vocabulary size:", vocab_size)
print("Number of samples:", len(dataset))


train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(
    dataset,
    [train_size, val_size]
)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


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
    embedding_dim=ModelConfig.embedding_dim,
    vocab_size=vocab_size
).to(device)


print("\nModel device:")
print("Transformer:", next(transformer.parameters()).device)
print("LM Head:", next(lm_head.parameters()).device)


# =========================
# Optimizer
# =========================

optimizer = optim.AdamW(
    list(transformer.parameters()) +
    list(lm_head.parameters()),
    lr=ModelConfig.learning_rate
)


# =========================
# Training
# =========================

epochs = 100


print("\nStarting training...")
training_start = time.perf_counter()

for epoch in range(epochs):
    if device.type == "cuda":
        torch.cuda.synchronize()
    epoch_start = time.perf_counter()
    total_loss = 0.0

    for x, y in train_dataloader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        transformer_output = transformer(x)

        logits = lm_head(transformer_output)

        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            y.view(-1)
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()


    average_loss = total_loss / len(train_dataloader)
    # Validation
    transformer.eval()
    lm_head.eval()

    validation_loss = 0.0

    with torch.no_grad():

        for x, y in val_dataloader:

            x = x.to(device)
            y = y.to(device)

            transformer_output = transformer(x)
            logits = lm_head(transformer_output)

            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1)
            )

            validation_loss += loss.item()

    average_validation_loss = (
        validation_loss / len(val_dataloader)
    )


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

torch.save(
    {
        "transformer": transformer.state_dict(),
        "lm_head": lm_head.state_dict(),
        "vocab_size": vocab_size,
        "tokenizer_chars": dataset.tokenizer.chars
    },
    "model.pt"
)

print("\nModel saved to model.pt")