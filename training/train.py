import os

import torch

from training.dataset import (
    TextDataset,
    split_conversations
)
from training.fast_loader import FastTensorLoader
from tokenizer.bpe_tokenizer import BPETokenizer
from training.engine import train_model
from config import ModelConfig, TrainConfig
from model.transformer import Transformer, LanguageModelHead


# =========================
# Reproducibility / Device
# =========================

torch.manual_seed(TrainConfig.seed)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    # Apple Silicon GPU. AMP/GradScaler stays disabled here (engine.py
    # only enables it for CUDA), so this trains in plain fp32 -- still
    # much faster than CPU for this model size.
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True
    # Let matmuls use TF32 on Ampere+ GPUs (RTX 3060 included) instead
    # of full fp32 -- free throughput for the handful of ops that run
    # outside the autocast region, at essentially no precision cost for
    # a model this size.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# =========================
# Dataset
# =========================

with open(
    "data/raw/training.txt",
    "r",
    encoding="utf-8"
) as file:

    text = file.read()

train_conversations, validation_conversations = split_conversations(
    text,
    validation_ratio=TrainConfig.validation_ratio,
    seed=TrainConfig.seed
)

print("\nTraining conversations:")
print(len(train_conversations))

print("Validation conversations:")
print(len(validation_conversations))


# =========================
# Tokenizer + architecture
# =========================
#
# If a pretrained checkpoint is configured, reuse its exact tokenizer
# and architecture (so the embedding table the pretrained weights were
# learned for still lines up) instead of fitting a fresh tokenizer on
# just the Q&A data. Otherwise, fall back to the original behavior:
# train a tokenizer from scratch on the Q&A training text.

pretrained_checkpoint = None
model_config = {
    "context_length": ModelConfig.context_length,
    "embedding_dim": ModelConfig.embedding_dim,
    "num_heads": ModelConfig.num_heads,
    "num_layers": ModelConfig.num_layers,
    "dropout": ModelConfig.dropout,
    "gradient_checkpointing": ModelConfig.gradient_checkpointing,
}

if TrainConfig.pretrained_checkpoint:

    if not os.path.exists(TrainConfig.pretrained_checkpoint):
        raise FileNotFoundError(
            f"TrainConfig.pretrained_checkpoint is set to "
            f"'{TrainConfig.pretrained_checkpoint}' but that file "
            "doesn't exist. Run training/pretrain.py first, or set "
            "TrainConfig.pretrained_checkpoint = None to train from "
            "scratch."
        )

    print(f"\nLoading pretrained checkpoint: {TrainConfig.pretrained_checkpoint}")

    pretrained_checkpoint = torch.load(
        TrainConfig.pretrained_checkpoint,
        map_location=device
    )

    tokenizer = BPETokenizer.from_saved(
        pretrained_checkpoint["tokenizer_tokens"],
        pretrained_checkpoint["tokenizer_merge_rules"]
    )

    # The pretrained checkpoint's own architecture wins, since the
    # weights we're about to load were trained at that shape.
    model_config = pretrained_checkpoint.get("model_config", model_config)

    vocab_size = pretrained_checkpoint["vocab_size"]

    print("Vocabulary size (from pretrained checkpoint):", vocab_size)

else:

    train_text = "\n".join(
        train_conversations
    )

    tokenizer = BPETokenizer(
        train_text,
        num_merges=TrainConfig.bpe_merges
    )

    vocab_size = tokenizer.vocab_size

    print("\nBPE tokenizer created.")
    print("Vocabulary size:", vocab_size)


# =========================
# Datasets
# =========================

train_dataset = TextDataset(
    train_conversations,
    tokenizer,
    model_config["context_length"]
)

val_dataset = TextDataset(
    validation_conversations,
    tokenizer,
    model_config["context_length"]
)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


# =========================
# DataLoaders
# =========================

train_dataloader = FastTensorLoader(
    train_dataset,
    batch_size=TrainConfig.batch_size,
    shuffle=True,
    device=device,
    pin_memory=(device.type == "cuda")
)

val_dataloader = FastTensorLoader(
    val_dataset,
    batch_size=TrainConfig.batch_size,
    shuffle=False,
    device=device,
    pin_memory=(device.type == "cuda")
)


# =========================
# Model
# =========================

transformer = Transformer(
    vocab_size=vocab_size,
    context_length=model_config["context_length"],
    embedding_dim=model_config["embedding_dim"],
    num_heads=model_config["num_heads"],
    num_layers=model_config["num_layers"],
    dropout=model_config["dropout"],
    gradient_checkpointing=model_config.get("gradient_checkpointing", False)
).to(device)

if pretrained_checkpoint is not None:
    transformer.load_state_dict(pretrained_checkpoint["transformer"])
    print("Initialized transformer weights from pretrained checkpoint.")

lm_head = LanguageModelHead(
    transformer.embedding.token_embedding.embedding.weight
).to(device)


print("\nModel device:")
print("Transformer:", next(transformer.parameters()).device)

num_params = sum(p.numel() for p in transformer.parameters())
print(f"Trainable parameters: {num_params:,}")


# =========================
# Fine-tune
# =========================

train_model(
    transformer,
    lm_head,
    train_dataloader,
    val_dataloader,
    vocab_size,
    device,
    learning_rate=TrainConfig.learning_rate,
    weight_decay=TrainConfig.weight_decay,
    grad_clip_norm=TrainConfig.grad_clip_norm,
    num_epochs=TrainConfig.num_epochs,
    warmup_steps=TrainConfig.warmup_steps,
    early_stopping_patience=TrainConfig.early_stopping_patience,
    checkpoint_path="model.pt",
    checkpoint_extra={
        "vocab_size": vocab_size,
        "tokenizer_tokens": tokenizer.tokens,
        "tokenizer_merge_rules": tokenizer.merge_rules,
        "model_config": model_config,
    },
    label="model",
    gradient_accumulation_steps=TrainConfig.gradient_accumulation_steps,
    label_smoothing=TrainConfig.label_smoothing
)

print("\nModel saved to model.pt")
