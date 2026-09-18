import os

import torch
from torch.utils.data import DataLoader

from tokenizer.bpe_tokenizer import BPETokenizer
from training.pretrain_dataset import PlainTextDataset
from training.engine import train_model
from config import ModelConfig, PretrainConfig
from model.transformer import Transformer, LanguageModelHead


# =========================
# Reproducibility / Device
# =========================

torch.manual_seed(PretrainConfig.seed)

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


# =========================
# Corpus
# =========================

if not os.path.exists(PretrainConfig.corpus_path):
    raise FileNotFoundError(
        f"No pretraining corpus found at '{PretrainConfig.corpus_path}'. "
        "Point PretrainConfig.corpus_path at a plain text file (a few MB "
        "of ordinary prose works well — e.g. a subset of the TinyStories "
        "dataset) and run this script again."
    )

with open(
    PretrainConfig.corpus_path,
    "r",
    encoding="utf-8"
) as file:

    corpus_text = file.read()

print(f"\nCorpus length: {len(corpus_text):,} characters")


# =========================
# Tokenizer
# =========================
#
# BPE merge-learning is O(num_merges * text_length) in this
# implementation, so the vocabulary is fit on a bounded sample rather
# than the full corpus -- the sample is still large enough to capture
# the common patterns, and this keeps tokenizer fitting fast regardless
# of how large a corpus you point this at.

tokenizer_fit_text = corpus_text[:PretrainConfig.tokenizer_fit_chars]

print(
    f"Fitting BPE tokenizer on {len(tokenizer_fit_text):,} characters "
    f"({PretrainConfig.bpe_merges} merges)..."
)

tokenizer = BPETokenizer(
    tokenizer_fit_text,
    num_merges=PretrainConfig.bpe_merges
)

vocab_size = tokenizer.vocab_size

print("Vocabulary size:", vocab_size)

print("Encoding full corpus...")

token_ids = tokenizer.encode(corpus_text)

print(f"Total tokens: {len(token_ids):,}")


# =========================
# Train / validation split
# =========================
#
# This is just running text, not a set of independent examples to
# shuffle -- so the split is a straight positional cut rather than a
# random sample of conversations.

split_index = int(
    len(token_ids) * (1 - PretrainConfig.validation_ratio)
)

train_ids = token_ids[:split_index]
val_ids = token_ids[split_index:]

train_dataset = PlainTextDataset(train_ids, ModelConfig.context_length)
val_dataset = PlainTextDataset(val_ids, ModelConfig.context_length)

print("Training chunks:", len(train_dataset))
print("Validation chunks:", len(val_dataset))


# =========================
# DataLoaders
# =========================

train_dataloader = DataLoader(
    train_dataset,
    batch_size=PretrainConfig.batch_size,
    shuffle=True,
    pin_memory=(device.type == "cuda")
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=PretrainConfig.batch_size,
    shuffle=False,
    pin_memory=(device.type == "cuda")
)


# =========================
# Model
# =========================

transformer = Transformer(
    vocab_size=vocab_size,
    context_length=ModelConfig.context_length,
    embedding_dim=ModelConfig.embedding_dim,
    num_heads=ModelConfig.num_heads,
    num_layers=ModelConfig.num_layers,
    dropout=ModelConfig.dropout,
    gradient_checkpointing=ModelConfig.gradient_checkpointing
).to(device)

lm_head = LanguageModelHead(
    transformer.embedding.token_embedding.embedding.weight
).to(device)

num_params = sum(p.numel() for p in transformer.parameters())
print(f"\nTrainable parameters: {num_params:,}")


# =========================
# Pretrain
# =========================

model_config = {
    "context_length": ModelConfig.context_length,
    "embedding_dim": ModelConfig.embedding_dim,
    "num_heads": ModelConfig.num_heads,
    "num_layers": ModelConfig.num_layers,
    "dropout": ModelConfig.dropout,
    "gradient_checkpointing": ModelConfig.gradient_checkpointing,
}

train_model(
    transformer,
    lm_head,
    train_dataloader,
    val_dataloader,
    vocab_size,
    device,
    learning_rate=PretrainConfig.learning_rate,
    weight_decay=PretrainConfig.weight_decay,
    grad_clip_norm=PretrainConfig.grad_clip_norm,
    num_epochs=PretrainConfig.num_epochs,
    warmup_steps=PretrainConfig.warmup_steps,
    early_stopping_patience=PretrainConfig.early_stopping_patience,
    checkpoint_path=PretrainConfig.checkpoint_path,
    checkpoint_extra={
        "vocab_size": vocab_size,
        "tokenizer_tokens": tokenizer.tokens,
        "tokenizer_merge_rules": tokenizer.merge_rules,
        "model_config": model_config,
    },
    label="pretrained model",
    gradient_accumulation_steps=PretrainConfig.gradient_accumulation_steps,
    label_smoothing=PretrainConfig.label_smoothing
)

print(f"\nPretrained checkpoint saved to {PretrainConfig.checkpoint_path}")
print(
    "Set TrainConfig.pretrained_checkpoint = "
    f"'{PretrainConfig.checkpoint_path}' in config.py, then run "
    "training/train.py to fine-tune it on your Q&A data."
)
