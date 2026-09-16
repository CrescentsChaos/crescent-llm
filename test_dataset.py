from pathlib import Path

from training.dataset import TextDataset


text = Path("data/raw/training.txt").read_text(
    encoding="utf-8"
)

dataset = TextDataset(
    text=text,
    context_length=32
)

print("Vocabulary size:", dataset.tokenizer.vocab_size)
print("Dataset size:", len(dataset))

x, y = dataset[0]

print("\nInput token IDs:")
print(x)

print("\nTarget token IDs:")
print(y)

print("\nInput text:")
print(dataset.tokenizer.decode(x.tolist()))

print("\nTarget text:")
print(dataset.tokenizer.decode(y.tolist()))