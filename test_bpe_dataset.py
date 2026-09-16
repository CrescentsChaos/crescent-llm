from training.dataset import TextDataset


# Load training text
with open(
    "data/raw/training.txt",
    "r",
    encoding="utf-8"
) as file:
    text = file.read()


# Create dataset
dataset = TextDataset(
    text,
    context_length=32
)


print("BPE Dataset Test")
print("================")

print(
    "Vocabulary size:",
    dataset.tokenizer.vocab_size
)

print(
    "Number of tokens:",
    len(dataset.tokens)
)

print(
    "Number of samples:",
    len(dataset)
)


# Get one training sample
x, y = dataset[0]


print("\nSample shapes:")

print(
    "x shape:",
    x.shape
)

print(
    "y shape:",
    y.shape
)


print("\nFirst 10 input tokens:")

print(
    x[:10].tolist()
)


print("\nFirst 10 target tokens:")

print(
    y[:10].tolist()
)


# Decode the sample
print("\nDecoded input:")

print(
    dataset.tokenizer.decode(
        x.tolist()
    )
)

print("\nDecoded target:")

print(
    dataset.tokenizer.decode(
        y.tolist()
    )
)