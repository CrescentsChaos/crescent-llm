import torch


checkpoint = torch.load(
    "model.pt",
    map_location="cpu"
)


print("BPE Checkpoint Test")
print("===================")

print(
    "Vocabulary size:",
    checkpoint["vocab_size"]
)

tokens = checkpoint[
    "tokenizer_tokens"
]

merge_rules = checkpoint[
    "tokenizer_merge_rules"
]


print(
    "Number of tokenizer tokens:",
    len(tokens)
)

print(
    "Number of merge rules:",
    len(merge_rules)
)


print("\nFirst 10 tokenizer tokens:")

for token_id, token in enumerate(tokens[:10]):

    print(
        token_id,
        repr(token)
    )


print("\nFirst 10 merge rules:")

for number, pair in enumerate(
    merge_rules[:10],
    start=1
):

    print(
        number,
        repr(pair[0]),
        "+",
        repr(pair[1]),
        "→",
        repr(pair[0] + pair[1])
    )