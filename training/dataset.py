import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):

    def __init__(
        self,
        conversations,
        tokenizer,
        context_length,
        stride=8
    ):
        self.samples = []

        user_id = tokenizer.token_to_id["<USER>"]
        assistant_id = tokenizer.token_to_id["<ASSISTANT>"]
        end_id = tokenizer.token_to_id["<END>"]

        for conversation in conversations:

            tokens = tokenizer.encode(conversation)

            if len(tokens) < 2:
                continue

            start = 0

            while start < len(tokens) - 1:

                chunk = tokens[
                    start:start + context_length + 1
                ]

                if len(chunk) < 2:
                    break

                x = chunk[:-1]
                y = chunk[1:]

                original_length = len(x)

                loss_mask = torch.zeros(
                    context_length,
                    dtype=torch.float32
                )

                inside_assistant = False

                for i, token in enumerate(y):

                    if token == user_id:
                        inside_assistant = False

                    elif token == assistant_id:
                        inside_assistant = True

                    elif token == end_id:

                        if inside_assistant:
                            loss_mask[i] = 1

                        inside_assistant = False

                    elif inside_assistant:
                        loss_mask[i] = 1

                if original_length < context_length:

                    padding_length = (
                        context_length - original_length
                    )

                    x = x + [0] * padding_length
                    y = y + [0] * padding_length

                # Only keep samples that actually
                # contain assistant targets.
                if loss_mask.sum() > 0:

                    self.samples.append(
                        (
                            torch.tensor(
                                x,
                                dtype=torch.long
                            ),
                            torch.tensor(
                                y,
                                dtype=torch.long
                            ),
                            loss_mask
                        )
                    )

                start += stride

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def split_conversations(
    text,
    validation_ratio=0.10,
    seed=42
):
    import random

    conversations = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    rng = random.Random(seed)
    rng.shuffle(conversations)

    validation_size = int(
        len(conversations) * validation_ratio
    )

    validation_conversations = (
        conversations[:validation_size]
    )

    train_conversations = (
        conversations[validation_size:]
    )

    return (
        train_conversations,
        validation_conversations
    )