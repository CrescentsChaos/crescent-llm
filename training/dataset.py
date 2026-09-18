import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """
    Builds one training window per conversational turn, always anchored
    at that turn's <USER> token so the model actually sees the question
    it's supposed to be answering.

    Earlier versions of this dataset slid a fixed-size window across
    the whole tokenized conversation at a fixed stride, regardless of
    where <USER>/<ASSISTANT> boundaries fell. For conversations longer
    than context_length (the common case, since most conversations
    tokenize to well over context_length tokens), that meant the vast
    majority of windows started mid-conversation, after the question
    had already scrolled out of view — the model was mostly trained on
    "continue this text" rather than "answer this question". Anchoring
    every window at a real <USER> position fixes that, and as a bonus
    mines every turn of a multi-turn conversation as its own example.
    """

    def __init__(
        self,
        conversations,
        tokenizer,
        context_length
    ):
        self.samples = []

        user_id = tokenizer.token_to_id["<USER>"]
        assistant_id = tokenizer.token_to_id["<ASSISTANT>"]
        end_id = tokenizer.token_to_id["<END>"]

        for conversation in conversations:

            tokens = tokenizer.encode(conversation)

            if len(tokens) < 2:
                continue

            turn_starts = [
                i for i, token in enumerate(tokens)
                if token == user_id
            ]

            # Lines with no <USER> tag at all can't be turned into a
            # question -> answer example; skip them.
            if not turn_starts:
                continue

            for start in turn_starts:

                # A turn runs from its <USER> tag up to the next one
                # (or the end of the conversation). Chunks longer than
                # context_length are truncated from the front, i.e.
                # the question is always kept intact and only a very
                # long answer's tail is ever cut off.
                chunk = tokens[
                    start:start + context_length + 1
                ]

                if len(chunk) < 2:
                    continue

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