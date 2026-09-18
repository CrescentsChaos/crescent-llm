import torch
from torch.utils.data import Dataset


class PlainTextDataset(Dataset):
    """
    Plain next-token-prediction dataset for pretraining on unstructured
    text. Unlike TextDataset (used for Q&A fine-tuning), there's no
    <USER>/<ASSISTANT> structure to anchor on and no loss masking to
    speak of — every token in the corpus is a valid training target.
    The token stream is simply packed into back-to-back, non-
    overlapping context_length-sized chunks.
    """

    def __init__(self, token_ids, context_length):
        self.samples = []

        for start in range(0, len(token_ids) - 1, context_length):

            chunk = token_ids[start:start + context_length + 1]

            if len(chunk) < 2:
                continue

            x = chunk[:-1]
            y = chunk[1:]

            original_length = len(x)

            if original_length < context_length:

                padding_length = context_length - original_length

                x = x + [0] * padding_length
                y = y + [0] * padding_length

                loss_mask = torch.tensor(
                    [1.0] * original_length + [0.0] * padding_length,
                    dtype=torch.float32
                )
            else:
                loss_mask = torch.ones(
                    context_length,
                    dtype=torch.float32
                )

            if loss_mask.sum() == 0:
                continue

            self.samples.append(
                (
                    torch.tensor(x, dtype=torch.long),
                    torch.tensor(y, dtype=torch.long),
                    loss_mask
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
