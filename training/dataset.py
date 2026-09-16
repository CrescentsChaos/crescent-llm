import torch

from tokenizer.tokenizer import CharacterTokenizer


class TextDataset:

    def __init__(self, text, context_length):

        self.tokenizer = CharacterTokenizer(text)

        self.tokens = torch.tensor(
            self.tokenizer.encode(text),
            dtype=torch.long
        )

        self.context_length = context_length

    def __len__(self):
        return len(self.tokens) - self.context_length

    def __getitem__(self, index):

        x = self.tokens[
            index:index + self.context_length
        ]

        y = self.tokens[
            index + 1:index + self.context_length + 1
        ]

        return x, y