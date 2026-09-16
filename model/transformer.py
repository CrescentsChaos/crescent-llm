import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

    def forward(self, tokens):
        return self.embedding(tokens)


class PositionalEmbedding(nn.Module):

    def __init__(self, context_length, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            context_length,
            embedding_dim
        )

    def forward(self, tokens):
        batch_size, sequence_length = tokens.shape

        positions = torch.arange(
            sequence_length,
            device=tokens.device
        )

        return self.embedding(positions)
class InputEmbedding(nn.Module):

    def __init__(
        self,
        vocab_size,
        context_length,
        embedding_dim
    ):
        super().__init__()

        self.token_embedding = TokenEmbedding(
            vocab_size,
            embedding_dim
        )

        self.position_embedding = PositionalEmbedding(
            context_length,
            embedding_dim
        )

    def forward(self, tokens):

        token_vectors = self.token_embedding(tokens)

        position_vectors = self.position_embedding(tokens)

        return token_vectors + position_vectors
class SelfAttention(nn.Module):

    def __init__(self, embedding_dim, context_length):
        super().__init__()

        self.query = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.key = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.value = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        # Create a lower-triangular mask
        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(
                    context_length,
                    context_length
                )
            )
        )

    def forward(self, x):

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Calculate attention scores
        scores = Q @ K.transpose(-2, -1)

        # Get current sequence length
        sequence_length = x.size(1)

        # Hide future tokens
        scores = scores.masked_fill(
            self.mask[
                :sequence_length,
                :sequence_length
            ] == 0,
            float("-inf")
        )
        attention_weights = torch.softmax(
    scores,dim=-1
)

        return Q, K, V, scores, attention_weights