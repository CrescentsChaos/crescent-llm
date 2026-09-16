import torch
import torch.nn as nn
import math


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

    def __init__(self, embedding_dim, num_heads, context_length):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        assert embedding_dim % num_heads == 0

        self.head_dim = embedding_dim // num_heads

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
        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim
        )
        self.layer_norm = nn.LayerNorm(
            embedding_dim
        )
        self.fc1 = nn.Linear(
            embedding_dim,
            4 * embedding_dim
        )
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(
            4 * embedding_dim,
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

        # Split embedding dimension into multiple heads
        Q = Q.view(
            Q.shape[0],
            Q.shape[1],
            self.num_heads,
            self.head_dim
        )

        K = K.view(
            K.shape[0],
            K.shape[1],
            self.num_heads,
            self.head_dim
        )

        V = V.view(
            V.shape[0],
            V.shape[1],
            self.num_heads,
            self.head_dim
        )
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        scores = (
        Q @ K.transpose(-2, -1)
    ) / math.sqrt(self.head_dim)

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
        # Convert scores into attention probabilities
        attention_weights = torch.softmax(
            scores,
            dim=-1
        )
        attention_output = attention_weights @ V

        # Move sequence dimension before heads
        attention_output = attention_output.transpose(1, 2)

        # Combine all heads
        attention_output = attention_output.contiguous().view(
            attention_output.shape[0],
            attention_output.shape[1],
            self.embedding_dim
        )
        attention_output = self.output_projection(
            attention_output
        )

        return (Q, K, V, scores, attention_weights, attention_output)

class FeedForward(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()

        self.fc1 = nn.Linear(
            embedding_dim,
            4 * embedding_dim
        )

        self.gelu = nn.GELU()

        self.fc2 = nn.Linear(
            4 * embedding_dim,
            embedding_dim
        )

    def forward(self, x):

        x = self.fc1(x)

        x = self.gelu(x)

        x = self.fc2(x)

        return x

class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        context_length
    ):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(
            embedding_dim
        )

        self.attention = SelfAttention(
            embedding_dim,
            num_heads,
            context_length
        )

        self.layer_norm_2 = nn.LayerNorm(
            embedding_dim
        )

        self.feed_forward = FeedForward(
            embedding_dim
        )
    def forward(self, x):

        # Normalize before attention
        normalized_x = self.layer_norm_1(x)

        # Self-attention
        (
        Q,
        K,
        V,
        scores,
        attention_weights,
        attention_output
    ) = self.attention(normalized_x)

        # Residual connection
        x = x + attention_output

        # Normalize before feed-forward
        normalized_x = self.layer_norm_2(x)

        # Feed-forward network
        feed_forward_output = self.feed_forward(
            normalized_x
        )

        # Residual connection
        x = x + feed_forward_output

        return x

class Transformer(nn.Module):

    def __init__(
        self,
        vocab_size,
        context_length,
        embedding_dim,
        num_heads,
        num_layers
    ):
        super().__init__()

        self.embedding = InputEmbedding(
            vocab_size,
            context_length,
            embedding_dim
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim,
                    num_heads,
                    context_length
                )
                for _ in range(num_layers)
            ]
        )
    def forward(self, tokens):

        x = self.embedding(tokens)

        for block in self.blocks:
            x = block(x)

        return x

class LanguageModelHead(nn.Module):

    def __init__(
        self,
        embedding_dim,
        vocab_size
    ):
        super().__init__()

        self.output_projection = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, x):

        logits = self.output_projection(x)

        return logits