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

        nn.init.normal_(
            self.embedding.weight,
            mean=0.0,
            std=0.02
        )

    def forward(self, tokens):
        return self.embedding(tokens)
        
class RMSNorm(nn.Module):

    def __init__(self, embedding_dim, eps=1e-8):
        super().__init__()

        self.eps = eps

        self.scale = nn.Parameter(
            torch.ones(embedding_dim)
        )

    def forward(self, x):

        rms = torch.sqrt(
            torch.mean(
                x ** 2,
                dim=-1,
                keepdim=True
            ) + self.eps
        )

        x = x / rms

        x = x * self.scale

        return x

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

    def forward(self, tokens):

        token_vectors = self.token_embedding(tokens)

        token_vectors = self.token_embedding(tokens)

        return token_vectors
class RotaryEmbedding(nn.Module):

    def __init__(self, head_dim, context_length):
        super().__init__()

        assert head_dim % 2 == 0

        frequencies = 1.0 / (
            10000 ** (
                torch.arange(
                    0,
                    head_dim,
                    2
                ).float() / head_dim
            )
        )

        positions = torch.arange(
            context_length
        ).float()

        angles = torch.outer(
            positions,
            frequencies
        )

        self.register_buffer(
            "cos",
            torch.cos(angles)
        )

        self.register_buffer(
            "sin",
            torch.sin(angles)
        )

    def forward(self, Q, K):

        sequence_length = Q.shape[2]

        cos = self.cos[:sequence_length]
        sin = self.sin[:sequence_length]

        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        Q_even = Q[..., 0::2]
        Q_odd = Q[..., 1::2]

        K_even = K[..., 0::2]
        K_odd = K[..., 1::2]

        Q_rotated = torch.stack(
            [
                Q_even * cos - Q_odd * sin,
                Q_even * sin + Q_odd * cos
            ],
            dim=-1
        )

        K_rotated = torch.stack(
            [
                K_even * cos - K_odd * sin,
                K_even * sin + K_odd * cos
            ],
            dim=-1
        )

        Q_rotated = Q_rotated.flatten(-2)
        K_rotated = K_rotated.flatten(-2)

        return Q_rotated, K_rotated
class SelfAttention(nn.Module):

    def __init__(self, embedding_dim, num_heads, context_length):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        assert embedding_dim % num_heads == 0

        self.head_dim = embedding_dim // num_heads
        self.attention_dropout = nn.Dropout(
    0.1
)
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
        self.rotary_embedding = RotaryEmbedding(
            self.head_dim,
            context_length
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

        Q, K = self.rotary_embedding(
            Q,
            K
        )

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

        attention_weights = self.attention_dropout(
            attention_weights
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

        return attention_output

class FeedForward(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()

        hidden_dim = 4 * embedding_dim

        self.gate = nn.Linear(
            embedding_dim,
            hidden_dim
        )

        self.up = nn.Linear(
            embedding_dim,
            hidden_dim
        )

        self.down = nn.Linear(
            hidden_dim,
            embedding_dim
        )

    def forward(self, x):

        gate = torch.nn.functional.silu(
            self.gate(x)
        )

        up = self.up(x)

        x = gate * up

        x = self.down(x)

        return x

class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        context_length
    ):
        super().__init__()

        self.layer_norm_1 = RMSNorm(
    embedding_dim
)

        self.attention = SelfAttention(
            embedding_dim,
            num_heads,
            context_length
        )

        self.layer_norm_2 = RMSNorm(
    embedding_dim
)

        self.feed_forward = FeedForward(
            embedding_dim
        )
    def forward(self, x):

        # Normalize before attention
        normalized_x = self.layer_norm_1(x)

        # Self-attention
        attention_output = self.attention(
    normalized_x
)

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
        self.final_layer_norm = RMSNorm(
    embedding_dim
)
    def forward(self, tokens):

        x = self.embedding(tokens)

        for block in self.blocks:

            x = block(x)

        x = self.final_layer_norm(x)

        return x

class LanguageModelHead(nn.Module):

    def __init__(
        self,
        embedding_weights
    ):
        super().__init__()

        self.embedding_weights = embedding_weights

    def forward(self, x):

        logits = x @ self.embedding_weights.t()

        return logits