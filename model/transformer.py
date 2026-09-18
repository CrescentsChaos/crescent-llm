import torch
import torch.nn as nn
import torch.utils.checkpoint


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

    def __init__(self, embedding_dim, eps=1e-6):
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

class InputEmbedding(nn.Module):
    """
    Token embedding only. Positional information is injected later by
    RotaryEmbedding inside SelfAttention, so no separate learned
    absolute-position embedding is needed here.
    """

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

        return token_vectors


class KVCache:
    """
    Holds the accumulated per-layer key/value tensors for a single
    generation run, so autoregressive decoding only has to run the
    transformer over the *new* token(s) at each step instead of
    recomputing every previous position's attention from scratch.

    Usage: create one `KVCache(num_layers)`, then feed the model the
    full prompt once (populating the cache), then feed it one new
    token at a time, passing the same cache object through each call.
    Each layer's entry is a (key, value) tensor pair of shape
    (batch, num_heads, seen_so_far, head_dim); `update` appends the
    newly computed keys/values for a layer and returns the full
    (past + new) tensors to attend over.
    """

    def __init__(self, num_layers):
        self.layers = [None] * num_layers

    def seq_len(self):
        if self.layers[0] is None:
            return 0
        return self.layers[0][0].shape[2]

    def update(self, layer_idx, k, v):
        cached = self.layers[layer_idx]

        if cached is None:
            self.layers[layer_idx] = (k, v)
        else:
            cached_k, cached_v = cached
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
            self.layers[layer_idx] = (k, v)

        return self.layers[layer_idx]


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

        # persistent=False: these are deterministic functions of
        # head_dim/context_length, so there's no need to bloat the
        # checkpoint by saving and loading them.
        self.register_buffer(
            "cos",
            torch.cos(angles),
            persistent=False
        )

        self.register_buffer(
            "sin",
            torch.sin(angles),
            persistent=False
        )

    def forward(self, Q, K, start_pos=0):

        sequence_length = Q.shape[2]

        # `start_pos` lets a query/key pair be rotated to its true
        # absolute position even when Q/K only cover a slice of the
        # sequence (e.g. a single new token during cached generation,
        # where position 0 of this call is really position
        # `start_pos` of the full sequence).
        cos = self.cos[start_pos:start_pos + sequence_length]
        sin = self.sin[start_pos:start_pos + sequence_length]

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

    def __init__(self, embedding_dim, num_heads, context_length, dropout=0.1):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        assert embedding_dim % num_heads == 0

        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.attention_dropout = nn.Dropout(dropout)
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

        # Create a lower-triangular mask (bool, so no per-forward
        # float comparison is needed to use it). Row q, column k is
        # True iff position q is allowed to attend to position k
        # (k <= q). This same buffer is reused verbatim for cached
        # generation: with a cache, queries live at rows
        # [start_pos, start_pos + T_new) and keys span columns
        # [0, start_pos + T_new), so slicing
        # mask[start_pos:start_pos+T_new, :start_pos+T_new] gives
        # exactly the right causal pattern without rebuilding anything.
        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(
                    context_length,
                    context_length,
                    dtype=torch.bool
                )
            ),
            persistent=False
        )

    def forward(self, x, kv_cache=None, layer_idx=None):

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

        # How many positions have already been processed (and cached)
        # before this call. Zero for a normal training/prefill forward
        # pass; equal to however many tokens are already cached during
        # incremental generation.
        start_pos = kv_cache.seq_len() if kv_cache is not None else 0

        Q, K = self.rotary_embedding(
            Q,
            K,
            start_pos=start_pos
        )

        if kv_cache is not None:
            # Append this call's new keys/values to whatever was
            # cached from earlier calls, and attend over the full
            # (past + new) set.
            K, V = kv_cache.update(layer_idx, K, V)

        scores = (
            Q @ K.transpose(-2, -1)
        ) * self.scale

        # Number of new query positions in this call, and the total
        # number of keys they may attend to (past + new).
        query_length = Q.shape[2]
        key_length = K.shape[2]

        scores = scores.masked_fill(
            ~self.mask[
                start_pos:start_pos + query_length,
                :key_length
            ],
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
        context_length,
        dropout=0.1
    ):
        super().__init__()

        self.layer_norm_1 = RMSNorm(
    embedding_dim
)

        self.attention = SelfAttention(
            embedding_dim,
            num_heads,
            context_length,
            dropout=dropout
        )

        self.layer_norm_2 = RMSNorm(
    embedding_dim
)

        self.feed_forward = FeedForward(
            embedding_dim
        )

        # Dropout on the output of each sub-layer, before it's added
        # back into the residual stream (standard pre-norm placement).
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x, kv_cache=None, layer_idx=None):

        # Normalize before attention
        normalized_x = self.layer_norm_1(x)

        # Self-attention
        attention_output = self.attention(
            normalized_x,
            kv_cache=kv_cache,
            layer_idx=layer_idx
        )

        # Residual connection
        x = x + self.residual_dropout(attention_output)

        # Normalize before feed-forward
        normalized_x = self.layer_norm_2(x)

        # Feed-forward network
        feed_forward_output = self.feed_forward(
            normalized_x
        )

        # Residual connection
        x = x + self.residual_dropout(feed_forward_output)

        return x

class Transformer(nn.Module):

    def __init__(
        self,
        vocab_size,
        context_length,
        embedding_dim,
        num_heads,
        num_layers,
        dropout=0.1,
        gradient_checkpointing=False
    ):
        super().__init__()

        self.num_layers = num_layers
        self.gradient_checkpointing = gradient_checkpointing

        self.embedding = InputEmbedding(
            vocab_size,
            context_length,
            embedding_dim
        )

        self.embedding_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim,
                    num_heads,
                    context_length,
                    dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = RMSNorm(
    embedding_dim
)

        # GPT-2-style scaled residual init: every sub-layer output
        # that gets added straight into the residual stream
        # (attention's output_projection, feed-forward's down
        # projection) is re-initialized with its standard deviation
        # divided by sqrt(2 * num_layers). Without this, the variance
        # of the residual stream grows with depth (each of the
        # 2 * num_layers additions injects a full-scale contribution),
        # which makes deeper stacks noticeably harder to train stably.
        # This only rescales those two projections; everything else
        # keeps PyTorch's default Linear init.
        residual_std = 0.02 / (2 * num_layers) ** 0.5

        for block in self.blocks:
            nn.init.normal_(
                block.attention.output_projection.weight,
                mean=0.0,
                std=residual_std
            )
            nn.init.zeros_(block.attention.output_projection.bias)

            nn.init.normal_(
                block.feed_forward.down.weight,
                mean=0.0,
                std=residual_std
            )
            nn.init.zeros_(block.feed_forward.down.bias)

    def forward(self, tokens, kv_cache=None):

        x = self.embedding(tokens)

        x = self.embedding_dropout(x)

        use_checkpointing = (
            self.gradient_checkpointing
            and self.training
            and kv_cache is None
        )

        for layer_idx, block in enumerate(self.blocks):

            if use_checkpointing:
                # Trades compute for memory: activations inside each
                # block are recomputed on the backward pass instead of
                # being kept around, which lets deeper/wider models
                # (or longer sequences) fit in memory at the cost of
                # roughly one extra forward pass per block. Not used
                # for cached generation (kv_cache is not None) or eval,
                # since there's nothing to save memory on there.
                x = torch.utils.checkpoint.checkpoint(
                    lambda inp, b=block: b(inp),
                    x,
                    use_reentrant=False
                )
            else:
                x = block(x, kv_cache=kv_cache, layer_idx=layer_idx)

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
