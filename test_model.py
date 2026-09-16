import torch
import torch.nn.functional as F
from config import ModelConfig
from model.transformer import (
    InputEmbedding,
    SelfAttention,
    FeedForward,
    TransformerBlock,
    Transformer,
    LanguageModelHead
)

transformer_block = TransformerBlock(
    embedding_dim=ModelConfig.embedding_dim,
    num_heads=ModelConfig.num_heads,
    context_length=ModelConfig.context_length
)

transformer = Transformer(
    vocab_size=ModelConfig.vocab_size,
    context_length=ModelConfig.context_length,
    embedding_dim=ModelConfig.embedding_dim,
    num_heads=ModelConfig.num_heads,
    num_layers=ModelConfig.num_layers
)

lm_head = LanguageModelHead(
    embedding_dim=ModelConfig.embedding_dim,
    vocab_size=ModelConfig.vocab_size
)

embedding = InputEmbedding(
    vocab_size=ModelConfig.vocab_size,
    context_length=ModelConfig.context_length,
    embedding_dim=ModelConfig.embedding_dim
)

attention = SelfAttention(
    ModelConfig.embedding_dim,
    ModelConfig.num_heads,
    ModelConfig.context_length
)
feed_forward = FeedForward(
    ModelConfig.embedding_dim
)

tokens = torch.tensor([
    [6, 14, 11, 1, 9, 7],
    [9, 7, 25, 1, 6, 14]
])
y = torch.tensor([
    [14, 11, 1, 9, 7, 6],
    [7, 25, 1, 6, 14, 9]
])

x = embedding(tokens)

ffn_test = feed_forward(x)

# print("\nFeedForward Test:")
# print(ffn_test.shape)

# block_output = transformer_block(x)

# print("\nTransformer Block Output:")
# print(block_output.shape)

transformer_output = transformer(tokens)

# print("\nFull Transformer Output:")
# print(transformer_output.shape)

logits = lm_head(transformer_output)

# print("\nLanguage Model Logits:")
# print(logits.shape)

probabilities = F.softmax(
    logits,
    dim=-1
)

# print("\nProbabilities:")
# print(probabilities.shape)

# print("\nProbabilities for first sequence, first position:")
# print(probabilities[0, 0])

# print("\nProbability sum:")
# print(probabilities[0, 0].sum())

loss = F.cross_entropy(
    logits.view(-1, ModelConfig.vocab_size),
    y.view(-1)
)

print("\nLoss:")
print(loss)

(Q, K, V, scores, attention_weights, attention_output) = attention(x)
residual_output = x + attention_output
normalized_output = attention.layer_norm(
    residual_output
)
ffn_hidden = attention.fc1(
    normalized_output
)
ffn_activated = attention.gelu(
    ffn_hidden
)
ffn_output = attention.fc2(
    ffn_activated
)

# print("Input:")
# print(x.shape)

# print("\nQuery:")
# print(Q.shape)

# print("\nKey:")
# print(K.shape)

# print("\nValue:")
# print(V.shape)

# print("\nAttention Scores:")
# print(scores.shape)

# print("\nMasked scores - Batch 0, Head 0:")
# print(scores[0, 0])

# print("\nAttention Weights - Batch 0, Head 0:")
# print(attention_weights[0, 0])

# print("\nAttention Output:")
# print(attention_output.shape)

# print("\nResidual Output:")
# print(residual_output.shape)

# print("\nNormalized Output:")
# print(normalized_output.shape)

# print("\nNormalized Mean:")
# print(normalized_output.mean())

# print("\nNormalized Standard Deviation:")
# print(normalized_output.std())

# print("\nFFN Hidden:")
# print(ffn_hidden.shape)

# print("\nFFN After GELU:")
# print(ffn_activated.shape)

# print("\nFFN Output:")
# print(ffn_output.shape)