import torch

from model.transformer import RotaryEmbedding


rope = RotaryEmbedding(
    head_dim=32,
    context_length=32
)

Q = torch.randn(
    2,
    4,
    8,
    32
)

K = torch.randn(
    2,
    4,
    8,
    32
)

Q_rotated, K_rotated = rope(
    Q,
    K
)

print("RoPE Test")
print("=========")

print("Q shape:", Q.shape)
print("K shape:", K.shape)

print(
    "Rotated Q shape:",
    Q_rotated.shape
)

print(
    "Rotated K shape:",
    K_rotated.shape
)

print(
    "Q changed:",
    not torch.equal(Q, Q_rotated)
)

print(
    "K changed:",
    not torch.equal(K, K_rotated)
)