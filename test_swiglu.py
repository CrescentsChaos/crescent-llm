import torch

from model.transformer import FeedForward


ffn = FeedForward(
    embedding_dim=128
)

x = torch.randn(
    2,
    8,
    128
)

output = ffn(x)

print("SwiGLU Test")
print("===========")

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Input device:", x.device)
print("Output device:", output.device)