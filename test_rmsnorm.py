import torch

from model.transformer import RMSNorm


norm = RMSNorm(
    embedding_dim=128
)

x = torch.randn(
    2,
    8,
    128
)

output = norm(x)

rms = torch.sqrt(
    torch.mean(
        output ** 2
    )
)

print("RMSNorm Test")
print("============")

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Output RMS:", rms.item())