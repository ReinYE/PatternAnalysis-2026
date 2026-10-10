import torch
from torch.nn import functional as F

from modules import StyleGAN2Discriminator


torch.manual_seed(2026)

discriminator = StyleGAN2Discriminator(
    image_resolution=32,
    image_channels=1,
    channel_base=256,
    channel_max=32,
    mbstd_group_size=4,
)

real = torch.randn(
    4, 1, 32, 32,
    requires_grad=True,
)

logits = discriminator(real)

assert logits.shape == (4, 1)
assert torch.isfinite(logits).all()

loss = F.softplus(-logits).mean()
loss.backward()

assert real.grad is not None
assert torch.isfinite(real.grad).all()

for name, parameter in discriminator.named_parameters():
    assert parameter.grad is not None, (
        f"Missing gradient: {name}"
    )
    assert torch.isfinite(parameter.grad).all(), (
        f"Non-finite gradient: {name}"
    )

print("Discriminator logits:", tuple(logits.shape))
print("Discriminator backward passed.")

discriminator.zero_grad(set_to_none=True)

real = torch.randn(
    4, 1, 32, 32,
    requires_grad=True,
)

logits = discriminator(real)

image_grad = torch.autograd.grad(
    outputs=logits.sum(),
    inputs=real,
    create_graph=True,
)[0]

r1_penalty = (
    image_grad.square()
    .flatten(1)
    .sum(dim=1)
    .mean()
)

assert torch.isfinite(r1_penalty)

r1_penalty.backward()

# R1 must produce gradients for the discriminator's feature extraction.
assert discriminator.from_image.weight.grad is not None
assert torch.isfinite(
    discriminator.from_image.weight.grad
).all()

for name, parameter in discriminator.named_parameters():
    if parameter.grad is not None:
        assert torch.isfinite(parameter.grad).all(), name

print("R1 backward passed.")

discriminator = StyleGAN2Discriminator(
    image_resolution=256,
    image_channels=1,
    channel_base=16384,
    channel_max=512,
)

discriminator.eval()

with torch.no_grad():
    logits = discriminator(
        torch.randn(1, 1, 256, 256)
    )

assert logits.shape == (1, 1)
assert torch.isfinite(logits).all()

print("Full-resolution logits:", tuple(logits.shape))