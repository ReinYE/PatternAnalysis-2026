from modules import ConvGANGenerator, IMAGE_SIZE, IMAGE_CHANNELS

import torch
from torchvision.utils import save_image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse
from pathlib import Path


def load_generator(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint["image_size"] != IMAGE_SIZE
        or checkpoint["image_channels"] != IMAGE_CHANNELS
    ):
        raise ValueError("Checkpoint image dimensions do not match this model.")

    generator = ConvGANGenerator(latent_dim=checkpoint["latent_dim"])
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator = generator.to(device)
    generator.eval()

    return checkpoint, generator


# Save samples from fixed noise so progress is comparable across epochs.
@torch.no_grad()
def save_generated_grid(generator, fixed_noise, output_path, nrow=8):
    was_training = generator.training
    generator.eval()
    generated_images = generator(fixed_noise)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(
        generated_images.cpu(),
        output_path,
        nrow=nrow,
        normalize=True,
        value_range=(-1, 1),
    )

    if was_training:
        generator.train()


def main():
    parser = argparse.ArgumentParser(
        description=("Generate a preview grid with the ConvGAN baseline.")
    )

    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-images", type=int, default=16)
    parser.add_argument("--nrow", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-fixed-noise",
        action="store_true",
        help="Use the fixed noise saved during training.",
    )
    args = parser.parse_args()

    if args.num_images < 1 or args.nrow < 1:
        parser.error("--num-images and --nrow must be positive.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint, generator = load_generator(args.checkpoint, device)

    if args.use_fixed_noise:
        fixed_noise = checkpoint.get("fixed_noise")

        if not isinstance(fixed_noise, torch.Tensor):
            raise ValueError("Checkpoint does not contain fixed_noise.")

        if args.num_images > len(fixed_noise):
            raise ValueError(
                f"Requested {args.num_images} images, but checkpoint "
                f"contains only {len(fixed_noise)} fixed noise vectors."
            )

        noise = fixed_noise[: args.num_images].to(device)

    else:
        noise_generator = torch.Generator(device=device)
        noise_generator.manual_seed(args.seed)

        noise = torch.randn(
            args.num_images,
            generator.latent_dim,
            generator=noise_generator,
            device=device,
        )

    save_generated_grid(generator, noise, args.output, nrow=args.nrow)

    print(f"Saved {args.num_images} generated images " f"as a grid: {args.output}")


if __name__ == "__main__":
    main()
