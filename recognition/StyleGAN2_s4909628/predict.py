from modules import DCGANGenerator, StyleGAN2Generator, IMAGE_SIZE, IMAGE_CHANNELS

import torch
from torchvision.utils import save_image

import argparse
from pathlib import Path


def load_generator(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    model_name = checkpoint.get("model_name")
    state_key = "generator_state_dict"

    if model_name == "stylegan2":
        config = checkpoint["config"]

        if config.get("model_name") != model_name:
            raise ValueError("Checkpoint and config disagree on model_name.")

        generator = StyleGAN2Generator(**config["generator_kwargs"])

        if config.get("ema_enabled", False):
            state_key = "generator_ema_state_dict"

    elif model_name == "dcgan":
        if (
            checkpoint["image_size"] != IMAGE_SIZE
            or checkpoint["image_channels"] != IMAGE_CHANNELS
        ):
            raise ValueError("DCGAN checkpoint image dimensions do not match.")

        generator = DCGANGenerator(latent_dim=checkpoint["latent_dim"])

    else:
        raise ValueError(f"Unsupported checkpoint model_name={model_name!r}.")

    state = checkpoint.get(state_key)

    if state is None:
        raise ValueError(f"Checkpoint is missing {state_key}.")

    generator.load_state_dict(state, strict=True)
    generator = generator.to(device)
    generator.eval()

    return checkpoint, generator


# Compatibility entry point for earlier baseline commands.
def load_dcgan_generator(checkpoint_path, device):
    checkpoint, generator = load_generator(checkpoint_path, device)
    if checkpoint["model_name"] != "dcgan":
        raise ValueError("Expected a DCGAN checkpoint.")
    return checkpoint, generator


@torch.no_grad()
def save_generated_grid(
    generator, fixed_noise, output_path, nrow=8, batch_size=4, noise_mode="const"
):
    if batch_size < 1 or nrow < 1:
        raise ValueError("batch_size and nrow must be positive.")
    if fixed_noise.ndim != 2 or fixed_noise.shape[0] == 0:
        raise ValueError("Expected nonempty fixed_noise [B, z_dim].")
    was_training = generator.training
    generator.eval()
    try:
        image_batches = []
        for noise_batch in fixed_noise.split(batch_size):
            if isinstance(generator, StyleGAN2Generator):
                images = generator(
                    noise_batch, noise_mode=noise_mode, update_w_avg=False
                )
            else:
                images = generator(noise_batch)
            if not torch.isfinite(images).all().item():
                raise FloatingPointError("Non-finite generated images.")
            image_batches.append(images.cpu())

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(
            torch.cat(image_batches, dim=0),
            output_path,
            nrow=nrow,
            normalize=True,
            value_range=(-1, 1),
        )
    finally:
        generator.train(was_training)


def main():
    parser = argparse.ArgumentParser(
        description="Generate previews from StyleGAN2 or saved DCGAN checkpoints."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-images", type=int, default=16)
    parser.add_argument("--nrow", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--noise-mode", choices=("const", "random", "none"), default="const"
    )
    parser.add_argument(
        "--use-fixed-noise",
        action="store_true",
        help="Use the latent vectors saved in the checkpoint.",
    )
    args = parser.parse_args()
    if min(args.num_images, args.nrow, args.batch_size) < 1:
        parser.error("num-images, nrow and batch-size must be positive.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint, generator = load_generator(args.checkpoint, device)
    z_dim = (
        generator.z_dim
        if isinstance(generator, StyleGAN2Generator)
        else generator.latent_dim
    )
    if args.use_fixed_noise:
        fixed_noise = checkpoint.get("fixed_noise")
        if not isinstance(fixed_noise, torch.Tensor):
            raise ValueError("Checkpoint does not contain fixed_noise.")
        if fixed_noise.ndim != 2 or fixed_noise.shape[1] != z_dim:
            raise ValueError("Checkpoint fixed_noise does not match the generator.")
        if args.num_images > len(fixed_noise):
            raise ValueError("Requested more images than saved fixed-noise vectors.")
        noise = fixed_noise[: args.num_images].to(device)
    else:
        rng = torch.Generator(device=device).manual_seed(args.seed)
        noise = torch.randn(args.num_images, z_dim, generator=rng, device=device)

    # Also seed StyleConv's internal noise when --noise-mode=random is requested.
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    save_generated_grid(
        generator,
        noise,
        args.output,
        nrow=args.nrow,
        batch_size=args.batch_size,
        noise_mode=args.noise_mode,
    )
    print(f"Saved {args.num_images} generated images as a grid: {args.output}")


if __name__ == "__main__":
    main()
