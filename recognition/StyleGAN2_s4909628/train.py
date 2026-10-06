import modules
import dataset
from modules import IMAGE_SIZE
from modules import IMAGE_CHANNELS
import predict

from pathlib import Path
import json
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from torchvision.utils import save_image


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# L_D​ = max(0, 1 − logits_real) + max(0, 1 + logits_fake​)
# Discriminator wants logits_real >= 1 and logits_fake <= -1
def discriminator_hinge_loss(real_logits, fake_logits):
    real_loss = F.relu(1.0 - real_logits).mean()
    fake_loss = F.relu(1.0 + fake_logits).mean()
    total_loss = real_loss + fake_loss
    return total_loss, real_loss, fake_loss


# L_G = -logits_fake
# Generator wants fake logits to become as large as possible.
def generator_loss(fake_logits):
    return -fake_logits.mean()


def discriminator_bce_loss(real_logits, fake_logits):
    real_loss = F.binary_cross_entropy_with_logits(
        real_logits, torch.ones_like(real_logits)
    )

    fake_loss = F.binary_cross_entropy_with_logits(
        fake_logits, torch.zeros_like(fake_logits)
    )

    total_loss = real_loss + fake_loss
    return total_loss, real_loss, fake_loss


def generator_bce_loss(fake_logits):
    return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))


def train_one_epoch_dcgan(
    generator, discriminator, train_loader, optimizer_g, optimizer_d, device, latent_dim
):
    generator.train()
    discriminator.train()
    total_g_loss = 0.0
    total_d_loss = 0.0
    total_real_loss = 0.0
    total_fake_loss = 0.0
    total_real_score = 0.0
    total_fake_score = 0.0
    sample_count = 0
    for real_images in train_loader:
        real_images = real_images.to(device, non_blocking=device.type == "cuda")
        current_batch_size = real_images.size(0)

        # Train the Discriminator
        for parameter in discriminator.parameters():
            parameter.requires_grad_(True)

        optimizer_d.zero_grad(set_to_none=True)

        z = torch.randn(current_batch_size, latent_dim, device=device)
        # Only used to train Discriminator, excluded from gradient updates
        with torch.no_grad():
            fake_images = generator(z)

        real_logits = discriminator(real_images)
        fake_logits_for_d = discriminator(fake_images)

        d_loss, d_real_loss, d_fake_loss = discriminator_bce_loss(
            real_logits,
            fake_logits_for_d,
        )

        if not torch.isfinite(d_loss).item():
            raise FloatingPointError("Non-finite discriminator loss detected")

        d_loss.backward()
        optimizer_d.step()
        # release the gradient
        optimizer_d.zero_grad(set_to_none=True)

        # Train the Generator
        for parameter in discriminator.parameters():
            parameter.requires_grad_(False)
        optimizer_g.zero_grad(set_to_none=True)

        z = torch.randn(current_batch_size, latent_dim, device=device)
        fake_images = generator(z)
        fake_logits_for_g = discriminator(fake_images)
        g_loss = generator_bce_loss(fake_logits_for_g)
        if not torch.isfinite(g_loss).item():
            raise FloatingPointError("Non-finite generator loss detected")

        g_loss.backward()
        optimizer_g.step()

        for parameter in discriminator.parameters():
            parameter.requires_grad_(True)

        total_g_loss += g_loss.item() * current_batch_size
        total_d_loss += d_loss.item() * current_batch_size
        total_real_loss += d_real_loss.item() * current_batch_size
        total_fake_loss += d_fake_loss.item() * current_batch_size
        total_real_score += real_logits.detach().mean().item() * current_batch_size
        total_fake_score += (
            fake_logits_for_d.detach().mean().item() * current_batch_size
        )
        sample_count += current_batch_size

    if sample_count == 0:
        raise RuntimeError("Training loader produced no images.")

    average_real_score = total_real_score / sample_count
    average_fake_score = total_fake_score / sample_count
    return {
        "generator_loss": total_g_loss / sample_count,
        "discriminator_loss": total_d_loss / sample_count,
        "real_loss": total_real_loss / sample_count,
        "fake_loss": total_fake_loss / sample_count,
        "real_score": average_real_score,
        "fake_score": average_fake_score,
        "score_gap": average_real_score - average_fake_score,
    }


@torch.no_grad()
def evaluate_dcgan(
    generator, discriminator, data_loader, device, latent_dim, random_seed=12345
):
    generator.eval()
    discriminator.eval()
    total_g_loss = 0.0
    total_d_loss = 0.0
    total_real_loss = 0.0
    total_fake_loss = 0.0
    total_real_score = 0.0
    total_fake_score = 0.0
    sample_count = 0

    random_generator = torch.Generator(device=device).manual_seed(random_seed)
    for real_images in data_loader:
        real_images = real_images.to(device, non_blocking=device.type == "cuda")
        current_batch_size = real_images.size(0)
        z = torch.randn(
            current_batch_size, latent_dim, device=device, generator=random_generator
        )
        fake_images = generator(z)
        real_logits = discriminator(real_images)
        fake_logits = discriminator(fake_images)

        d_loss, d_real_loss, d_fake_loss = discriminator_bce_loss(
            real_logits,
            fake_logits,
        )
        if not torch.isfinite(d_loss).item():
            raise FloatingPointError("Non-finite discriminator loss detected")
        g_loss = generator_bce_loss(fake_logits)
        if not torch.isfinite(g_loss).item():
            raise FloatingPointError("Non-finite generator loss detected")

        total_g_loss += g_loss.item() * current_batch_size
        total_d_loss += d_loss.item() * current_batch_size
        total_real_loss += d_real_loss.item() * current_batch_size
        total_fake_loss += d_fake_loss.item() * current_batch_size
        total_real_score += real_logits.detach().mean().item() * current_batch_size
        total_fake_score += fake_logits.detach().mean().item() * current_batch_size
        sample_count += current_batch_size

    if sample_count == 0:
        raise RuntimeError("Evaluation loader produced no images.")

    average_real_score = total_real_score / sample_count
    average_fake_score = total_fake_score / sample_count
    return {
        "generator_loss": total_g_loss / sample_count,
        "discriminator_loss": total_d_loss / sample_count,
        "real_loss": total_real_loss / sample_count,
        "fake_loss": total_fake_loss / sample_count,
        "real_score": average_real_score,
        "fake_score": average_fake_score,
        "score_gap": average_real_score - average_fake_score,
    }


# Save a reference grid of normalized real ADNI images.
def save_real_grid(data_loader, output_path, num_images=64, nrow=8):
    image_batches = []
    collected = 0
    for images in data_loader:
        image_batches.append(images)
        collected += images.size(0)
        if collected >= num_images:
            break

    if not image_batches:
        raise RuntimeError("Cannot save a real-image grid from an empty loader.")

    images = torch.cat(image_batches, dim=0)[:num_images]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(
        images,
        output_path,
        nrow=nrow,
        normalize=True,
        value_range=(-1, 1),
    )


# Plot adversarial losses and discriminator scores over completed epochs.
def plot_training_history(history, output_path):
    if not history["train"]:
        return

    epoch_numbers = range(1, len(history["train"]) + 1)
    train = history["train"]
    valid = history["valid"]

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].plot(
        epoch_numbers,
        [metrics["generator_loss"] for metrics in train],
        label="Train",
    )
    axes[0, 0].plot(
        epoch_numbers,
        [metrics["generator_loss"] for metrics in valid],
        label="Validation",
    )
    axes[0, 0].set_title("Generator adversarial loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(
        epoch_numbers,
        [metrics["discriminator_loss"] for metrics in train],
        label="Train",
    )
    axes[0, 1].plot(
        epoch_numbers,
        [metrics["discriminator_loss"] for metrics in valid],
        label="Validation",
    )
    axes[0, 1].set_title("Discriminator adversarial loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(
        epoch_numbers,
        [metrics["real_score"] for metrics in train],
        label="Train real",
    )
    axes[1, 0].plot(
        epoch_numbers,
        [metrics["fake_score"] for metrics in train],
        label="Train fake",
    )
    axes[1, 0].plot(
        epoch_numbers,
        [metrics["real_score"] for metrics in valid],
        linestyle="--",
        label="Validation real",
    )
    axes[1, 0].plot(
        epoch_numbers,
        [metrics["fake_score"] for metrics in valid],
        linestyle="--",
        label="Validation fake",
    )
    axes[1, 0].set_title("Discriminator logits")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(
        epoch_numbers,
        [metrics["score_gap"] for metrics in train],
        label="Train",
    )
    axes[1, 1].plot(
        epoch_numbers,
        [metrics["score_gap"] for metrics in valid],
        label="Validation",
    )
    axes[1, 1].set_title("Real score - fake score")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.25)

    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


# Restore models, optimizers, history, epoch, and fixed monitoring noise.
def load_training_checkpoint(
    checkpoint_path,
    generator,
    discriminator,
    optimizer_g,
    optimizer_d,
    train_loader,
    device,
    expected_latent_dim,
    expected_model_name,
):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_model_name = checkpoint.get(
        "model_name",
        checkpoint.get("config", {}).get("model_name"),
    )

    if checkpoint_model_name != expected_model_name:
        raise ValueError(
            f"Checkpoint model_name={checkpoint_model_name!r}, "
            f"but the current model is {expected_model_name!r}."
        )
    checkpoint_latent_dim = checkpoint.get("latent_dim", expected_latent_dim)
    if checkpoint_latent_dim != expected_latent_dim:
        raise ValueError(
            f"Checkpoint latent_dim={checkpoint_latent_dim}, but the current "
            f"configuration uses latent_dim={expected_latent_dim}."
        )

    generator.load_state_dict(checkpoint["generator_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
    optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])

    history = checkpoint.get("history", {"train": [], "valid": []})
    completed_epoch = int(checkpoint.get("epoch", 0))
    fixed_noise = checkpoint.get("fixed_noise")
    if fixed_noise is not None:
        fixed_noise = fixed_noise.to(device)

    torch_rng_state = checkpoint.get("torch_rng_state")
    if torch_rng_state is not None:
        torch.set_rng_state(torch_rng_state.cpu())
    cuda_rng_state_all = checkpoint.get("cuda_rng_state_all")
    if device.type == "cuda" and cuda_rng_state_all is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng_state_all])

    loader_rng_state = checkpoint.get("loader_rng_state")
    if loader_rng_state is not None:
        if train_loader.generator is None:
            raise ValueError(
                "Checkpoint contains loader_rng_state, but "
                "train_loader has no generator."
            )

        train_loader.generator.set_state(loader_rng_state.cpu())

    return completed_epoch, history, fixed_noise


def save_checkpoint(
    output_path,
    completed_epoch,
    generator,
    discriminator,
    optimizer_g,
    optimizer_d,
    history,
    fixed_noise,
    config,
    train_loader,
    test_metrics=None,
):
    checkpoint = {
        "model_name": config["model_name"],
        "epoch": completed_epoch,
        "latent_dim": config["latent_dim"],
        "image_size": config["image_size"],
        "image_channels": config["image_channels"],
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_g_state_dict": optimizer_g.state_dict(),
        "optimizer_d_state_dict": optimizer_d.state_dict(),
        "history": history,
        "fixed_noise": fixed_noise.detach().cpu(),
        "config": config,
        "test_metrics": test_metrics,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "loader_rng_state": (
            train_loader.generator.get_state().clone()
            if train_loader.generator is not None
            else None
        ),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(output_path)


def main():
    data_root = "/home/groups/comp3710/ADNI/AD_NC"
    metadata_path = "/home/groups/comp3710/ADNI/meta_data_with_label.json"
    manifest_path = (
        Path(__file__).resolve().parent / "splits" / "splits_ad_nc_seed42.json"
    )

    model_name = "dcgan"
    run_name = "dcgan_adni"

    home_dir = Path(__file__).resolve().parent
    # Set this to latest_checkpoint_path to continue an interrupted run.
    resume_checkpoint = None

    if resume_checkpoint is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = home_dir / "runs" / f"{run_name}_{run_id}"
    else:
        resume_checkpoint = Path(resume_checkpoint).resolve()

        if not resume_checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {resume_checkpoint}")

        # run_dir/checkpoints/latest.pt
        run_dir = resume_checkpoint.parent.parent

    checkpoint_dir = run_dir / "checkpoints"
    sample_dir = run_dir / "samples"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Training configuration for the DCGAN.
    epochs = 100
    batch_size = 16
    learning_rate_g = 2e-4
    learning_rate_d = 1e-4
    latent_dim = 128
    random_seed = 42
    validation_seed = 12345
    num_sample_images = 64
    sample_every = 5
    checkpoint_every = 10
    adam_betas = (0.5, 0.999)

    # data loader configration
    split_seed = 42
    loader_seed = 42
    val_ratio = 0.2
    included_classes = ("AD", "NC")

    latest_checkpoint_path = checkpoint_dir / "latest.pt"
    final_checkpoint_path = checkpoint_dir / "final.pt"
    history_path = run_dir / "history.pt"
    test_metrics_path = run_dir / "test_metrics.json"
    history_plot_path = run_dir / "training_history.png"
    config_path = run_dir / "config.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True
    num_workers = 4 if device.type == "cuda" else 0

    torch.manual_seed(random_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(random_seed)

    (
        train_dataset,
        train_loader,
        test_dataset,
        test_loader,
        valid_dataset,
        valid_loader,
    ) = dataset.build_dataloaders(
        data_root=data_root,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        included_classes=included_classes,
        val_ratio=val_ratio,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        split_seed=split_seed,
        loader_seed=loader_seed,
    )

    print("Training images:", len(train_dataset))
    print("Validation images:", len(valid_dataset))
    print("Test images:", len(test_dataset))

    generator = modules.DCGANGenerator(latent_dim=latent_dim).to(device)
    discriminator = modules.DCGANDiscriminator().to(device)

    optimizer_g = torch.optim.Adam(
        generator.parameters(), lr=learning_rate_g, betas=adam_betas
    )

    optimizer_d = torch.optim.Adam(
        discriminator.parameters(), lr=learning_rate_d, betas=adam_betas
    )

    config = {
        "model_name": model_name,
        "run_name": run_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate_g": learning_rate_g,
        "learning_rate_d": learning_rate_d,
        "adam_betas": list(adam_betas),
        "latent_dim": latent_dim,
        "random_seed": random_seed,
        "validation_seed": validation_seed,
        "loss": "bce_with_logits",
        "spectral_normalization": False,
        "minibatch_stddev": False,
        "manifest_path": str(manifest_path.resolve()),
        "included_classes": included_classes,
        "split_seed": split_seed,
        "loader_seed": loader_seed,
        "val_ratio": val_ratio,
        "image_size": IMAGE_SIZE,
        "image_channels": IMAGE_CHANNELS,
    }

    history = {"train": [], "valid": []}
    start_epoch = 0
    fixed_noise = torch.randn(num_sample_images, latent_dim, device=device)

    if resume_checkpoint is not None:
        resume_checkpoint = Path(resume_checkpoint)
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {resume_checkpoint}")
        start_epoch, history, saved_fixed_noise = load_training_checkpoint(
            checkpoint_path=resume_checkpoint,
            generator=generator,
            discriminator=discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            train_loader=train_loader,
            device=device,
            expected_latent_dim=latent_dim,
            expected_model_name=model_name,
        )
        if saved_fixed_noise is not None:
            fixed_noise = saved_fixed_noise
        print(f"Resumed from {resume_checkpoint} after epoch {start_epoch}")

    if resume_checkpoint is None:
        save_real_grid(
            valid_loader,
            sample_dir / "real_validation.png",
            num_images=num_sample_images,
        )
        with config_path.open("w", encoding="utf-8") as file:
            json.dump(config, file, ensure_ascii=False, indent=2)

    for epoch_index in range(start_epoch, epochs):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch_dcgan(
            generator=generator,
            discriminator=discriminator,
            train_loader=train_loader,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            device=device,
            latent_dim=latent_dim,
        )
        valid_metrics = evaluate_dcgan(
            generator=generator,
            discriminator=discriminator,
            data_loader=valid_loader,
            device=device,
            latent_dim=latent_dim,
            random_seed=validation_seed,
        )

        history["train"].append(train_metrics)
        history["valid"].append(valid_metrics)
        completed_epoch = epoch_index + 1
        elapsed_seconds = time.perf_counter() - epoch_start

        print(
            f"Epoch [{completed_epoch:03d}/{epochs:03d}] "
            f"time={elapsed_seconds:.1f}s | "
            f"train G={train_metrics['generator_loss']:.4f} "
            f"D={train_metrics['discriminator_loss']:.4f} "
            f"real={train_metrics['real_score']:.4f} "
            f"fake={train_metrics['fake_score']:.4f} "
            f"gap={train_metrics['score_gap']:.4f} | "
            f"valid G={valid_metrics['generator_loss']:.4f} "
            f"D={valid_metrics['discriminator_loss']:.4f} "
            f"real={valid_metrics['real_score']:.4f} "
            f"fake={valid_metrics['fake_score']:.4f} "
            f"gap={valid_metrics['score_gap']:.4f}",
            flush=True,
        )

        if (
            completed_epoch == 1
            or completed_epoch % sample_every == 0
            or completed_epoch == epochs
        ):
            predict.save_generated_grid(
                generator,
                fixed_noise,
                sample_dir / f"epoch_{completed_epoch:03d}.png",
            )

        torch.save(history, history_path)
        plot_training_history(history, history_plot_path)
        save_checkpoint(
            output_path=latest_checkpoint_path,
            completed_epoch=completed_epoch,
            generator=generator,
            discriminator=discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            history=history,
            fixed_noise=fixed_noise,
            config=config,
            train_loader=train_loader,
        )

        if completed_epoch % checkpoint_every == 0:
            save_checkpoint(
                output_path=(checkpoint_dir / f"epoch_{completed_epoch:03d}.pt"),
                completed_epoch=completed_epoch,
                generator=generator,
                discriminator=discriminator,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
                history=history,
                fixed_noise=fixed_noise,
                config=config,
                train_loader=train_loader,
            )

    test_metrics = evaluate_dcgan(
        generator=generator,
        discriminator=discriminator,
        data_loader=test_loader,
        device=device,
        latent_dim=latent_dim,
        random_seed=validation_seed + 1,
    )
    print(
        "Test | "
        f"G={test_metrics['generator_loss']:.4f} "
        f"D={test_metrics['discriminator_loss']:.4f} "
        f"real_loss={test_metrics['real_loss']:.4f} "
        f"fake_loss={test_metrics['fake_loss']:.4f} "
        f"real={test_metrics['real_score']:.4f} "
        f"fake={test_metrics['fake_score']:.4f} "
        f"gap={test_metrics['score_gap']:.4f}",
        flush=True,
    )

    save_checkpoint(
        output_path=final_checkpoint_path,
        completed_epoch=epochs,
        generator=generator,
        discriminator=discriminator,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        history=history,
        fixed_noise=fixed_noise,
        config=config,
        train_loader=train_loader,
        test_metrics=test_metrics,
    )
    with test_metrics_path.open("w", encoding="utf-8") as output_file:
        json.dump(test_metrics, output_file, indent=2)

    print("Saved latest checkpoint to", latest_checkpoint_path)
    print("Saved final checkpoint to", final_checkpoint_path)
    print("Saved training history plot to", history_plot_path)
    print("Saved test metrics to", test_metrics_path)


if __name__ == "__main__":
    # my GAN baseline
    main()
