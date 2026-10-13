import modules
import dataset
from modules import IMAGE_SIZE
from modules import IMAGE_CHANNELS
import predict

from pathlib import Path
import argparse
import copy
import hashlib
import json
import platform
import time
from datetime import datetime, timezone

import torch
import torchvision
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


# StyleGAN2 loss
def discriminator_logistic_loss(real_logits, fake_logits):
    real_loss = F.softplus(-real_logits).mean()
    fake_loss = F.softplus(fake_logits).mean()

    total_loss = real_loss + fake_loss
    return total_loss, real_loss, fake_loss


def generator_logistic_loss(fake_logits):
    return F.softplus(-fake_logits).mean()


def discriminator_r1_penalty(discriminator, real_images):
    # Create an input leaf that requires gradients.
    r1_images = real_images.detach().requires_grad_(True)

    # Fresh discriminator forward for this regularization step.
    real_logits = discriminator(r1_images)

    # Differentiate discriminator logits with respect to image pixels.
    image_gradients = torch.autograd.grad(
        outputs=real_logits.sum(), inputs=r1_images, create_graph=True
    )[0]

    # Sum over C, H, W, then average over the batch.
    per_image_penalty = image_gradients.square().flatten(start_dim=1).sum(dim=1)

    return per_image_penalty.mean()


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


# Configuration is constructed here and written once by main(), before training.
def build_config(args, device, saved_config=None):
    import math

    def is_finite_number(value):
        return type(value) in (int, float) and math.isfinite(value)

    if saved_config is not None:
        config = copy.deepcopy(saved_config)
        if (
            config.get("schema_version") != 3
            or config.get("model_name") != "stylegan2"
            or config.get("training_stage") != "7.5"
        ):
            raise ValueError(
                "Resume requires a stage-7.5 StyleGAN2 checkpoint with config "
                "schema_version=3. Start a new run when upgrading older stages; "
                "EMA and PL state must not be silently reinitialized."
            )
        for key in ("batch_size", "num_workers"):
            requested = getattr(args, key)
            if requested is not None and requested != config[key]:
                raise ValueError(f"Cannot change {key} when resuming this run.")
        if args.epochs is not None:
            config["epochs"] = args.epochs
    else:
        config = {
            "schema_version": 3,
            "model_name": "stylegan2",
            "training_stage": "7.5",
            "run_name": "stylegan2_adni_full_test",
            "epochs": args.epochs if args.epochs is not None else 2,
            "batch_size": args.batch_size if args.batch_size is not None else 64,
            "learning_rate_g": 0.002,
            "learning_rate_d": 0.002,
            "adam_betas": [0.0, 0.99],
            "adam_eps": 1e-8,
            "weight_decay": 0.0,
            "loss": "non_saturating_logistic",
            "generator_kwargs": {
                "z_dim": 512,
                "w_dim": 512,
                "image_resolution": IMAGE_SIZE,
                "image_channels": IMAGE_CHANNELS,
                "channel_base": 16384,
                "channel_max": 512,
                "mapping_layers": 8,
                "mapping_lr_multiplier": 0.01,
                "w_avg_beta": 0.995,
            },
            "discriminator_kwargs": {
                "image_resolution": IMAGE_SIZE,
                "image_channels": IMAGE_CHANNELS,
                "channel_base": 16384,
                "channel_max": 512,
                "mbstd_group_size": 4,
            },
            "architecture": {
                "generator": "skip",
                "discriminator": "residual",
                "conditional": False,
                "resample_filter_taps": [1, 3, 3, 1],
                "leaky_relu_slope": 0.2,
                "activation_gain": "sqrt(2 / (1 + 0.2**2))",
                "demodulation_eps": 1e-8,
                "mbstd_eps": 1e-8,
                "mbstd_added_channels": 1,
                "batch_normalization": False,
                "spectral_normalization": False,
                "generator_output_activation": "linear",
                "discriminator_output_activation": "linear",
            },
            "data_root": "/home/groups/comp3710/ADNI/AD_NC",
            "metadata_path": "/home/groups/comp3710/ADNI/meta_data_with_label.json",
            "manifest_path": str(
                Path(__file__).resolve().parent / "splits" / "splits_ad_nc_seed42.json"
            ),
            "included_classes": ["AD", "NC"],
            "val_ratio": 0.2,
            "split_seed": 42,
            "loader_seed": 42,
            "num_workers": (
                args.num_workers
                if args.num_workers is not None
                else (4 if device.type == "cuda" else 0)
            ),
            # Recreate workers each epoch, so worker-seed consumption also
            # matches after restoring the loader's generator at an epoch boundary.
            "persistent_workers": False,
            "pin_memory_policy": "cuda_only",
            "drop_last_train": True,
            "preprocessing": "grayscale_square_zero_padding_no_resize",
            "real_image_range": [-1.0, 1.0],
            "random_seed": 42,
            "validation_seed": 12345,
            "fixed_noise_seed": 2026,
            "num_sample_images": 16,
            "sample_batch_size": 4,
            "sample_nrow": 4,
            "sample_every": 1,
            # latest.pt is still overwritten every epoch; final.pt is saved at
            # completion. At 100 epochs this adds only two archived snapshots.
            # This interval does not prune old files or enforce a disk quota.
            "checkpoint_every": 25,
            "train_noise_mode": "random",
            "evaluation_noise_mode": "const",
            "sample_noise_mode": "const",
            "w_avg_update_policy": "g_main_only",
            "precision": "float32",
            "amp_enabled": False,
            "allow_tf32": True,
            # Avoid convolution algorithm benchmarking during the initial
            # full-training smoke run, including the new R1/PL backward paths.
            "cudnn_benchmark": True,
            "cudnn_deterministic": False,
            "gradient_accumulation_steps": 1,
            # Count adversarial updates separately from the extra PL/R1 steps.
            "g_main_updates_per_batch": 1,
            "d_main_updates_per_batch": 1,
            "lazy_regularization": True,
            "r1_gamma": 10.0,
            "r1_interval": 16,
            "r1_update_mode": "separate_optimizer_step",
            "r1_schedule": "zero_based_global_step",
            "pl_weight": 2.0,
            "pl_interval": 4,
            "pl_batch_shrink": 2,
            "pl_decay": 0.01,
            "pl_eps": 1e-8,
            "pl_update_mode": "separate_optimizer_step",
            "pl_schedule": "zero_based_global_step",
            "style_mixing_prob": 0.9,
            "style_mixing_phases": ["d_main", "g_main", "g_pl"],
            "ema_enabled": True,
            "ema_kimg": 10.0,
            "ema_rampup": 0.05,
            "ema_update_policy": "once_after_all_g_updates",
            # These describe the existing evaluation/sampling call sites.
            "evaluation_model": "generator",
            "ada_enabled": False,
            "augmentation": "none",
            "truncation_psi": 1.0,
        }
    positive_counts = (
        "epochs",
        "batch_size",
        "num_sample_images",
        "sample_batch_size",
        "sample_nrow",
        "sample_every",
        "checkpoint_every",
    )
    if any(type(config[key]) is not int or config[key] < 1 for key in positive_counts):
        raise ValueError(
            "Epochs, batch sizes, sample counts and intervals must be positive integers."
        )
    if type(config["num_workers"]) is not int or config["num_workers"] < 0:
        raise ValueError("num_workers must be a nonnegative integer.")
    if config["persistent_workers"]:
        raise ValueError("Stage 7.5 recreates workers for epoch-boundary resume.")
    if config["train_noise_mode"] not in ("random", "const", "none"):
        raise ValueError("Invalid training noise mode.")
    if config["evaluation_noise_mode"] not in ("const", "none"):
        raise ValueError("Evaluation noise must be const or none.")
    if config["sample_noise_mode"] not in ("const", "none"):
        raise ValueError("Monitoring previews require const or none noise.")
    if config.get("training_stage") != "7.5":
        raise ValueError("This entry point implements stage 7.5 only.")
    if config["loss"] != "non_saturating_logistic":
        raise ValueError("Stage 7.5 requires non-saturating logistic adversarial loss.")
    if (
        config["ada_enabled"]
        or config["augmentation"] != "none"
        or config["amp_enabled"]
        or config["truncation_psi"] != 1
    ):
        raise ValueError(
            "ADA, augmentation, AMP and truncation are not enabled in this stage."
        )

    for key in (
        "lazy_regularization",
        "ema_enabled",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "allow_tf32",
        "amp_enabled",
        "ada_enabled",
    ):
        if type(config[key]) is not bool:
            raise ValueError(f"{key} must be a boolean.")
    for key in ("r1_gamma", "pl_weight"):
        if not is_finite_number(config[key]) or config[key] < 0:
            raise ValueError(f"{key} must be a finite, nonnegative number.")
    for prefix in ("r1", "pl"):
        if (
            config[f"{prefix}_update_mode"] != "separate_optimizer_step"
            or config[f"{prefix}_schedule"] != "zero_based_global_step"
        ):
            raise ValueError(f"Unsupported {prefix} update mode or schedule.")

    if config["r1_gamma"] > 0:
        if type(config["r1_interval"]) is not int or config["r1_interval"] < 1:
            raise ValueError(
                "Positive r1_gamma requires a positive integer r1_interval."
            )
    elif config["r1_interval"] is not None:
        raise ValueError("Disabled R1 requires r1_interval=None.")

    if not is_finite_number(config["pl_eps"]) or config["pl_eps"] <= 0:
        raise ValueError("pl_eps must be finite and positive.")
    if config["pl_weight"] > 0:
        for key in ("pl_interval", "pl_batch_shrink"):
            if type(config[key]) is not int or config[key] < 1:
                raise ValueError(f"{key} must be a positive integer.")
        if not is_finite_number(config["pl_decay"]) or not 0 < config["pl_decay"] <= 1:
            raise ValueError("pl_decay must be finite and in (0, 1].")
    else:
        for key in ("pl_interval", "pl_batch_shrink", "pl_decay"):
            if config[key] is not None:
                raise ValueError(f"Disabled PL requires {key}=None.")
    if config["lazy_regularization"] != (
        config["r1_gamma"] > 0 or config["pl_weight"] > 0
    ):
        raise ValueError(
            "lazy_regularization must be enabled exactly when R1 or PL is active."
        )

    if (
        not is_finite_number(config["style_mixing_prob"])
        or not 0 <= config["style_mixing_prob"] <= 1
    ):
        raise ValueError("style_mixing_prob must be finite and in [0, 1].")
    if config["style_mixing_phases"] != ["d_main", "g_main", "g_pl"]:
        raise ValueError("Unsupported style mixing phases.")
    if config["ema_update_policy"] != "once_after_all_g_updates":
        raise ValueError("Unsupported EMA update policy.")
    if config["ema_enabled"]:
        if not is_finite_number(config["ema_kimg"]) or config["ema_kimg"] <= 0:
            raise ValueError("ema_kimg must be finite and positive.")
        rampup = config["ema_rampup"]
        if rampup is not None and (not is_finite_number(rampup) or rampup <= 0):
            raise ValueError("ema_rampup must be None or finite and positive.")
    elif config["ema_kimg"] is not None or config["ema_rampup"] is not None:
        raise ValueError("Disabled EMA requires ema_kimg=None and ema_rampup=None.")
    if config["evaluation_model"] != "generator":
        raise ValueError("Adversarial evaluation currently uses the original G.")

    if (
        config["precision"] != "float32"
        or config["gradient_accumulation_steps"] != 1
        or config["g_main_updates_per_batch"] != 1
        or config["d_main_updates_per_batch"] != 1
        or config["w_avg_update_policy"] != "g_main_only"
    ):
        raise ValueError("Unsupported stage-7.5 update policy.")
    for key in ("image_resolution", "image_channels"):
        if config["generator_kwargs"][key] != config["discriminator_kwargs"][key]:
            raise ValueError(f"G and D disagree on {key}.")

    for key in ("learning_rate_g", "learning_rate_d", "adam_eps", "weight_decay"):
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number.")
        if value < 0 or (key != "weight_decay" and value == 0):
            raise ValueError(f"Invalid optimizer setting: {key}.")
    betas = config["adam_betas"]
    if (
        not isinstance(betas, (list, tuple))
        or len(betas) != 2
        or any(
            type(beta) not in (int, float)
            or not math.isfinite(beta)
            or not 0 <= beta < 1
            for beta in betas
        )
    ):
        raise ValueError("adam_betas must contain two finite numbers in [0, 1).")

    # Resolve actual optimizer settings once from the unadjusted base values.
    # PL and R1 add independent optimizer steps, so adjust G and D separately.
    g_ratio = (
        config["pl_interval"] / (config["pl_interval"] + 1)
        if config["pl_weight"] > 0
        else 1.0
    )
    d_ratio = (
        config["r1_interval"] / (config["r1_interval"] + 1)
        if config["r1_gamma"] > 0
        else 1.0
    )
    resolved_optimizers = {
        "optimizer_g_kwargs": {
            "lr": config["learning_rate_g"] * g_ratio,
            "betas": [beta**g_ratio for beta in betas],
            "eps": config["adam_eps"],
            "weight_decay": config["weight_decay"],
        },
        "optimizer_d_kwargs": {
            "lr": config["learning_rate_d"] * d_ratio,
            "betas": [beta**d_ratio for beta in betas],
            "eps": config["adam_eps"],
            "weight_decay": config["weight_decay"],
        },
    }
    if saved_config is None:
        config.update(resolved_optimizers)
    else:
        # Validate saved values rather than silently changing resume settings
        # or applying the lazy-regularization adjustment a second time.
        for key, expected in resolved_optimizers.items():
            if config.get(key) != expected:
                raise ValueError(f"Saved {key} disagrees with the base/PL/R1 settings.")

    sample_model = "generator_ema" if config["ema_enabled"] else "generator"
    if saved_config is None:
        config["sample_model"] = sample_model
    elif config.get("sample_model") != sample_model:
        raise ValueError("Saved sample_model disagrees with ema_enabled.")
    return config


def write_json(output_path, value):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    temporary_path.replace(output_path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def measure_phase(device, operation):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = operation()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    return result, {
        "seconds": seconds,
        "peak_gpu_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "peak_gpu_reserved_bytes": (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        ),
    }


def record_phase(runtime, phase, measurement):
    runtime["phase_seconds"][phase] += measurement["seconds"]
    peaks = runtime["phase_peaks"][phase]
    for key in ("peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes"):
        value = measurement[key]
        if value is not None:
            peaks[key] = max(peaks[key] or 0, value)


def describe_environment(device):
    info = {
        "python": platform.python_version(),
        "pytorch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "gpu_name": None,
        "gpu_total_memory_bytes": None,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        info["gpu_name"] = properties.name
        info["gpu_total_memory_bytes"] = properties.total_memory
    return info


def new_runtime(generator, discriminator, data_signature):
    phases = ("train", "validation", "sampling", "test")
    return {
        "data_signature": data_signature,
        "generator_parameters": sum(p.numel() for p in generator.parameters()),
        "discriminator_parameters": sum(p.numel() for p in discriminator.parameters()),
        "num_ws": generator.num_ws,
        "synthesis_channels": generator.synthesis.channels,
        "discriminator_channels": discriminator.channels,
        "phase_seconds": {phase: 0.0 for phase in phases},
        "phase_peaks": {
            phase: {
                "peak_gpu_allocated_bytes": None,
                "peak_gpu_reserved_bytes": None,
            }
            for phase in phases
        },
        "sessions": [],
    }


def runtime_report(runtime, training_state, completed_epoch):
    train_seconds = runtime["phase_seconds"]["train"]
    overall_peaks = {
        key: max(
            (
                peaks[key]
                for peaks in runtime["phase_peaks"].values()
                if peaks[key] is not None
            ),
            default=None,
        )
        for key in ("peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes")
    }
    return {
        "completed_epoch": completed_epoch,
        "training_state": dict(training_state),
        "kimg": training_state["images_seen"] / 1000,
        "training_images_per_second": (
            training_state["images_seen"] / train_seconds if train_seconds else None
        ),
        **overall_peaks,
        "timing_scope": (
            "train includes batch loading, transfer, forward, backward and optimizer "
            "updates; excludes validation, sampling, plotting and checkpoint I/O. "
            "CUDA phases are synchronized. Session wall time includes maintenance."
        ),
        "gpu_memory_scope": (
            "PyTorch allocator peaks for this process, not total device usage. "
            "Reserved memory may include cache retained from an earlier phase. "
            "CPU-only phases use null, not zero."
        ),
        **runtime,
    }


# Restore at epoch boundaries. Configuration is read before constructing models.
def load_training_checkpoint(
    checkpoint,
    generator,
    discriminator,
    optimizer_g,
    optimizer_d,
    train_loader,
    device,
    expected_config,
    data_signature,
    *,
    generator_ema,
    regularization_state,
):
    if checkpoint.get("checkpoint_version") != 2:
        raise ValueError("Resume requires checkpoint_version=2 with EMA state support.")
    if checkpoint.get("model_name") != "stylegan2":
        raise ValueError("Expected a StyleGAN2 checkpoint.")
    saved_config = checkpoint["config"]
    for key, value in expected_config.items():
        if key != "epochs" and saved_config.get(key) != value:
            raise ValueError(f"Checkpoint configuration mismatch: {key}.")
    if checkpoint["runtime"]["data_signature"] != data_signature:
        raise ValueError("Dataset metadata, split manifest or split sizes changed.")

    completed_epoch = int(checkpoint["epoch"])
    if expected_config["epochs"] < completed_epoch:
        raise ValueError("--epochs cannot be smaller than the completed epoch.")
    history = checkpoint["history"]
    if any(
        len(history[key]) != completed_epoch
        for key in ("train", "valid", "performance")
    ):
        raise ValueError("Checkpoint history does not match its completed epoch.")
    training_state = dict(checkpoint["training_state"])
    if min(training_state["global_step"], training_state["images_seen"]) < 0:
        raise ValueError("Invalid training counters.")
    fixed_noise = checkpoint["fixed_noise"]
    expected_shape = (
        expected_config["num_sample_images"],
        expected_config["generator_kwargs"]["z_dim"],
    )
    if tuple(fixed_noise.shape) != expected_shape:
        raise ValueError("Checkpoint fixed_noise shape does not match configuration.")

    generator.load_state_dict(checkpoint["generator_state_dict"], strict=True)
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"], strict=True)

    if expected_config["ema_enabled"] != (generator_ema is not None):
        raise ValueError("EMA configuration and model disagree.")

    if generator_ema is not None:
        ema_state = checkpoint.get("generator_ema_state_dict")

        if ema_state is None:
            raise ValueError("Checkpoint is missing EMA weights.")

        generator_ema.load_state_dict(ema_state, strict=True)
        generator_ema.eval()
        generator_ema.requires_grad_(False)

    optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
    optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])

    saved_regularization = checkpoint.get("regularization_state")

    if not isinstance(saved_regularization, dict):
        raise ValueError("Checkpoint is missing regularization state.")

    saved_pl_mean = saved_regularization.get("pl_mean")

    if (
        not isinstance(saved_pl_mean, torch.Tensor)
        or saved_pl_mean.ndim != 0
        or not saved_pl_mean.is_floating_point()
        or not torch.isfinite(saved_pl_mean).item()
        or saved_pl_mean.item() < 0
    ):
        raise ValueError("Invalid checkpoint pl_mean.")

    with torch.no_grad():
        regularization_state["pl_mean"].copy_(
            saved_pl_mean.to(device=device, dtype=torch.float32)
        )

    # Restore RNG states last, after model construction and state loading.
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    cuda_states = checkpoint["cuda_rng_state_all"]
    if device.type == "cuda":
        if cuda_states is None or len(cuda_states) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from the saved RNG state.")
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])
    if train_loader.generator is None or checkpoint["loader_rng_state"] is None:
        raise ValueError("A loader generator state is required for resume.")
    train_loader.generator.set_state(checkpoint["loader_rng_state"].cpu())
    return (
        completed_epoch,
        history,
        fixed_noise.to(device),
        training_state,
        checkpoint["runtime"],
    )


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
    training_state,
    runtime,
    device,
    test_metrics=None,
    *,
    generator_ema,
    regularization_state,
):
    checkpoint = {
        "checkpoint_version": 2,
        "model_name": config["model_name"],
        "epoch": completed_epoch,
        "generator_state_dict": generator.state_dict(),
        "generator_ema_state_dict": (
            generator_ema.state_dict() if generator_ema is not None else None
        ),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_g_state_dict": optimizer_g.state_dict(),
        "optimizer_d_state_dict": optimizer_d.state_dict(),
        "history": history,
        "fixed_noise": fixed_noise.detach().cpu(),
        "config": config,
        "training_state": dict(training_state),
        "regularization_state": {
            "pl_mean": (regularization_state["pl_mean"].detach().cpu().clone())
        },
        "runtime": runtime,
        "test_metrics": test_metrics,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if device.type == "cuda" else None
        ),
        "loader_rng_state": train_loader.generator.get_state().clone(),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(output_path)


@torch.no_grad()
def evaluate_stylegan2(
    generator,
    discriminator,
    data_loader,
    device,
    random_seed=12345,
    noise_mode="const",
):
    if noise_mode not in ("const", "none"):
        raise ValueError("Deterministic evaluation requires const or none noise.")
    generator_was_training = generator.training
    discriminator_was_training = discriminator.training
    generator.eval()
    discriminator.eval()
    rng = torch.Generator(device=device).manual_seed(random_seed)
    totals = {}
    sample_count = 0
    try:
        # DataLoader iterator creation also draws a CPU base seed. Keep that
        # draw isolated from the training RNG, not just the latent-vector draws.
        with torch.random.fork_rng(devices=[]):
            for real_images in data_loader:
                real_images = real_images.to(device, non_blocking=device.type == "cuda")
                batch_size = real_images.shape[0]
                z = torch.randn(
                    batch_size, generator.z_dim, device=device, generator=rng
                )
                fake_images = generator(z, noise_mode=noise_mode, update_w_avg=False)
                real_logits = discriminator(real_images)
                fake_logits = discriminator(fake_images)
                d_loss, real_loss, fake_loss = discriminator_logistic_loss(
                    real_logits, fake_logits
                )
                g_loss = generator_logistic_loss(fake_logits)
                if not (
                    torch.isfinite(d_loss).item() and torch.isfinite(g_loss).item()
                ):
                    raise FloatingPointError("Non-finite StyleGAN2 evaluation loss.")
                real_score = real_logits.mean().item()
                fake_score = fake_logits.mean().item()
                metrics = {
                    "generator_loss": g_loss.item(),
                    "discriminator_loss": d_loss.item(),
                    "real_loss": real_loss.item(),
                    "fake_loss": fake_loss.item(),
                    "real_score": real_score,
                    "fake_score": fake_score,
                    "score_gap": real_score - fake_score,
                }
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.0) + value * batch_size
                sample_count += batch_size
    finally:
        generator.train(generator_was_training)
        discriminator.train(discriminator_was_training)
    if sample_count == 0:
        raise RuntimeError("Evaluation loader produced no images.")
    return {key: value / sample_count for key, value in totals.items()}


@torch.no_grad()
def update_generator_ema(
    generator_ema,
    generator,
    *,
    batch_size,
    images_seen_before,
    ema_kimg,
    ema_rampup,
):
    half_life_images = ema_kimg * 1000.0

    if ema_rampup is not None:
        half_life_images = min(half_life_images, images_seen_before * ema_rampup)

    beta = 0.5 ** (batch_size / max(half_life_images, 1e-8))

    source_parameters = dict(generator.named_parameters())

    for name, ema_parameter in generator_ema.named_parameters():
        ema_parameter.lerp_(source_parameters[name], 1.0 - beta)

    source_buffers = dict(generator.named_buffers())

    for name, ema_buffer in generator_ema.named_buffers():
        ema_buffer.copy_(source_buffers[name])

    return beta


def run_generator_for_training(
    generator,
    z,
    *,
    style_mixing_prob,
    noise_mode,
    update_w_avg,
):
    ws = generator.mapping(z, update_w_avg=update_w_avg)

    should_mix = (
        style_mixing_prob > 0
        and generator.num_ws > 1
        and torch.rand((), device=z.device).item() < style_mixing_prob
    )

    if should_mix:
        cutoff = torch.randint(
            low=1, high=generator.num_ws, size=(), device=z.device
        ).item()

        second_z = torch.randn_like(z)

        second_ws = generator.mapping(second_z, update_w_avg=False)

        ws = torch.cat([ws[:, :cutoff], second_ws[:, cutoff:]], dim=1)

    images = generator.synthesis(ws, noise_mode=noise_mode)

    return images, ws


def generator_path_length_penalty(
    images,
    ws,
    previous_mean,
    *,
    decay,
    eps,
):
    height, width = images.shape[-2:]

    # Random image-space direction, normalized by spatial resolution.
    image_noise = torch.randn_like(images)
    image_noise = image_noise / ((height * width) ** 0.5)

    projected_output = (images * image_noise).sum()

    # Differentiate the generated image projection with respect to styles.
    style_gradients = torch.autograd.grad(
        outputs=projected_output, inputs=ws, create_graph=True
    )[0]

    # [B, num_ws, w_dim] -> [B]
    lengths = style_gradients.square().sum(dim=2).mean(dim=1).add(eps).sqrt()

    next_mean = previous_mean.detach().lerp(lengths.mean(), decay)

    penalty = (lengths - next_mean).square().mean()

    return (penalty, lengths.detach().mean(), next_mean.detach())


def train_step_stylegan2(
    generator,
    discriminator,
    real_images,
    optimizer_g,
    optimizer_d,
    *,
    global_step,
    r1_gamma,
    r1_interval,
    style_mixing_prob,
    regularization_state,
    pl_weight,
    pl_interval,
    pl_batch_shrink,
    pl_decay,
    pl_eps,
    noise_mode="random",
):
    generator.train()
    discriminator.train()

    batch_size = real_images.shape[0]
    device = real_images.device

    # -------------------------
    # 1. Update discriminator.
    # -------------------------
    generator.requires_grad_(False)
    discriminator.requires_grad_(True)

    optimizer_g.zero_grad(set_to_none=True)
    optimizer_d.zero_grad(set_to_none=True)

    z = torch.randn(batch_size, generator.z_dim, device=device)

    with torch.no_grad():
        fake_images, _ = run_generator_for_training(
            generator,
            z,
            style_mixing_prob=style_mixing_prob,
            noise_mode=noise_mode,
            update_w_avg=False,
        )

    real_logits = discriminator(real_images)
    fake_logits_d = discriminator(fake_images)

    d_loss, real_loss, fake_loss = discriminator_logistic_loss(
        real_logits, fake_logits_d
    )

    if not torch.isfinite(d_loss).item():
        raise FloatingPointError("Non-finite discriminator loss.")

    d_loss.backward()
    optimizer_d.step()
    optimizer_d.zero_grad(set_to_none=True)
    r1_penalty_value = None
    r1_loss_value = None

    apply_r1 = r1_gamma > 0 and global_step % r1_interval == 0

    if apply_r1:
        # D is still trainable here; G is still frozen.
        optimizer_d.zero_grad(set_to_none=True)

        r1_penalty = discriminator_r1_penalty(discriminator, real_images)

        r1_loss = 0.5 * r1_gamma * r1_penalty
        r1_backward_loss = r1_loss * r1_interval

        if not torch.isfinite(r1_backward_loss).item():
            raise FloatingPointError("Non-finite R1 loss.")

        r1_backward_loss.backward()
        optimizer_d.step()
        optimizer_d.zero_grad(set_to_none=True)

        r1_penalty_value = r1_penalty.detach().item()
        r1_loss_value = r1_loss.detach().item()

    # -------------------------
    # 2. Update generator.
    # -------------------------
    generator.requires_grad_(True)
    discriminator.requires_grad_(False)

    optimizer_g.zero_grad(set_to_none=True)

    z = torch.randn(batch_size, generator.z_dim, device=device)

    fake_images, _ = run_generator_for_training(
        generator,
        z,
        style_mixing_prob=style_mixing_prob,
        noise_mode=noise_mode,
        update_w_avg=True,
    )

    fake_logits_g = discriminator(fake_images)
    g_loss = generator_logistic_loss(fake_logits_g)

    if not torch.isfinite(g_loss).item():
        raise FloatingPointError("Non-finite generator loss.")

    g_loss.backward()
    optimizer_g.step()
    optimizer_g.zero_grad(set_to_none=True)

    # -------------------------
    # 3. Optional generator path length regularization.
    # -------------------------
    pl_penalty_value = None
    pl_loss_value = None
    pl_length_value = None
    pl_sample_count = 0

    apply_pl = pl_weight > 0 and global_step % pl_interval == 0

    if apply_pl:
        pl_batch_size = max(1, batch_size // pl_batch_shrink)

        optimizer_g.zero_grad(set_to_none=True)

        pl_z = torch.randn(pl_batch_size, generator.z_dim, device=device)

        pl_images, pl_ws = run_generator_for_training(
            generator,
            pl_z,
            style_mixing_prob=style_mixing_prob,
            noise_mode=noise_mode,
            update_w_avg=False,
        )

        pl_penalty, mean_length, next_pl_mean = generator_path_length_penalty(
            pl_images,
            pl_ws,
            regularization_state["pl_mean"],
            decay=pl_decay,
            eps=pl_eps,
        )

        pl_loss = pl_weight * pl_penalty

        # Compensate for applying this loss only every pl_interval batches.
        pl_backward_loss = pl_loss * pl_interval

        if not torch.isfinite(pl_backward_loss).item():
            raise FloatingPointError("Non-finite PL loss.")

        pl_backward_loss.backward()
        optimizer_g.step()
        optimizer_g.zero_grad(set_to_none=True)

        with torch.no_grad():
            regularization_state["pl_mean"].copy_(next_pl_mean)

        pl_penalty_value = pl_penalty.detach().item()
        pl_loss_value = pl_loss.detach().item()
        pl_length_value = mean_length.item()
        pl_sample_count = pl_batch_size

    discriminator.requires_grad_(True)

    real_score = real_logits.detach().mean().item()
    fake_score = fake_logits_d.detach().mean().item()

    return {
        "generator_loss": g_loss.item(),
        "discriminator_loss": d_loss.item(),
        "real_loss": real_loss.item(),
        "fake_loss": fake_loss.item(),
        "real_score": real_score,
        "fake_score": fake_score,
        "score_gap": real_score - fake_score,
        "r1_penalty": r1_penalty_value,
        "r1_loss": r1_loss_value,
        "pl_penalty": pl_penalty_value,
        "pl_loss": pl_loss_value,
        "pl_length": pl_length_value,
        "pl_samples": pl_sample_count,
    }


def train_one_epoch_stylegan2(
    generator,
    discriminator,
    train_loader,
    optimizer_g,
    optimizer_d,
    device,
    training_state,
    *,
    generator_ema,
    ema_kimg,
    ema_rampup,
    r1_gamma,
    r1_interval,
    style_mixing_prob,
    regularization_state,
    pl_weight,
    pl_interval,
    pl_batch_shrink,
    pl_decay,
    pl_eps,
    noise_mode="random",
):
    totals = {}
    sample_count = 0

    r1_penalty_sum = 0.0
    r1_loss_sum = 0.0
    r1_sample_count = 0
    r1_steps = 0

    pl_totals = {"pl_penalty": 0.0, "pl_loss": 0.0, "pl_length": 0.0}
    pl_sample_count = 0
    pl_steps = 0

    for real_images in train_loader:
        real_images = real_images.to(device, non_blocking=device.type == "cuda")

        metrics = train_step_stylegan2(
            generator,
            discriminator,
            real_images,
            optimizer_g,
            optimizer_d,
            global_step=training_state["global_step"],
            r1_gamma=r1_gamma,
            r1_interval=r1_interval,
            style_mixing_prob=style_mixing_prob,
            regularization_state=regularization_state,
            pl_weight=pl_weight,
            pl_interval=pl_interval,
            pl_batch_shrink=pl_batch_shrink,
            pl_decay=pl_decay,
            pl_eps=pl_eps,
            noise_mode=noise_mode,
        )

        batch_size = real_images.shape[0]

        # Update EMA after all generator updates for this batch.
        if generator_ema is not None:
            update_generator_ema(
                generator_ema,
                generator,
                batch_size=batch_size,
                images_seen_before=training_state["images_seen"],
                ema_kimg=ema_kimg,
                ema_rampup=ema_rampup,
            )

        for key, value in metrics.items():
            if key in {
                "r1_penalty",
                "r1_loss",
                "pl_penalty",
                "pl_loss",
                "pl_length",
                "pl_samples",
            }:
                continue

            totals[key] = totals.get(key, 0.0) + value * batch_size

        if metrics["r1_penalty"] is not None:
            r1_penalty_sum += metrics["r1_penalty"] * batch_size
            r1_loss_sum += metrics["r1_loss"] * batch_size
            r1_sample_count += batch_size
            r1_steps += 1

        if metrics["pl_samples"] > 0:
            count = metrics["pl_samples"]

            for key in pl_totals:
                pl_totals[key] += metrics[key] * count

            pl_sample_count += count
            pl_steps += 1

        sample_count += batch_size

        training_state["global_step"] += 1
        training_state["images_seen"] += batch_size

    if sample_count == 0:
        raise RuntimeError("Training loader produced no images.")

    epoch_metrics = {key: value / sample_count for key, value in totals.items()}

    epoch_metrics.update(
        {
            "r1_penalty": (
                r1_penalty_sum / r1_sample_count if r1_sample_count > 0 else None
            ),
            "r1_loss": (r1_loss_sum / r1_sample_count if r1_sample_count > 0 else None),
            "r1_steps": r1_steps,
        }
    )

    for key, total in pl_totals.items():
        epoch_metrics[key] = total / pl_sample_count if pl_sample_count > 0 else None

    epoch_metrics["pl_steps"] = pl_steps
    epoch_metrics["pl_mean"] = regularization_state["pl_mean"].item()

    return epoch_metrics


def main():
    parser = argparse.ArgumentParser(description="Train StyleGAN2 without ADA.")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--epochs", type=int, help="Total target epochs, including completed epochs."
    )
    parser.add_argument("--batch-size", type=int, help="New runs only; default: 4.")
    parser.add_argument(
        "--num-workers", type=int, help="New runs only; default: 4 on CUDA."
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    session_started = time.perf_counter()
    resume_checkpoint = args.resume.resolve() if args.resume is not None else None
    checkpoint = None
    if resume_checkpoint is not None:
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(resume_checkpoint)
        checkpoint = torch.load(
            resume_checkpoint, map_location="cpu", weights_only=True
        )

    config = build_config(
        args, device, checkpoint["config"] if checkpoint is not None else None
    )
    torch.manual_seed(config["random_seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["random_seed"])
        torch.backends.cudnn.benchmark = config["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = config["cudnn_deterministic"]
        torch.backends.cudnn.allow_tf32 = config["allow_tf32"]
        torch.backends.cuda.matmul.allow_tf32 = config["allow_tf32"]
    print("Using device:", device, flush=True)

    home_dir = Path(__file__).resolve().parent
    if resume_checkpoint is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
        run_dir = home_dir / "runs" / f"{config['run_name']}_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir = resume_checkpoint.parent.parent
    checkpoint_dir = run_dir / "checkpoints"
    sample_dir = run_dir / "samples"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    regularization_state = {
        "pl_mean": torch.zeros((), device=device, dtype=torch.float32)
    }

    (
        train_dataset,
        train_loader,
        test_dataset,
        test_loader,
        valid_dataset,
        valid_loader,
    ) = dataset.build_dataloaders(
        data_root=config["data_root"],
        metadata_path=config["metadata_path"],
        manifest_path=config["manifest_path"],
        included_classes=tuple(config["included_classes"]),
        val_ratio=config["val_ratio"],
        image_size=config["generator_kwargs"]["image_resolution"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda",
        split_seed=config["split_seed"],
        loader_seed=config["loader_seed"],
        drop_last_train=config["drop_last_train"],
        persistent_workers=config["persistent_workers"],
    )
    data_signature = {
        "manifest_sha256": file_sha256(config["manifest_path"]),
        "metadata_sha256": file_sha256(config["metadata_path"]),
        "split_images": {
            "train": len(train_dataset),
            "validation": len(valid_dataset),
            "test": len(test_dataset),
        },
    }
    print("Dataset images:", data_signature["split_images"], flush=True)

    generator = modules.StyleGAN2Generator(**config["generator_kwargs"]).to(
        device=device, dtype=torch.float32
    )
    generator_ema = None

    if config["ema_enabled"]:
        generator_ema = copy.deepcopy(generator)
        generator_ema.eval()
        generator_ema.requires_grad_(False)

    discriminator = modules.StyleGAN2Discriminator(**config["discriminator_kwargs"]).to(
        device=device, dtype=torch.float32
    )
    optimizer_g = torch.optim.Adam(
        generator.parameters(), **config["optimizer_g_kwargs"]
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(), **config["optimizer_d_kwargs"]
    )

    history = {"train": [], "valid": [], "performance": []}
    training_state = {"global_step": 0, "images_seen": 0}
    runtime = new_runtime(generator, discriminator, data_signature)
    start_epoch = 0
    fixed_rng = torch.Generator(device=device).manual_seed(config["fixed_noise_seed"])
    fixed_noise = torch.randn(
        config["num_sample_images"], generator.z_dim, device=device, generator=fixed_rng
    )
    if checkpoint is not None:
        start_epoch, history, fixed_noise, training_state, runtime = (
            load_training_checkpoint(
                checkpoint,
                generator,
                discriminator,
                optimizer_g,
                optimizer_d,
                train_loader,
                device,
                config,
                data_signature,
                generator_ema=generator_ema,
                regularization_state=regularization_state,
            )
        )
        del checkpoint
        print(f"Resumed after epoch {start_epoch}: {training_state}", flush=True)

    runtime["sessions"].append(
        {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
            "start_epoch": start_epoch,
            "target_epochs": config["epochs"],
            "environment": describe_environment(device),
            "source_sha256": {
                name: file_sha256(home_dir / name)
                for name in ("modules.py", "dataset.py", "train.py", "predict.py")
            },
            "wall_seconds": 0.0,
        }
    )
    if device.type == "cuda":
        print("GPU:", runtime["sessions"][-1]["environment"]["gpu_name"], flush=True)

    # Sole config.json write: fresh runs and resumes both use the resolved config.
    # Resumes may extend epochs; all other training settings come from the checkpoint.
    write_json(run_dir / "config.json", config)

    if resume_checkpoint is None:
        with torch.random.fork_rng(devices=[]):
            save_real_grid(
                valid_loader,
                sample_dir / "real_validation.png",
                num_images=config["num_sample_images"],
                nrow=config["sample_nrow"],
            )

    def save_progress(paths, completed_epoch, test_metrics=None):
        runtime["sessions"][-1]["wall_seconds"] = time.perf_counter() - session_started
        for path in paths:
            save_checkpoint(
                output_path=path,
                completed_epoch=completed_epoch,
                generator=generator,
                generator_ema=generator_ema,
                regularization_state=regularization_state,
                discriminator=discriminator,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
                history=history,
                fixed_noise=fixed_noise,
                config=config,
                train_loader=train_loader,
                training_state=training_state,
                runtime=runtime,
                device=device,
                test_metrics=test_metrics,
            )
        torch.save(history, run_dir / "history.pt")
        runtime["sessions"][-1]["wall_seconds"] = time.perf_counter() - session_started
        write_json(
            run_dir / "runtime.json",
            runtime_report(runtime, training_state, completed_epoch),
        )

    for epoch_index in range(start_epoch, config["epochs"]):
        epoch_started = time.perf_counter()
        images_before = training_state["images_seen"]
        steps_before = training_state["global_step"]
        train_metrics, train_perf = measure_phase(
            device,
            lambda: train_one_epoch_stylegan2(
                generator,
                discriminator,
                train_loader,
                optimizer_g,
                optimizer_d,
                device,
                training_state,
                generator_ema=generator_ema,
                ema_kimg=config["ema_kimg"],
                ema_rampup=config["ema_rampup"],
                r1_gamma=config["r1_gamma"],
                r1_interval=config["r1_interval"],
                style_mixing_prob=config["style_mixing_prob"],
                regularization_state=regularization_state,
                pl_weight=config["pl_weight"],
                pl_interval=config["pl_interval"],
                pl_batch_shrink=config["pl_batch_shrink"],
                pl_decay=config["pl_decay"],
                pl_eps=config["pl_eps"],
                noise_mode=config["train_noise_mode"],
            ),
        )
        train_perf["images"] = training_state["images_seen"] - images_before
        train_perf["batches"] = training_state["global_step"] - steps_before
        train_perf["images_per_second"] = train_perf["images"] / train_perf["seconds"]
        train_perf["seconds_per_batch"] = train_perf["seconds"] / train_perf["batches"]
        record_phase(runtime, "train", train_perf)

        valid_metrics, valid_perf = measure_phase(
            device,
            lambda: evaluate_stylegan2(
                generator,
                discriminator,
                valid_loader,
                device,
                random_seed=config["validation_seed"],
                noise_mode=config["evaluation_noise_mode"],
            ),
        )
        record_phase(runtime, "validation", valid_perf)
        completed_epoch = epoch_index + 1
        sample_perf = None
        if (
            completed_epoch == 1
            or completed_epoch % config["sample_every"] == 0
            or completed_epoch == config["epochs"]
        ):
            sample_generator = generator_ema if generator_ema is not None else generator

            _, sample_perf = measure_phase(
                device,
                lambda: predict.save_generated_grid(
                    sample_generator,
                    fixed_noise,
                    sample_dir / f"epoch_{completed_epoch:03d}.png",
                    nrow=config["sample_nrow"],
                    batch_size=config["sample_batch_size"],
                    noise_mode=config["sample_noise_mode"],
                ),
            )

            record_phase(runtime, "sampling", sample_perf)

        history["train"].append(train_metrics)
        history["valid"].append(valid_metrics)
        history["performance"].append(
            {
                "epoch": completed_epoch,
                "global_step": training_state["global_step"],
                "images_seen": training_state["images_seen"],
                "train": train_perf,
                "validation": valid_perf,
                "sampling": sample_perf,
                "wall_seconds_before_checkpoint": time.perf_counter() - epoch_started,
            }
        )
        allocated = train_perf["peak_gpu_allocated_bytes"]
        peak_text = (
            f"{allocated / 1024**3:.2f} GiB" if allocated is not None else "N/A (CPU)"
        )
        reserved = train_perf["peak_gpu_reserved_bytes"]
        reserved_text = (
            f"{reserved / 1024**3:.2f} GiB" if reserved is not None else "N/A (CPU)"
        )

        pl_loss_value = train_metrics["pl_loss"]
        pl_loss_text = f"{pl_loss_value:.4f}" if pl_loss_value is not None else "N/A"

        print(
            f"Epoch [{completed_epoch:03d}/{config['epochs']:03d}] "
            f"train_time={train_perf['seconds']:.1f}s "
            f"valid_time={valid_perf['seconds']:.1f}s "
            f"train_peak_allocated={peak_text} "
            f"train_peak_reserved={reserved_text} "
            f"kimg={training_state['images_seen'] / 1000:.3f} | "
            f"train G={train_metrics['generator_loss']:.4f} "
            f"D={train_metrics['discriminator_loss']:.4f} "
            f"real={train_metrics['real_score']:.4f} "
            f"fake={train_metrics['fake_score']:.4f} "
            f"gap={train_metrics['score_gap']:.4f} "
            f"PL={pl_loss_text} "
            f"PL_steps={train_metrics['pl_steps']} "
            f"PL_mean={train_metrics['pl_mean']:.4f} | "
            f"valid G={valid_metrics['generator_loss']:.4f} "
            f"D={valid_metrics['discriminator_loss']:.4f} "
            f"real={valid_metrics['real_score']:.4f} "
            f"fake={valid_metrics['fake_score']:.4f} "
            f"gap={valid_metrics['score_gap']:.4f}",
            flush=True,
        )
        plot_training_history(history, run_dir / "training_history.png")
        paths = [checkpoint_dir / "latest.pt"]
        if completed_epoch % config["checkpoint_every"] == 0:
            paths.append(checkpoint_dir / f"epoch_{completed_epoch:03d}.pt")
        save_progress(paths, completed_epoch)

    test_metrics, test_perf = measure_phase(
        device,
        lambda: evaluate_stylegan2(
            generator,
            discriminator,
            test_loader,
            device,
            random_seed=config["validation_seed"] + 1,
            noise_mode=config["evaluation_noise_mode"],
        ),
    )
    record_phase(runtime, "test", test_perf)
    print(
        f"Test | G={test_metrics['generator_loss']:.4f} "
        f"D={test_metrics['discriminator_loss']:.4f} "
        f"real={test_metrics['real_score']:.4f} "
        f"fake={test_metrics['fake_score']:.4f} "
        f"gap={test_metrics['score_gap']:.4f}",
        flush=True,
    )
    write_json(
        run_dir / "test_metrics.json",
        {
            **test_metrics,
            "performance": test_perf,
            "epoch": config["epochs"],
            "model_weights": "generator",
        },
    )
    runtime["sessions"][-1]["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    save_progress(
        [checkpoint_dir / "latest.pt", checkpoint_dir / "final.pt"],
        config["epochs"],
        test_metrics=test_metrics,
    )
    print(
        f"Completed. Cumulative training time: "
        f"{runtime['phase_seconds']['train']:.1f}s. Results: {run_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
