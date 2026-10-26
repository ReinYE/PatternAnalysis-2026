import argparse
import gc
import math
import random
import time

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

import evaluation as ev
from predict import load_generator

MODELS = ("dcgan", "stylegan2")
DIAGNOSES = ("AD", "NC")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    ev.atomic_write_ssim_csv(path, rows)


def random_z(dimension, seed, device):
    # Generate on CPU so seed-to-latent mapping does not depend on GPU.
    rng = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(1, dimension, generator=rng, dtype=torch.float32).to(device)


def to_uint8(image):
    """Fixed display conversion; no per-image contrast normalization."""
    if image.shape != (1, 1, 256, 256):
        raise ValueError(f"Unexpected image shape: {tuple(image.shape)}")

    if not torch.isfinite(image).all().item():
        raise FloatingPointError("Image contains NaN/Inf.")

    # Input is generator output in nominal [-1, 1].
    pixels = (
        ((image.detach().cpu() + 1.0) / 2.0)
        .clamp(0, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
    )
    return pixels[0, 0].numpy()


def save_gray(path, pixels):
    Image.fromarray(pixels).save(path)


def save_strip(images, labels, path, title):
    figure, axes = plt.subplots(
        1,
        len(images),
        figsize=(2.2 * len(images), 3.1),
        squeeze=False,
    )

    for axis, pixels, label in zip(axes[0], images, labels):
        axis.imshow(pixels, cmap="gray", vmin=0, vmax=255)
        axis.set_title(label, fontsize=9)
        axis.axis("off")

    figure.suptitle(title, fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(path, dpi=160)
    plt.close(figure)


def generate(generator, model_name, latent, space):
    """
    Deliberately NOT decorated with no_grad:
    projection requires gradients with respect to latent.
    """
    if model_name == "dcgan":
        if space != "z":
            raise ValueError("DCGAN supports only Z here.")
        image = generator(latent)

    elif space == "z":
        image = generator(
            latent,
            noise_mode="const",
            update_w_avg=False,
        )

    elif space == "w":
        if latent.shape != (1, generator.w_dim):
            raise ValueError("Expected one shared W vector.")

        ws = latent.unsqueeze(1).repeat(1, generator.num_ws, 1)
        image = generator.synthesis(ws, noise_mode="const")

    else:
        raise ValueError(f"Unsupported space: {space}")

    if image.shape != (1, 1, 256, 256):
        raise ValueError(f"Unexpected output: {tuple(image.shape)}")

    if not torch.isfinite(image).all().item():
        raise FloatingPointError("Non-finite generator output.")

    return image


def linear_path(a, b, frames):
    return [(1.0 - t) * a + t * b for t in np.linspace(0.0, 1.0, frames)]


def spherical_path(a, b, frames):
    """
    Interpolate direction on a sphere and interpolate endpoint radii.
    Explicitly preserve endpoints.
    """
    norm_a = a.norm()
    norm_b = b.norm()

    if norm_a.item() < 1e-8 or norm_b.item() < 1e-8:
        return linear_path(a, b, frames)

    direction_a = a / norm_a
    direction_b = b / norm_b

    cosine = torch.sum(direction_a * direction_b).clamp(-1, 1)
    angle = torch.acos(cosine)
    sine = torch.sin(angle)

    if abs(sine.item()) < 1e-6:
        return linear_path(a, b, frames)

    result = []

    for index, t in enumerate(np.linspace(0.0, 1.0, frames)):
        if index == 0:
            latent = a
        elif index == frames - 1:
            latent = b
        else:
            direction = (
                torch.sin((1.0 - t) * angle) / sine * direction_a
                + torch.sin(t * angle) / sine * direction_b
            )
            radius = (1.0 - t) * norm_a + t * norm_b
            latent = direction * radius

        result.append(latent)

    return result


@torch.no_grad()
def export_path(
    generator,
    model_name,
    space,
    latents,
    output,
    title,
    original_before=None,
    original_after=None,
):
    output.mkdir(parents=True, exist_ok=True)

    pixels = []
    frame_rows = []

    for index, latent in enumerate(latents):
        raw = generate(generator, model_name, latent, space)
        array = to_uint8(raw)

        filename = f"frame_{index:02d}.png"
        save_gray(output / filename, array)
        pixels.append(array)

        frame_rows.append(
            {
                "frame": index,
                "alpha": index / (len(latents) - 1),
                "file": filename,
                "space": space,
                "raw_min": float(raw.min()),
                "raw_max": float(raw.max()),
                "clipped_fraction": float(((raw < -1) | (raw > 1)).float().mean()),
            }
        )

    torch.save(
        {
            "space": space,
            "latents": torch.cat([latent.detach().cpu() for latent in latents]),
            "alphas": torch.linspace(0, 1, len(latents)),
            "noise_mode": ("const" if model_name == "stylegan2" else None),
        },
        output / "path_latents.pt",
    )
    write_csv(output / "frames.csv", frame_rows)

    labels = [f"alpha={t:.2f}" for t in np.linspace(0, 1, len(latents))]

    displayed = list(pixels)

    if original_before is not None:
        displayed.insert(0, original_before)
        labels.insert(0, "Real A\nnot G output")

    if original_after is not None:
        displayed.append(original_after)
        labels.append("Real target\nnot G output")

    save_strip(
        displayed,
        labels,
        output / "interpolation.png",
        title,
    )


def select_targets(args, stage_a):
    snapshot = args.cohort_dir / "source_splits.json"

    ev.verify_hash(
        snapshot,
        stage_a["real_reference"]["source_manifest_sha256"],
        "split snapshot",
    )

    payload = ev.read_json(snapshot)
    rng = random.Random(args.seed)
    selected = []
    used_patients = set()

    for split in ("train", "validation"):
        records = payload["splits"][split]

        for diagnosis in DIAGNOSES:
            by_patient = defaultdict(list)

            for record in records:
                if record["diagnosis"] == diagnosis:
                    by_patient[str(record["subject_id"])].append(record)

            candidates = sorted(
                patient for patient in by_patient if patient not in used_patients
            )

            if len(candidates) < args.per_class:
                raise ValueError(f"Insufficient distinct patients: {split}/{diagnosis}")

            patients = rng.sample(candidates, args.per_class)

            for index, patient in enumerate(patients):
                choices = sorted(
                    by_patient[patient],
                    key=lambda row: row["relative_path"],
                )
                row = dict(rng.choice(choices))
                row["reference_id"] = f"{split}_{diagnosis}_{index:02d}"
                row["evaluation_split"] = split
                row["selection_index"] = index

                path = ev.resolve_image_path(args.data_root, row["relative_path"])
                row["image_file_sha256"] = ev.file_sha256(path)

                selected.append(row)
                used_patients.add(patient)

    spec = {
        "root": args.data_root,
        "rows": selected,
        "id_field": "reference_id",
        "manifest_sha256": ev.object_sha256(selected),
        "kind": "real",
    }

    dataset = ev.EvaluationImageDataset("projection_targets", spec)

    targets = {}

    for index, row in enumerate(selected):
        pixels, returned_index = dataset[index]

        if returned_index != index:
            raise ValueError("Target index mismatch.")

        # [1, 1, H, W], [0, 1].
        targets[row["reference_id"]] = pixels.unsqueeze(0).float() / 255.0

    return selected, targets


@torch.no_grad()
def latent_statistics(generator, model_name, args, device):
    if model_name == "dcgan":
        dimension = generator.latent_dim
        return (
            torch.zeros(1, dimension, device=device),
            torch.ones(1, dimension, device=device),
        )

    rng = torch.Generator(device="cpu").manual_seed(args.seed + 500)
    collected = []

    for start in range(0, args.w_samples, 64):
        count = min(64, args.w_samples - start)
        z = torch.randn(
            count,
            generator.z_dim,
            generator=rng,
        ).to(device)

        ws = generator.mapping(z, update_w_avg=False)
        collected.append(ws[:, 0, :].cpu())

    w = torch.cat(collected, dim=0)
    center = w.mean(dim=0, keepdim=True)
    scale = w.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)

    return center.to(device), scale.to(device)


def projection_objective(raw, target, u, args):
    """
    Optimize raw floating-point output, without clipping.
    Clipping here could suppress useful gradients outside display range.
    """
    predicted = (raw + 1.0) / 2.0

    # Give target foreground more weight while retaining background errors.
    weight = 1.0 + 4.0 * (target > 0.05).to(target.dtype)

    multiscale = raw.new_zeros(())

    for factor, coefficient in ((1, 0.5), (2, 0.3), (4, 0.2)):
        if factor == 1:
            p, t, w = predicted, target, weight
        else:
            p = F.avg_pool2d(predicted, factor)
            t = F.avg_pool2d(target, factor)
            w = F.avg_pool2d(weight, factor)

        weighted_l1 = ((p - t).abs() * w).sum() / w.sum()
        multiscale = multiscale + coefficient * weighted_l1

    pred_dx = predicted[:, :, :, 1:] - predicted[:, :, :, :-1]
    real_dx = target[:, :, :, 1:] - target[:, :, :, :-1]

    pred_dy = predicted[:, :, 1:, :] - predicted[:, :, :-1, :]
    real_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

    gradient = (pred_dx - real_dx).abs().mean() + (pred_dy - real_dy).abs().mean()

    prior = u.square().mean()

    total = multiscale + args.gradient_weight * gradient + args.prior_weight * prior

    return total, multiscale, gradient, prior


def projection_metrics(raw, target):
    reconstruction = to_uint8(raw).astype(np.float64) / 255.0
    target_array = target[0, 0].detach().cpu().numpy().astype(np.float64)

    difference = np.abs(reconstruction - target_array)
    support = (reconstruction > 0.05) | (target_array > 0.05)

    return {
        "ssim": ev.calculate_image_ssim(target_array, reconstruction),
        "mae_full": float(difference.mean()),
        "mae_support": (float(difference[support].mean()) if support.any() else None),
        "mse_full": float(np.mean((reconstruction - target_array) ** 2)),
        "raw_min": float(raw.min()),
        "raw_max": float(raw.max()),
        "clipped_fraction": float(((raw < -1) | (raw > 1)).float().mean()),
    }


def save_projection_panel(target, raw, path, sample_id, metrics):
    real = target[0, 0].cpu().numpy()
    reconstructed = to_uint8(raw).astype(np.float64) / 255.0
    difference = np.abs(real - reconstructed)

    figure, axes = plt.subplots(1, 3, figsize=(10, 3.6))

    axes[0].imshow(real, cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(reconstructed, cmap="gray", vmin=0, vmax=1)
    axes[2].imshow(difference, cmap="magma", vmin=0, vmax=1)

    axes[0].set_title("Real target")
    axes[1].set_title(f"Projection\nSSIM={metrics['ssim']:.4f}")
    axes[2].set_title("Absolute difference [0, 1]")

    for axis in axes:
        axis.axis("off")

    figure.suptitle(sample_id)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def project_one(
    generator,
    model_name,
    row,
    target_cpu,
    center,
    scale,
    args,
    device,
    output,
    target_index,
):
    output.mkdir(parents=True, exist_ok=True)

    space = "w" if model_name == "stylegan2" else "z"
    dimension = generator.z_dim if model_name == "stylegan2" else generator.latent_dim
    target = target_cpu.to(device)

    best_loss = float("inf")
    best_latent = None
    best_restart = None
    best_step = None
    logs = []

    sync(device)
    started = time.perf_counter()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for restart in range(args.restarts):
        initialization_seed = args.seed + 10000 + target_index * 100 + restart

        with torch.no_grad():
            if model_name == "stylegan2" and restart == 0:
                # Shared-W mean as the first initialization.
                initial = center.clone()
            else:
                z = random_z(dimension, initialization_seed, device)

                if model_name == "stylegan2":
                    initial = generator.mapping(z, update_w_avg=False)[:, 0, :]
                else:
                    initial = z

            # Optimize standardized coordinates for a more comparable
            # numerical step size across latent dimensions.
            initial_u = (initial - center) / scale

        u = torch.nn.Parameter(initial_u.detach().clone())

        optimizer = torch.optim.Adam(
            [u],
            lr=args.projection_lr,
            betas=(0.9, 0.999),
        )

        # Includes evaluation of the initial and final states.
        # Exactly args.steps optimizer updates are performed.
        for step in range(args.steps + 1):
            optimizer.zero_grad(set_to_none=True)

            latent = center + scale * u
            raw = generate(generator, model_name, latent, space)

            total, multiscale, gradient, prior = projection_objective(
                raw, target, u, args
            )

            if not torch.isfinite(total).item():
                raise FloatingPointError("Non-finite projection loss.")

            score = float(total.detach())

            if score < best_loss:
                best_loss = score
                best_latent = latent.detach().clone()
                best_restart = restart
                best_step = step

            progress = step / args.steps
            learning_rate = args.projection_lr * (
                0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
            )

            logs.append(
                {
                    "restart": restart,
                    "initialization_seed": initialization_seed,
                    "step": step,
                    "total": score,
                    "multiscale_l1": float(multiscale.detach()),
                    "gradient_l1": float(gradient.detach()),
                    "latent_prior": float(prior.detach()),
                    "learning_rate": learning_rate,
                }
            )

            if step % 100 == 0 or step == args.steps:
                print(
                    f"{model_name} {row['reference_id']} "
                    f"restart={restart + 1}/{args.restarts} "
                    f"step={step}/{args.steps} loss={score:.6f}",
                    flush=True,
                )

            if step == args.steps:
                break

            total.backward()

            if u.grad is None or not torch.isfinite(u.grad).all().item():
                raise FloatingPointError("Invalid latent gradient.")

            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate

            optimizer.step()

        del optimizer, u, raw, total, multiscale, gradient, prior, latent

    with torch.no_grad():
        best_raw = generate(generator, model_name, best_latent, space).detach()

    sync(device)
    elapsed = time.perf_counter() - started

    metrics = projection_metrics(best_raw, target_cpu)

    metrics.update(
        {
            "model": model_name,
            "target_id": row["reference_id"],
            "evaluation_split": row["evaluation_split"],
            "diagnosis": row["diagnosis"],
            "subject_id": row["subject_id"],
            "space": space,
            "latent_dimension": int(best_latent.shape[1]),
            "best_total_loss": best_loss,
            "best_restart": best_restart,
            "best_step": best_step,
            "optimizer_steps_per_restart": args.steps,
            "restarts": args.restarts,
            "optimization_seconds": elapsed,
            "peak_allocated_gib": (
                torch.cuda.max_memory_allocated(device) / 1024**3
                if device.type == "cuda"
                else None
            ),
        }
    )

    torch.save(
        {
            "model": model_name,
            "space": space,
            "latent": best_latent.cpu(),
            "target_record": row,
            "metrics": metrics,
            "raw_reconstruction": best_raw.cpu(),
            "noise_mode": ("const" if model_name == "stylegan2" else None),
        },
        output / "projection.pt",
    )

    save_gray(output / "reconstruction.png", to_uint8(best_raw))
    write_csv(output / "optimization_history.csv", logs)
    ev.atomic_write_json(output / "metrics.json", metrics)

    save_projection_panel(
        target_cpu,
        best_raw,
        output / "reconstruction_comparison.png",
        row["reference_id"],
        metrics,
    )

    return best_latent.cpu(), metrics


@torch.no_grad()
def run_random_interpolations(generator, model_name, args, device, output):
    dimension = generator.z_dim if model_name == "stylegan2" else generator.latent_dim

    for pair_index in range(args.random_pairs):
        seed_a = args.seed + 20000 + pair_index * 2
        seed_b = seed_a + 1
        z_a = random_z(dimension, seed_a, device)
        z_b = random_z(dimension, seed_b, device)

        pair_dir = output / f"pair_{pair_index:02d}"

        export_path(
            generator,
            model_name,
            "z",
            spherical_path(z_a, z_b, args.frames),
            pair_dir / "z_slerp",
            f"{model_name}: random Z endpoints, spherical interpolation",
        )

        if model_name == "stylegan2":
            w_a = generator.mapping(z_a, update_w_avg=False)[:, 0, :]
            w_b = generator.mapping(z_b, update_w_avg=False)[:, 0, :]

            export_path(
                generator,
                model_name,
                "w",
                linear_path(w_a, w_b, args.frames),
                pair_dir / "w_lerp",
                "StyleGAN2: shared-W linear interpolation",
            )

        ev.atomic_write_json(
            pair_dir / "endpoints.json",
            {
                "seed_a": seed_a,
                "seed_b": seed_b,
                "frames": args.frames,
                "quality_filtering": False,
            },
        )


def run_model(args, model_name, checkpoint_path, rows, targets, device):
    saved = ev.read_json(args.cohort_dir / model_name / "generation_config.json")

    # Verify this is exactly the checkpoint used in Stage A.
    ev.verify_hash(
        checkpoint_path,
        saved["checkpoint_sha256"],
        f"{model_name} checkpoint",
    )

    checkpoint, generator = load_generator(checkpoint_path, device)

    if checkpoint["model_name"] != model_name:
        raise ValueError("Checkpoint model mismatch.")

    if checkpoint["epoch"] != saved["checkpoint_epoch"]:
        raise ValueError("Checkpoint epoch mismatch.")

    if model_name == "stylegan2":
        if not checkpoint["config"].get("ema_enabled", False):
            raise ValueError("StyleGAN2 EMA weights are required.")
        state_key = "generator_ema_state_dict"
    else:
        state_key = "generator_state_dict"

    if state_key != saved["weights_key"]:
        raise ValueError("Weight selection differs from Stage A.")

    generator.eval()
    generator.requires_grad_(False)

    del checkpoint
    gc.collect()

    output = args.output_dir / model_name
    output.mkdir(parents=True, exist_ok=True)

    ev.atomic_write_json(
        output / "model_info.json",
        {
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": saved["checkpoint_sha256"],
            "epoch": saved["checkpoint_epoch"],
            "weights_key": state_key,
            "generator_frozen": True,
            "noise_optimized": False,
            "noise_mode": ("const" if model_name == "stylegan2" else None),
        },
    )

    run_random_interpolations(
        generator,
        model_name,
        args,
        device,
        output / "random_interpolation",
    )

    center, scale = latent_statistics(generator, model_name, args, device)

    torch.save(
        {"center": center.cpu(), "scale": scale.cpu()},
        output / "latent_statistics.pt",
    )

    projected = {}
    metric_rows = []
    space = "w" if model_name == "stylegan2" else "z"
    dimension = generator.z_dim if model_name == "stylegan2" else generator.latent_dim

    for index, row in enumerate(rows):
        target_id = row["reference_id"]
        target = targets[target_id]

        latent, metrics = project_one(
            generator,
            model_name,
            row,
            target,
            center,
            scale,
            args,
            device,
            output / "projection" / target_id,
            index,
        )

        projected[target_id] = latent
        metric_rows.append(metrics)

        # Explicit requirement:
        # a synthetic image -> projection of a real training/validation image.
        z = random_z(
            dimension,
            args.seed + 30000 + index,
            device,
        )

        with torch.no_grad():
            start = (
                generator.mapping(z, update_w_avg=False)[:, 0, :]
                if model_name == "stylegan2"
                else z
            )

        target_pixels = target[0, 0].mul(255).round().to(torch.uint8).numpy()

        export_path(
            generator,
            model_name,
            space,
            linear_path(
                start,
                latent.to(device),
                args.frames,
            ),
            output / "synthetic_to_projection" / target_id,
            (
                f"{model_name}: synthetic -> {target_id} projection; "
                "last generated frame is a reconstruction"
            ),
            original_after=target_pixels,
        )

        write_csv(output / "projection_metrics.csv", metric_rows)

    # Additional real-to-real interpolation:
    # pair train and validation targets with the same diagnosis.
    # Diagnosis is metadata only, not a conditioning input.
    for diagnosis in DIAGNOSES:
        for index in range(args.per_class):
            first_id = f"train_{diagnosis}_{index:02d}"
            second_id = f"validation_{diagnosis}_{index:02d}"

            first_pixels = (
                targets[first_id][0, 0].mul(255).round().to(torch.uint8).numpy()
            )
            second_pixels = (
                targets[second_id][0, 0].mul(255).round().to(torch.uint8).numpy()
            )

            export_path(
                generator,
                model_name,
                space,
                linear_path(
                    projected[first_id].to(device),
                    projected[second_id].to(device),
                    args.frames,
                ),
                output / "projected_real_interpolation" / f"{diagnosis}_{index:02d}",
                (
                    f"{model_name}: two projected real endpoints; "
                    "not a clinical disease trajectory"
                ),
                original_before=first_pixels,
                original_after=second_pixels,
            )

    if any(parameter.grad is not None for parameter in generator.parameters()):
        raise RuntimeError("Generator parameters unexpectedly accumulated gradients.")

    del generator, center, scale
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return metric_rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Latent interpolation and small-cohort real-image projection."
    )
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dcgan-checkpoint", type=Path, required=True)
    parser.add_argument("--stylegan2-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--per-class", type=int, default=1)
    parser.add_argument("--random-pairs", type=int, default=3)
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--projection-lr", type=float, default=0.05)
    parser.add_argument("--gradient-weight", type=float, default=0.1)
    parser.add_argument("--prior-weight", type=float, default=0.001)
    parser.add_argument("--w-samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if (
        min(
            args.per_class,
            args.random_pairs,
            args.steps,
            args.restarts,
            args.w_samples,
        )
        < 1
        or args.frames < 2
    ):
        parser.error("Counts must be positive and frames must be at least 2.")

    if args.w_samples < 2:
        parser.error("w-samples must be at least 2.")

    numeric = (
        args.projection_lr,
        args.gradient_weight,
        args.prior_weight,
    )
    if not all(math.isfinite(value) for value in numeric):
        parser.error("Loss and learning-rate parameters must be finite.")

    if (
        args.projection_lr <= 0
        or args.gradient_weight < 0
        or args.prior_weight < 0
        or args.seed < 0
    ):
        parser.error("Invalid optimization parameters or seed.")

    for name in (
        "cohort_dir",
        "data_root",
        "dcgan_checkpoint",
        "stylegan2_checkpoint",
    ):
        setattr(args, name, getattr(args, name).resolve())

    args.output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else args.cohort_dir / "stage_e" / "latent_analysis"
    )

    return args


def main():
    args = parse_args()
    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # Avoid first-run cuDNN algorithm-search memory spikes.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    stage_a = ev.read_json(args.cohort_dir / "protocol.json")

    if stage_a.get("status") != "complete":
        raise ValueError("Stage A is incomplete.")

    project_dir = Path(__file__).resolve().parent

    for filename in ("dataset.py", "modules.py"):
        ev.verify_hash(
            project_dir / filename,
            stage_a["source_hashes"][filename],
            filename,
        )

    rows, targets = select_targets(args, stage_a)

    # Protect completed experiments and avoid accidentally replacing results.
    if args.output_dir.exists():
        raise FileExistsError(
            f"Output already exists: {args.output_dir}\n"
            "Use a new --output-dir for a new attempt."
        )

    args.output_dir.mkdir(parents=True)
    target_dir = args.output_dir / "targets"
    target_dir.mkdir()

    write_csv(args.output_dir / "target_manifest.csv", rows)

    for target_id, target in targets.items():
        array = target[0, 0].mul(255).round().to(torch.uint8).numpy()
        save_gray(target_dir / f"{target_id}.png", array)

    protocol = {
        "schema_version": 1,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "source_hashes": {
            name: ev.file_sha256(project_dir / name)
            for name in (
                Path(__file__).name,
                "evaluation.py",
                "predict.py",
                "modules.py",
                "dataset.py",
            )
        },
        "stage_a_protocol_sha256": ev.file_sha256(args.cohort_dir / "protocol.json"),
        "target_manifest_sha256": ev.file_sha256(
            args.output_dir / "target_manifest.csv"
        ),
        "target_count": len(rows),
        "selection": "fixed-seed distinct patients, then one slice per patient",
        "stylegan2_projection_space": "shared W, not W+",
        "dcgan_projection_space": "Z",
        "generator_frozen": True,
        "noise_optimized": False,
        "stylegan2_noise_mode": "const",
        "projection_loss": {
            "multiscale_weighted_l1": {
                "pool_factors": [1, 2, 4],
                "coefficients": [0.5, 0.3, 0.2],
                "target_foreground_threshold": 0.05,
                "background_weight": 1.0,
                "foreground_weight": 5.0,
            },
            "gradient_weight": args.gradient_weight,
            "standardized_latent_prior_weight": args.prior_weight,
            "clamp_during_optimization": False,
        },
        "metrics": "computed on exported uint8 reconstruction / 255",
        "environment": {
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "tf32": False,
            "cudnn_benchmark": False,
            "amp": False,
        },
        "limitations": [
            "Projection is approximate and depends on optimization settings.",
            "DCGAN Z and StyleGAN2 W have different dimensions and constraints.",
            "Reconstruction is not evidence of spontaneous training-image copying.",
            "Interpolation is not a clinical progression or diagnosis prediction.",
            "Only a small fixed cohort is projected; no population accuracy claim.",
        ],
    }
    ev.atomic_write_json(args.output_dir / "protocol.json", protocol)

    status = {
        "status": "running",
        "started_utc": ev.utc_now(),
        "completed_models": [],
    }
    status_path = args.output_dir / "status.json"
    ev.atomic_write_json(status_path, status)

    all_metrics = []
    started = time.perf_counter()

    try:
        paths = {
            "dcgan": args.dcgan_checkpoint,
            "stylegan2": args.stylegan2_checkpoint,
        }

        for model_name in MODELS:
            print(f"\nStarting {model_name}", flush=True)

            metrics = run_model(
                args,
                model_name,
                paths[model_name],
                rows,
                targets,
                device,
            )
            all_metrics.extend(metrics)
            write_csv(
                args.output_dir / "projection_metrics_all.csv",
                all_metrics,
            )

            status["completed_models"].append(model_name)
            ev.atomic_write_json(status_path, status)

        status["status"] = "complete"

    except Exception as error:
        status["status"] = "failed"
        status["error"] = f"{type(error).__name__}: {error}"
        raise

    finally:
        sync(device)
        status["finished_utc"] = ev.utc_now()
        status["elapsed_seconds"] = time.perf_counter() - started
        ev.atomic_write_json(status_path, status)

    print(f"\nStage E complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
