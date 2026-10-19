"""Export reproducible image cohorts for DCGAN / StyleGAN2 evaluation."""

import argparse
import csv
import gc
import hashlib
import json
import math
import random
import shutil
import statistics
import time

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from PIL import Image
from torchvision.utils import save_image

from predict import load_generator

RECORD_FIELDS = [
    "relative_path",
    "subject_id",
    "image_id",
    "slice_index",
    "diagnosis",
    "source_split",
]

SAMPLE_FIELDS = [
    "sample_id",
    "relative_path",
    "model_name",
    "latent_index",
    "latent_seed",
    "batch_index",
    "index_in_batch",
    "synthesis_noise_seed",
    "raw_min",
    "raw_max",
    "clipped_fraction",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def seed_synthesis(seed):
    # The model's random synthesis noise uses the global torch RNG.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reset_gpu_peak(device):
    if device.type == "cuda":
        synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def gpu_peak(device):
    if device.type != "cuda":
        return {
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }

    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def environment_info(device, args):
    return {
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "allow_tf32": args.allow_tf32,
        "cudnn_benchmark": args.cudnn_benchmark,
        "cudnn_deterministic": False,
        "precision": "float32",
        "amp_enabled": False,
    }


def load_reference_records(manifest_path, split):
    """Read an existing split manifest without creating new splits."""
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if payload.get("schema_version") != 1:
        raise ValueError("Expected split manifest schema_version=1.")

    items = payload.get("splits", {}).get(split)
    if not isinstance(items, list) or not items:
        raise ValueError(f"Missing or empty split: {split}")

    records = []
    seen_paths = set()

    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Invalid split record.")

        if any(field not in item for field in RECORD_FIELDS):
            raise ValueError(f"Record is missing required fields: {item}")

        record = {field: item[field] for field in RECORD_FIELDS}
        relative_path = record["relative_path"]

        if relative_path in seen_paths:
            raise ValueError(f"Duplicate reference path: {relative_path}")

        if record["diagnosis"] not in {"AD", "NC"}:
            raise ValueError(f"Unexpected diagnosis: {record['diagnosis']}")

        seen_paths.add(relative_path)
        records.append(record)

    return records


def stratified_subset(records, count, seed):
    """Sample without replacement, preserving AD/NC proportions."""
    if count > len(records):
        raise ValueError(
            f"Requested {count} real images, but split has {len(records)}."
        )

    groups = {}
    for record in records:
        groups.setdefault(record["diagnosis"], []).append(record)

    labels = sorted(groups)
    total = len(records)

    # Largest-remainder allocation ensures exactly 'count' records.
    quotas = {label: count * len(groups[label]) / total for label in labels}
    allocations = {label: math.floor(quotas[label]) for label in labels}

    remainder = count - sum(allocations.values())
    ranked = sorted(
        labels,
        key=lambda label: (
            -(quotas[label] - allocations[label]),
            label,
        ),
    )
    for label in ranked[:remainder]:
        allocations[label] += 1

    rng = random.Random(seed)
    selected = []

    for label in labels:
        candidates = sorted(
            groups[label],
            key=lambda record: record["relative_path"],
        )
        selected.extend(rng.sample(candidates, allocations[label]))

    # Stable order for future image loading and feature caching.
    selected.sort(key=lambda record: record["relative_path"])
    return selected


def prepare_real_reference(args, output_dir):
    records = load_reference_records(args.manifest, args.reference_split)

    selected = stratified_subset(
        records,
        count=args.real_count,
        seed=args.real_subset_seed,
    )

    data_root = args.data_root.resolve()

    for record in selected:
        path = (data_root / record["relative_path"]).resolve()

        if not path.is_relative_to(data_root):
            raise ValueError(f"Reference path escapes data_root: {path}")

        if not path.is_file():
            raise FileNotFoundError(path)

    rows = [
        {
            "reference_id": f"real_{index:06d}",
            **record,
        }
        for index, record in enumerate(selected)
    ]

    manifest_out = output_dir / "real_reference_manifest.csv"
    write_csv(
        manifest_out,
        rows,
        ["reference_id", *RECORD_FIELDS],
    )

    # Preserve the original full split manifest for future train-NN auditing.
    snapshot = output_dir / "source_splits.json"
    shutil.copyfile(args.manifest, snapshot)

    return {
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": file_sha256(snapshot),
        "source_manifest_snapshot": snapshot.name,
        "reference_manifest": manifest_out.name,
        "reference_manifest_sha256": file_sha256(manifest_out),
        "reference_split": args.reference_split,
        "source_split_count": len(records),
        "selected_count": len(selected),
        "class_counts": dict(Counter(record["diagnosis"] for record in selected)),
        "patient_count": len({record["subject_id"] for record in selected}),
        "data_root": str(data_root),
        "selection_seed": args.real_subset_seed,
        "selection": "proportional_AD_NC_without_replacement",
        "preprocessing": "ADNIDataset/preprocess_image",
        "real_images_copied": False,
    }


def make_latents(count, z_dim, base_seed):
    """Give every image an explicit latent seed and saved latent row."""
    rng = torch.Generator(device="cpu")
    latents = torch.empty(count, z_dim, dtype=torch.float32)

    for index in range(count):
        rng.manual_seed(base_seed + index)
        latents[index] = torch.randn(z_dim, generator=rng)

    return latents


def generate(generator, model_name, z, noise_mode):
    if model_name == "stylegan2":
        return generator(
            z,
            noise_mode=noise_mode,
            update_w_avg=False,
        )
    return generator(z)


def validate_images(images, expected_batch):
    expected = (expected_batch, 1, 256, 256)

    if tuple(images.shape) != expected:
        raise ValueError(
            f"Expected generated shape {expected}, " f"got {tuple(images.shape)}."
        )

    if not torch.isfinite(images).all().item():
        raise FloatingPointError("Generated images contain NaN or Inf.")


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@torch.no_grad()
def benchmark_forward(
    generator,
    model_name,
    z,
    device,
    noise_mode,
    warmup,
    repeats,
    seed,
):
    """Measure synchronized forward wall time with inputs already on device.

    Excludes latent generation, host-to-device transfer, image validation,
    quantization, and PNG writing.
    """
    seed_synthesis(seed)

    for _ in range(warmup):
        images = generate(generator, model_name, z, noise_mode)
        validate_images(images, len(z))
        del images

    synchronize(device)
    reset_gpu_peak(device)

    durations = []

    for _ in range(repeats):
        synchronize(device)
        started = time.perf_counter()

        images = generate(generator, model_name, z, noise_mode)

        synchronize(device)
        elapsed = time.perf_counter() - started
        durations.append(elapsed)

        # Outside the timed region.
        validate_images(images, len(z))
        del images

    total_images = len(z) * repeats
    total_seconds = sum(durations)

    return {
        "batch_size": len(z),
        "warmup_steps": warmup,
        "measured_steps": repeats,
        "total_forward_seconds": total_seconds,
        "images_per_second": total_images / total_seconds,
        "mean_batch_latency_ms": statistics.mean(durations) * 1000,
        "median_batch_latency_ms": statistics.median(durations) * 1000,
        "p95_batch_latency_ms": percentile(durations, 0.95) * 1000,
        "amortized_ms_per_image": total_seconds / total_images * 1000,
        "timing_scope": "synchronized_forward_inputs_already_on_device",
        **gpu_peak(device),
    }


def save_previews(image_dir, preview_dir, count):
    """Use predefined positions; do not select images by quality."""
    preview_dir.mkdir()

    available = min(count, 48)
    for start in range(0, available, 16):
        ids = list(range(start, min(start + 16, available)))
        tensors = []

        for index in ids:
            with Image.open(image_dir / f"{index:06d}.png") as image:
                pixels = np.array(image, dtype=np.uint8, copy=True)

            tensors.append(torch.from_numpy(pixels).unsqueeze(0).float() / 255.0)

        save_image(
            torch.stack(tensors),
            preview_dir / f"group_{start // 16 + 1:02d}.png",
            nrow=4,
            normalize=False,
        )

    write_json(
        preview_dir / "selection.json",
        {
            "policy": "first_up_to_48_samples_in_id_order",
            "sample_indices": list(range(available)),
            "quality_filtering": False,
        },
    )


@torch.no_grad()
def export_model(
    model_name,
    checkpoint_path,
    args,
    device,
    root,
):
    model_dir = root / model_name
    image_dir = model_dir / "images"
    image_dir.mkdir(parents=True)

    print(f"\nLoading {model_name}: {checkpoint_path}", flush=True)

    checkpoint, generator = load_generator(checkpoint_path, device)

    if checkpoint.get("model_name") != model_name:
        raise ValueError(f"Expected {model_name}, got {checkpoint.get('model_name')}.")

    epoch = checkpoint.get("epoch")
    if epoch != args.expected_epoch:
        raise ValueError(
            f"{model_name}: expected epoch {args.expected_epoch}, got {epoch}."
        )

    saved_config = checkpoint.get("config", {})

    if model_name == "stylegan2":
        if not saved_config.get("ema_enabled", False):
            raise ValueError("This evaluation requires StyleGAN2 EMA weights.")
        state_key = "generator_ema_state_dict"
        z_dim = generator.z_dim
    else:
        state_key = "generator_state_dict"
        z_dim = generator.latent_dim

    generator.eval()
    generator.requires_grad_(False)

    # The loader has copied the required weights into the generator.
    # Release the full training checkpoint, including optimizer states.
    del checkpoint
    gc.collect()

    latents = make_latents(
        args.num_images,
        z_dim,
        args.latent_seed,
    )
    torch.save(latents, model_dir / "latents.pt")

    generation_config = {
        "schema_version": 1,
        "model_name": model_name,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_epoch": epoch,
        "weights_key": state_key,
        "training_config": saved_config,
        "num_images": args.num_images,
        "z_dim": z_dim,
        "latent_distribution": "standard_normal",
        "latent_seed_policy": "latent_seed_base + sample_index",
        "latent_seed_base": args.latent_seed,
        "latent_file": "latents.pt",
        "generation_batch_size": args.batch_size,
        "noise_mode": (args.noise_mode if model_name == "stylegan2" else None),
        "synthesis_noise_seed_policy": (
            "noise_seed_base + batch_index" if model_name == "stylegan2" else None
        ),
        "noise_seed_base": (args.noise_seed if model_name == "stylegan2" else None),
        "style_mixing_enabled": False,
        "truncation_applied": False,
        "update_w_avg": False,
        "output_shape": [1, 256, 256],
        "output_format": "PNG_L_uint8",
        "output_conversion": "round(255 * clamp((x + 1) / 2, 0, 1))",
        "quality_filtering": False,
        "environment": environment_info(device, args),
        "started_utc": utc_now(),
    }
    write_json(model_dir / "generation_config.json", generation_config)

    # Benchmark batch inference and true batch-size-1 latency separately.
    z_benchmark = latents[: args.batch_size].to(device)

    batch_performance = benchmark_forward(
        generator,
        model_name,
        z_benchmark,
        device,
        args.noise_mode,
        args.warmup,
        args.benchmark_steps,
        seed=args.noise_seed,
    )
    del z_benchmark

    z_single = latents[:1].to(device)

    single_performance = benchmark_forward(
        generator,
        model_name,
        z_single,
        device,
        args.noise_mode,
        args.warmup,
        args.benchmark_steps,
        seed=args.noise_seed,
    )
    del z_single

    print(
        f"{model_name}: "
        f"{batch_performance['images_per_second']:.2f} images/s "
        f"at batch={args.batch_size}; "
        f"batch-1 mean latency="
        f"{single_performance['mean_batch_latency_ms']:.2f} ms",
        flush=True,
    )

    # Every export batch is reseeded, so benchmark RNG consumption
    # cannot change the exported images.
    synchronize(device)
    reset_gpu_peak(device)
    export_started = time.perf_counter()

    raw_minimum = math.inf
    raw_maximum = -math.inf
    clipped_pixels = 0
    total_pixels = 0
    exported = 0

    samples_path = model_dir / "samples.csv"

    with samples_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()

        for batch_index, start in enumerate(range(0, args.num_images, args.batch_size)):
            end = min(start + args.batch_size, args.num_images)
            batch_noise_seed = args.noise_seed + batch_index

            if model_name == "stylegan2":
                seed_synthesis(batch_noise_seed)

            z = latents[start:end].to(device)
            images = generate(
                generator,
                model_name,
                z,
                args.noise_mode,
            )
            validate_images(images, end - start)

            raw = images.detach().cpu()
            del images, z

            for offset, sample in enumerate(raw):
                index = start + offset
                minimum = float(sample.min().item())
                maximum = float(sample.max().item())

                outside = int(((sample < -1.0) | (sample > 1.0)).sum().item())
                pixel_count = sample.numel()

                raw_minimum = min(raw_minimum, minimum)
                raw_maximum = max(raw_maximum, maximum)
                clipped_pixels += outside
                total_pixels += pixel_count

                pixels = (
                    ((sample + 1.0) / 2.0)
                    .clamp(0.0, 1.0)
                    .mul(255.0)
                    .round()
                    .to(torch.uint8)
                    .squeeze(0)
                    .numpy()
                )

                filename = f"{index:06d}.png"
                Image.fromarray(pixels).save(image_dir / filename)

                writer.writerow(
                    {
                        "sample_id": f"{model_name}_{index:06d}",
                        "relative_path": f"images/{filename}",
                        "model_name": model_name,
                        "latent_index": index,
                        "latent_seed": args.latent_seed + index,
                        "batch_index": batch_index,
                        "index_in_batch": offset,
                        "synthesis_noise_seed": (
                            batch_noise_seed if model_name == "stylegan2" else ""
                        ),
                        "raw_min": minimum,
                        "raw_max": maximum,
                        "clipped_fraction": outside / pixel_count,
                    }
                )
                exported += 1

            del raw

            if batch_index % 50 == 0 or end == args.num_images:
                print(
                    f"{model_name}: exported {end}/{args.num_images}",
                    flush=True,
                )

    synchronize(device)
    export_seconds = time.perf_counter() - export_started
    export_memory = gpu_peak(device)

    if exported != args.num_images:
        raise RuntimeError("Exported image count does not match request.")

    actual_count = sum(1 for _ in image_dir.glob("*.png"))
    if actual_count != args.num_images:
        raise RuntimeError("PNG file count does not match request.")

    # Previews are excluded from the timed export interval.
    save_previews(image_dir, model_dir / "previews", args.num_images)

    summary = {
        "status": "complete",
        "model_name": model_name,
        "checkpoint_epoch": epoch,
        "weights_key": state_key,
        "num_images": exported,
        "generator_parameters": sum(
            parameter.numel() for parameter in generator.parameters()
        ),
        "batch_forward_benchmark": batch_performance,
        "single_image_forward_benchmark": single_performance,
        "export": {
            "seconds": export_seconds,
            "images_per_second": exported / export_seconds,
            "timing_scope": (
                "noise_seeding_latent_transfer_forward_validation_"
                "cpu_transfer_quantization_png_and_csv_writing"
            ),
            "excludes": (
                "checkpoint_loading_hashing_latent_creation_"
                "warmup_benchmarks_and_previews"
            ),
            **export_memory,
        },
        "raw_output": {
            "minimum": raw_minimum,
            "maximum": raw_maximum,
            "clipped_pixel_fraction": clipped_pixels / total_pixels,
        },
        "samples_manifest_sha256": file_sha256(samples_path),
        "latent_file_sha256": file_sha256(model_dir / "latents.pt"),
        "png_total_bytes": sum(path.stat().st_size for path in image_dir.glob("*.png")),
        "completed_utc": utc_now(),
    }
    write_json(model_dir / "generation_summary.json", summary)

    del generator, latents
    gc.collect()

    if device.type == "cuda":
        synchronize(device)
        torch.cuda.empty_cache()

    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export fixed DCGAN and StyleGAN2 evaluation cohorts."
    )
    parser.add_argument(
        "--dcgan-checkpoint",
        type=Path,
        default=Path("runs/dcgan_adni_20261006_115201/checkpoints/final.pt"),
    )
    parser.add_argument(
        "--stylegan2-checkpoint",
        type=Path,
        default=Path(
            "runs/stylegan2_adni_full_test_20261014_061632_618300Z/"
            "checkpoints/final.pt"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("splits/splits_ad_nc_seed42.json"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/groups/comp3710/ADNI/AD_NC"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-epoch", type=int, default=100)
    parser.add_argument(
        "--reference-split",
        choices=("validation", "test"),
        default="test",
    )
    parser.add_argument("--real-count", type=int, default=5000)
    parser.add_argument("--num-images", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--real-subset-seed", type=int, default=20261019)
    parser.add_argument("--latent-seed", type=int, default=10001)
    parser.add_argument("--noise-seed", type=int, default=20001)
    parser.add_argument(
        "--noise-mode",
        choices=("random", "const"),
        default="random",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--benchmark-steps", type=int, default=50)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cudnn-benchmark",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    args = parser.parse_args()

    for name in (
        "expected_epoch",
        "real_count",
        "num_images",
        "batch_size",
        "warmup",
        "benchmark_steps",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")

    if args.batch_size > args.num_images:
        parser.error("--batch-size cannot exceed --num-images.")

    for name in ("real_subset_seed", "latent_seed", "noise_seed"):
        if not 0 <= getattr(args, name) < 2**63:
            parser.error(f"{name} must be in [0, 2**63).")

    if args.latent_seed + args.num_images >= 2**63:
        parser.error("Per-image latent seeds would exceed the supported range.")

    if args.noise_seed + args.num_images >= 2**63:
        parser.error("Per-batch noise seeds would exceed the supported range.")

    return args


def main():
    args = parse_args()

    # Validate source paths before creating an output directory.
    for path in (
        args.dcgan_checkpoint,
        args.stylegan2_checkpoint,
        args.manifest,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    if not args.data_root.is_dir():
        raise NotADirectoryError(args.data_root)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    if args.output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {args.output_dir}\n"
            "Use a new directory; this exporter does not overwrite or resume."
        )

    device = torch.device(args.device)

    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    torch.backends.cudnn.deterministic = False

    args.output_dir.mkdir(parents=True)

    protocol = {
        "schema_version": 1,
        "status": "incomplete",
        "started_utc": utc_now(),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "environment": environment_info(device, args),
        "real_reference": None,
        "models_completed": [],
        "source_hashes": {
            name: file_sha256(Path(__file__).resolve().parent / name)
            for name in (
                "export_evaluation.py",
                "predict.py",
                "modules.py",
                "dataset.py",
            )
        },
    }

    protocol_path = args.output_dir / "protocol.json"
    write_json(protocol_path, protocol)

    try:
        protocol["real_reference"] = prepare_real_reference(args, args.output_dir)
        write_json(protocol_path, protocol)

        summaries = {}

        for model_name, checkpoint_path in (
            ("dcgan", args.dcgan_checkpoint),
            ("stylegan2", args.stylegan2_checkpoint),
        ):
            summaries[model_name] = export_model(
                model_name,
                checkpoint_path,
                args,
                device,
                args.output_dir,
            )
            protocol["models_completed"].append(model_name)
            write_json(protocol_path, protocol)

        write_json(args.output_dir / "summary.json", summaries)

        protocol["status"] = "complete"
        protocol["completed_utc"] = utc_now()
        write_json(protocol_path, protocol)

    except Exception as error:
        protocol["status"] = "failed"
        protocol["error"] = f"{type(error).__name__}: {error}"
        write_json(protocol_path, protocol)
        raise

    print(f"\nExport complete: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
