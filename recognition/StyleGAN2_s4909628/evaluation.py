import argparse
import csv
import hashlib
import json
import os

from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import random

import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader

from dataset import preprocess_image

CACHE_SCHEMA_VERSION = 1

EXTRACTOR_NAME = "inception-v3-compat"
FEATURE_LAYER = "2048"
FEATURE_DIM = 2048

REAL_FIELDS = (
    "reference_id",
    "relative_path",
    "subject_id",
    "image_id",
    "slice_index",
    "diagnosis",
    "source_split",
)

FAKE_FIELDS = (
    "sample_id",
    "relative_path",
    "model_name",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def object_sha256(value):
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")

    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")

        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_torch_save(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")

    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def verify_hash(path, expected_hash, description):
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError(f"Missing or invalid SHA256 for {description}.")

    actual_hash = file_sha256(path)

    if actual_hash != expected_hash:
        raise ValueError(
            f"{description} SHA256 mismatch.\n"
            f"Expected: {expected_hash}\n"
            f"Actual:   {actual_hash}\n"
            f"File:     {path}"
        )

    return actual_hash


def resolve_image_path(root, relative_path):
    root = Path(root).resolve()

    if not relative_path:
        raise ValueError("Empty relative image path.")

    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError(f"Expected a relative path: {relative_path}")

    path = (root / relative).resolve()

    if not path.is_relative_to(root):
        raise ValueError(f"Image path escapes its root directory: {path}")

    if not path.is_file():
        raise FileNotFoundError(path)

    return path


def read_manifest(path, required_fields, id_field):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")

        missing = set(required_fields) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty manifest: {path}")

    seen_ids = set()
    seen_paths = set()

    for row in rows:
        sample_id = row[id_field]
        relative_path = row["relative_path"]

        if not sample_id or sample_id in seen_ids:
            raise ValueError(f"Empty or duplicate ID in {path}: {sample_id}")

        if not relative_path or relative_path in seen_paths:
            raise ValueError(
                f"Empty or duplicate image path in {path}: {relative_path}"
            )

        seen_ids.add(sample_id)
        seen_paths.add(relative_path)

    return rows


def load_evaluation_inputs(cohort_dir, data_root):
    """Validate Stage A metadata and preserve CSV row order."""
    cohort_dir = Path(cohort_dir).resolve()
    protocol = read_json(cohort_dir / "protocol.json")

    if protocol.get("schema_version") != 1:
        raise ValueError("Unsupported Stage A protocol schema.")

    if protocol.get("status") != "complete":
        raise ValueError("Stage A protocol is not complete.")

    completed = set(protocol.get("models_completed", []))
    if not {"dcgan", "stylegan2"}.issubset(completed):
        raise ValueError("Stage A did not complete both models.")

    reference = protocol["real_reference"]

    real_manifest = cohort_dir / "real_reference_manifest.csv"
    real_manifest_hash = verify_hash(
        real_manifest,
        reference["reference_manifest_sha256"],
        "real reference manifest",
    )

    real_rows = read_manifest(
        real_manifest,
        required_fields=REAL_FIELDS,
        id_field="reference_id",
    )

    if len(real_rows) != reference["selected_count"]:
        raise ValueError("Real reference count does not match protocol.")

    for row in real_rows:
        if row["diagnosis"] not in {"AD", "NC"}:
            raise ValueError(f"Unexpected diagnosis: {row['diagnosis']}")

        expected_source = "test" if reference["reference_split"] == "test" else "train"
        if row["source_split"] != expected_source:
            raise ValueError(
                f"Unexpected source split in real manifest: {row['source_split']}"
            )

    specs = {
        "real": {
            "root": Path(data_root).resolve(),
            "rows": real_rows,
            "id_field": "reference_id",
            "manifest_sha256": real_manifest_hash,
            "kind": "real",
        }
    }

    for model_name in ("dcgan", "stylegan2"):
        model_dir = cohort_dir / model_name
        summary = read_json(model_dir / "generation_summary.json")
        generation_config = read_json(model_dir / "generation_config.json")

        if summary.get("status") != "complete":
            raise ValueError(f"{model_name} export is incomplete.")

        if summary.get("model_name") != model_name:
            raise ValueError(f"{model_name} summary has inconsistent model name.")

        if generation_config.get("model_name") != model_name:
            raise ValueError(f"{model_name} generation config is inconsistent.")

        manifest_path = model_dir / "samples.csv"

        manifest_hash = verify_hash(
            manifest_path,
            summary["samples_manifest_sha256"],
            f"{model_name} sample manifest",
        )

        rows = read_manifest(
            manifest_path,
            required_fields=FAKE_FIELDS,
            id_field="sample_id",
        )

        if len(rows) != summary["num_images"]:
            raise ValueError(f"{model_name} sample count does not match summary.")

        if len(rows) != generation_config["num_images"]:
            raise ValueError(f"{model_name} sample count does not match config.")

        for row in rows:
            if row["model_name"] != model_name:
                raise ValueError(f"Wrong model_name in {manifest_path}")

        if generation_config.get("output_shape") != [1, 256, 256]:
            raise ValueError(f"Unexpected {model_name} output shape.")

        if generation_config.get("output_format") != "PNG_L_uint8":
            raise ValueError(f"Unexpected {model_name} output format.")

        specs[model_name] = {
            "root": model_dir,
            "rows": rows,
            "id_field": "sample_id",
            "manifest_sha256": manifest_hash,
            "kind": "generated",
        }

    return protocol, specs


class EvaluationImageDataset(Dataset):
    """Return (uint8 grayscale image, stable row index).

    Real images:
        Reuse the training preprocessing, then map back to uint8.

    Generated images:
        Read the exported grayscale PNG without further padding or resizing.
    """

    def __init__(self, name, spec):
        self.name = name
        self.kind = spec["kind"]
        self.root = spec["root"]
        self.rows = spec["rows"]
        self.id_field = spec["id_field"]
        self.manifest_sha256 = spec["manifest_sha256"]

        self.sample_ids = [row[self.id_field] for row in self.rows]
        self.relative_paths = [row["relative_path"] for row in self.rows]

        self.paths = [
            resolve_image_path(self.root, relative_path)
            for relative_path in self.relative_paths
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path = self.paths[index]

        with Image.open(path) as image:
            if self.kind == "real":
                image.load()
                tensor = preprocess_image(image)

                if tensor.shape != (1, 256, 256):
                    raise ValueError(
                        f"Unexpected real image shape at {path}: "
                        f"{tuple(tensor.shape)}"
                    )

                if not torch.isfinite(tensor).all().item():
                    raise FloatingPointError(f"Non-finite real image: {path}")

                pixels = (
                    ((tensor + 1.0) / 2.0)
                    .clamp(0.0, 1.0)
                    .mul(255.0)
                    .round()
                    .to(torch.uint8)
                )

            else:
                if image.format != "PNG":
                    raise ValueError(f"Expected PNG: {path}")

                if image.mode != "L":
                    raise ValueError(
                        f"Expected grayscale mode L, got {image.mode}: {path}"
                    )

                if image.size != (256, 256):
                    raise ValueError(f"Expected 256x256, got {image.size}: {path}")

                pixels = torch.from_numpy(
                    np.array(image, dtype=np.uint8, copy=True)
                ).unsqueeze(0)

        if pixels.dtype != torch.uint8:
            raise TypeError(f"Expected uint8 pixels: {path}")

        if pixels.shape != (1, 256, 256):
            raise ValueError(f"Unexpected image shape: {path}")

        return pixels, index

    def content_sha256(self):
        """Fingerprint image bytes together with their ordered identities."""
        digest = hashlib.sha256()

        for index, (sample_id, relative_path, path) in enumerate(
            zip(self.sample_ids, self.relative_paths, self.paths)
        ):
            entry = {
                "sample_id": sample_id,
                "relative_path": relative_path,
                "file_sha256": file_sha256(path),
            }

            digest.update(
                json.dumps(
                    entry,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            digest.update(b"\n")

            if (index + 1) % 1000 == 0 or index + 1 == len(self):
                print(
                    f"{self.name}: fingerprinted {index + 1}/{len(self)} images",
                    flush=True,
                )

        return digest.hexdigest()


def build_feature_extractor(weights_path, device):
    # Import only when extraction is required. Cached features need no model.
    from torch_fidelity.feature_extractor_inceptionv3 import (
        FeatureExtractorInceptionV3,
    )

    extractor = FeatureExtractorInceptionV3(
        name=EXTRACTOR_NAME,
        features_list=[FEATURE_LAYER],
        feature_extractor_weights_path=str(weights_path),
    ).to(device)

    extractor.eval()
    extractor.requires_grad_(False)

    return extractor


def make_extraction_config(args, device):
    script_dir = Path(__file__).resolve().parent

    return {
        "extractor": EXTRACTOR_NAME,
        "feature_layer": FEATURE_LAYER,
        "feature_dim": FEATURE_DIM,
        "weights_sha256": file_sha256(args.inception_weights),
        "input_dtype": "uint8",
        "input_range": [0, 255],
        "input_shape": [3, 256, 256],
        "grayscale_to_rgb": "repeat_same_channel_three_times",
        "extractor_resize": "handled_internally_by_torch_fidelity",
        "real_preprocessing": ("dataset.preprocess_image_then_round_uint8_fixed_range"),
        "generated_preprocessing": "read_exported_L_uint8_PNG_unchanged",
        "feature_normalization": "none",
        "feature_precision": "float32",
        "amp_enabled": False,
        "allow_tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "batch_size": args.batch_size,
        "device_type": device.type,
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "versions": {
            "torch": str(torch.__version__),
            "torchvision": version("torchvision"),
            "torch-fidelity": version("torch-fidelity"),
            "numpy": version("numpy"),
            "Pillow": version("Pillow"),
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "dataset_source_sha256": file_sha256(script_dir / "dataset.py"),
        "evaluation_source_sha256": file_sha256(Path(__file__)),
    }


def validate_feature_tensor(features, expected_count):
    if not isinstance(features, torch.Tensor):
        raise TypeError("Cached features must be a tensor.")

    if features.shape != (expected_count, FEATURE_DIM):
        raise ValueError(
            f"Expected features [{expected_count}, {FEATURE_DIM}], "
            f"got {tuple(features.shape)}."
        )

    if features.dtype != torch.float32:
        raise TypeError(f"Expected float32 features, got {features.dtype}.")

    if not torch.isfinite(features).all().item():
        raise FloatingPointError("Features contain NaN or Inf.")


def load_valid_cache(cache_path, dataset, fingerprint):
    payload = torch.load(
        cache_path,
        map_location="cpu",
        weights_only=True,
    )

    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported cache schema: {cache_path}")

    if payload.get("fingerprint") != fingerprint:
        raise ValueError(
            f"Cache does not match current data or extraction settings:\n"
            f"{cache_path}\n"
            "Use a new --cache-dir to rebuild; existing caches are not overwritten."
        )

    if payload.get("sample_ids") != dataset.sample_ids:
        raise ValueError(f"Cached sample order differs: {cache_path}")

    if payload.get("relative_paths") != dataset.relative_paths:
        raise ValueError(f"Cached image paths differ: {cache_path}")

    validate_feature_tensor(payload["features"], len(dataset))
    return payload


@torch.no_grad()
def extract_features(dataset, extractor, args, device):
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )

    features = torch.empty(
        len(dataset),
        FEATURE_DIM,
        dtype=torch.float32,
        device="cpu",
    )
    written = torch.zeros(len(dataset), dtype=torch.bool)

    for batch_number, (gray, indices) in enumerate(loader):
        indices = indices.to(dtype=torch.long, device="cpu")

        if indices.unique().numel() != indices.numel():
            raise ValueError("Duplicate row indices within a batch.")

        if written[indices].any().item():
            raise ValueError("A feature row would be written twice.")

        rgb = gray.repeat(1, 3, 1, 1).to(
            device,
            non_blocking=(device.type == "cuda"),
        )

        batch_features = extractor(rgb)[0]

        if batch_features.shape != (len(indices), FEATURE_DIM):
            raise ValueError(
                f"Unexpected extractor output: {tuple(batch_features.shape)}"
            )

        if not torch.isfinite(batch_features).all().item():
            raise FloatingPointError(
                f"Non-finite features in {dataset.name}, batch {batch_number}."
            )

        features[indices] = batch_features.detach().cpu().float()
        written[indices] = True

        if batch_number % 20 == 0 or written.all().item():
            print(
                f"{dataset.name}: extracted "
                f"{int(written.sum().item())}/{len(dataset)}",
                flush=True,
            )

    if not written.all().item():
        raise RuntimeError(f"Missing feature rows for {dataset.name}.")

    validate_feature_tensor(features, len(dataset))
    return features


def extract_or_load_features(
    dataset,
    cache_path,
    extraction_config,
    get_extractor,
    args,
    device,
):
    content_hash = dataset.content_sha256()

    identity = {
        "dataset_name": dataset.name,
        "sample_count": len(dataset),
        "manifest_sha256": dataset.manifest_sha256,
        "image_content_sha256": content_hash,
        "extraction_config": extraction_config,
    }
    fingerprint = object_sha256(identity)

    if cache_path.exists():
        payload = load_valid_cache(cache_path, dataset, fingerprint)
        print(f"{dataset.name}: loaded cache {cache_path}", flush=True)
        return payload, True

    features = extract_features(
        dataset,
        get_extractor(),
        args,
        device,
    )

    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "dataset_name": dataset.name,
        "created_utc": utc_now(),
        "fingerprint": fingerprint,
        "identity": identity,
        "features": features,
        "sample_ids": dataset.sample_ids,
        "relative_paths": dataset.relative_paths,
        # Preserve metadata for reference matching and later visualisation.
        "records": dataset.rows,
    }

    atomic_torch_save(cache_path, payload)

    print(f"{dataset.name}: saved cache {cache_path}", flush=True)
    return payload, False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage B1: cache unified Inception features."
    )

    parser.add_argument(
        "--cohort-dir",
        type=Path,
        default=Path("evaluation/cohort_final_674672"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/groups/comp3710/ADNI/AD_NC"),
    )
    parser.add_argument(
        "--inception-weights",
        type=Path,
        required=True,
        help="Official torch-fidelity InceptionV3 weights file.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Default: <cohort-dir>/stage_b/features",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")

    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative.")

    if args.cache_dir is None:
        args.cache_dir = args.cohort_dir / "stage_b" / "features"

    return args


def main():
    args = parse_args()

    if not args.cohort_dir.is_dir():
        raise NotADirectoryError(args.cohort_dir)

    if not args.data_root.is_dir():
        raise NotADirectoryError(args.data_root)

    if not args.inception_weights.is_file():
        raise FileNotFoundError(args.inception_weights)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    device = torch.device(args.device)

    # Evaluation uses FP32 feature extraction with TF32 disabled.
    # This does not change the already-exported Stage A images.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    protocol, specs = load_evaluation_inputs(
        args.cohort_dir,
        args.data_root,
    )

    datasets = {
        name: EvaluationImageDataset(name, spec) for name, spec in specs.items()
    }

    extraction_config = make_extraction_config(args, device)

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    extractor = None

    def get_extractor():
        nonlocal extractor

        if extractor is None:
            print("Loading frozen Inception feature extractor...", flush=True)
            extractor = build_feature_extractor(
                args.inception_weights,
                device,
            )

        return extractor

    summary = {
        "schema_version": 1,
        "status": "incomplete",
        "started_utc": utc_now(),
        "cohort_dir": str(args.cohort_dir.resolve()),
        "data_root": str(args.data_root.resolve()),
        "reference_split": protocol["real_reference"]["reference_split"],
        "extraction_config": extraction_config,
        "datasets": {},
    }

    summary_path = args.cache_dir / "feature_summary.json"

    try:
        for name in ("real", "dcgan", "stylegan2"):
            dataset = datasets[name]
            cache_path = args.cache_dir / f"{name}_inception2048.pt"

            payload, cache_hit = extract_or_load_features(
                dataset=dataset,
                cache_path=cache_path,
                extraction_config=extraction_config,
                get_extractor=get_extractor,
                args=args,
                device=device,
            )

            features = payload["features"]

            summary["datasets"][name] = {
                "cache_file": cache_path.name,
                "cache_hit": cache_hit,
                "fingerprint": payload["fingerprint"],
                "shape": list(features.shape),
                "dtype": str(features.dtype),
                "finite": True,
                "feature_minimum": float(features.min().item()),
                "feature_maximum": float(features.max().item()),
                "feature_mean": float(features.mean().item()),
                "first_sample_id": payload["sample_ids"][0],
                "last_sample_id": payload["sample_ids"][-1],
            }

            atomic_write_json(summary_path, summary)

            print(
                f"{name}: shape={tuple(features.shape)}, "
                f"dtype={features.dtype}, finite=True",
                flush=True,
            )

            del features, payload

        summary["status"] = "complete"
        summary["completed_utc"] = utc_now()
        atomic_write_json(summary_path, summary)

    except Exception as error:
        summary["status"] = "failed"
        summary["error"] = f"{type(error).__name__}: {error}"
        atomic_write_json(summary_path, summary)
        raise

    print(
        f"\nFeature caching complete: {args.cache_dir.resolve()}",
        flush=True,
    )


def load_fid_feature_caches(cache_dir, expected_count):
    """Load and validate the three completed feature caches.

    Validation uses the provenance saved when features were extracted.
    It does not compare the saved extraction-script hash against the
    current evaluation.py, because FID code may have been added later.
    """
    cache_dir = Path(cache_dir)
    summary_path = cache_dir / "feature_summary.json"

    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)

    summary = read_json(summary_path)

    if summary.get("schema_version") != 1:
        raise ValueError("Unsupported feature summary schema.")

    if summary.get("status") != "complete":
        raise ValueError("Feature extraction did not complete successfully.")

    extraction_config = summary.get("extraction_config")

    if not isinstance(extraction_config, dict):
        raise ValueError("Missing extraction configuration.")

    if extraction_config.get("extractor") != "inception-v3-compat":
        raise ValueError("Expected inception-v3-compat features.")

    if extraction_config.get("feature_layer") != "2048":
        raise ValueError("Expected Inception feature layer 2048.")

    if extraction_config.get("feature_dim") != 2048:
        raise ValueError("Expected 2048-dimensional features.")

    if extraction_config.get("feature_normalization") != "none":
        raise ValueError("FID must use the original, unnormalised Inception features.")

    features_by_name = {}
    provenance = {}

    for name in ("real", "dcgan", "stylegan2"):
        cache_path = cache_dir / f"{name}_inception2048.pt"

        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)

        entry = summary.get("datasets", {}).get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"Missing feature summary entry for {name}.")

        if entry.get("cache_file") != cache_path.name:
            raise ValueError(f"Unexpected cache filename for {name}.")

        payload = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=True,
        )

        if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported feature cache schema: {cache_path}")

        if payload.get("dataset_name") != name:
            raise ValueError(f"Dataset name mismatch: {cache_path}")

        identity = payload.get("identity")
        if not isinstance(identity, dict):
            raise ValueError(f"Missing cache identity: {cache_path}")

        fingerprint = payload.get("fingerprint")

        # Check consistency of saved metadata, without recreating features.
        if object_sha256(identity) != fingerprint:
            raise ValueError(f"Cache identity/fingerprint mismatch: {cache_path}")

        if entry.get("fingerprint") != fingerprint:
            raise ValueError(f"Cache/summary fingerprint mismatch: {cache_path}")

        if identity.get("dataset_name") != name:
            raise ValueError(f"Identity dataset name mismatch: {cache_path}")

        if identity.get("extraction_config") != extraction_config:
            raise ValueError(f"{name} was extracted using different feature settings.")

        sample_ids = payload.get("sample_ids")
        relative_paths = payload.get("relative_paths")
        records = payload.get("records")

        if not isinstance(sample_ids, list) or len(sample_ids) != expected_count:
            raise ValueError(f"{name}: expected {expected_count} sample IDs.")

        if (
            not all(isinstance(item, str) and item for item in sample_ids)
            or len(set(sample_ids)) != expected_count
        ):
            raise ValueError(f"{name}: invalid or duplicate sample IDs.")

        if (
            not isinstance(relative_paths, list)
            or len(relative_paths) != expected_count
            or not all(isinstance(item, str) and item for item in relative_paths)
            or len(set(relative_paths)) != expected_count
        ):
            raise ValueError(f"{name}: invalid or duplicate image paths.")

        if not isinstance(records, list) or len(records) != expected_count:
            raise ValueError(f"{name}: invalid record count.")

        id_field = "reference_id" if name == "real" else "sample_id"

        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"{name}: invalid record at index {index}.")

            if record.get(id_field) != sample_ids[index]:
                raise ValueError(f"{name}: sample order mismatch at index {index}.")

            if record.get("relative_path") != relative_paths[index]:
                raise ValueError(f"{name}: path order mismatch at index {index}.")

        if identity.get("sample_count") != expected_count:
            raise ValueError(f"{name}: identity sample count mismatch.")

        features = payload.get("features")
        validate_feature_tensor(features, expected_count)

        if entry.get("shape") != list(features.shape):
            raise ValueError(f"{name}: feature shape disagrees with summary.")

        if entry.get("dtype") != str(features.dtype):
            raise ValueError(f"{name}: feature dtype disagrees with summary.")

        features_by_name[name] = features

        provenance[name] = {
            "cache_path": str(cache_path.resolve()),
            "cache_sha256": file_sha256(cache_path),
            "fingerprint": fingerprint,
            "sample_count": expected_count,
            "feature_shape": list(features.shape),
            "manifest_sha256": identity.get("manifest_sha256"),
            "image_content_sha256": identity.get("image_content_sha256"),
        }

        print(
            f"Loaded {name}: "
            f"shape={tuple(features.shape)}, "
            f"dtype={features.dtype}, finite=True",
            flush=True,
        )

        del payload

    return (
        features_by_name,
        provenance,
        extraction_config,
        summary.get("reference_split"),
    )


def compute_fid_statistics(features):
    """Calculate mean/covariance using all samples in float64 on CPU."""
    from torch_fidelity.metric_fid import fid_features_to_statistics

    features64 = features.detach().to(
        device="cpu",
        dtype=torch.float64,
    )

    statistics = fid_features_to_statistics(features64)

    mean = statistics["mu"]
    covariance = statistics["sigma"]

    if mean.shape != (2048,):
        raise ValueError(f"Unexpected mean shape: {mean.shape}")

    if covariance.shape != (2048, 2048):
        raise ValueError(f"Unexpected covariance shape: {covariance.shape}")

    if mean.dtype != np.float64 or covariance.dtype != np.float64:
        raise TypeError("FID statistics must use float64.")

    if not np.isfinite(mean).all():
        raise FloatingPointError("FID mean contains NaN or Inf.")

    if not np.isfinite(covariance).all():
        raise FloatingPointError("FID covariance contains NaN or Inf.")

    if not np.allclose(
        covariance,
        covariance.T,
        rtol=1e-10,
        atol=1e-10,
    ):
        raise ValueError("FID covariance is not symmetric.")

    return statistics


def compute_fid_value(real_statistics, fake_statistics):
    """Use the installed torch-fidelity implementation."""
    import math

    from torch_fidelity.metric_fid import fid_statistics_to_metric

    result = fid_statistics_to_metric(
        real_statistics,
        fake_statistics,
        verbose=False,
    )

    value = float(result["frechet_inception_distance"])

    if not math.isfinite(value):
        raise FloatingPointError(f"FID is not finite: {value}")

    return value


def atomic_write_fid_csv(path, rows):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")

    fields = [
        "model",
        "reference",
        "real_count",
        "generated_count",
        "feature_dim",
        "fid",
        "fid_raw",
        "numerical_zero_clamped",
    ]

    try:
        with temporary.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_fid_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Calculate FID from the three existing feature caches."
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("evaluation/cohort_final_674672/stage_b/features"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <cache-dir parent>/fid",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--self-check-atol",
        type=float,
        default=1e-3,
        help="Absolute tolerance for the Real-vs-Real numerical check.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=4,
    )

    args = parser.parse_args(argv)

    if args.expected_count < 2:
        parser.error("--expected-count must be at least 2.")

    if not np.isfinite(args.self_check_atol) or args.self_check_atol <= 0:
        parser.error("--self-check-atol must be finite and positive.")

    if args.torch_threads < 1:
        parser.error("--torch-threads must be positive.")

    if args.output_dir is None:
        args.output_dir = args.cache_dir.parent / "fid"

    return args


def fid_main(argv=None):
    import time

    args = parse_fid_args(argv)

    if not args.cache_dir.is_dir():
        raise NotADirectoryError(args.cache_dir)

    json_path = args.output_dir / "fid_results.json"
    csv_path = args.output_dir / "fid_results.csv"

    # Preserve previous evaluation results.
    for path in (json_path, csv_path):
        if path.exists():
            raise FileExistsError(
                f"Result already exists: {path}\n"
                "Use a different --output-dir for another evaluation."
            )

    torch.set_num_threads(args.torch_threads)
    started = time.perf_counter()

    (
        features_by_name,
        provenance,
        extraction_config,
        reference_split,
    ) = load_fid_feature_caches(
        args.cache_dir,
        args.expected_count,
    )

    # Keep the metric implementation consistent with the recorded environment.
    installed_fidelity_version = version("torch-fidelity")
    extraction_fidelity_version = extraction_config.get("versions", {}).get(
        "torch-fidelity"
    )

    if installed_fidelity_version != extraction_fidelity_version:
        raise ValueError(
            "torch-fidelity version differs from the feature-extraction run.\n"
            f"Extraction: {extraction_fidelity_version}\n"
            f"Current:    {installed_fidelity_version}\n"
            "Use the recorded version to keep this evaluation consistent."
        )

    print("\nComputing Real statistics in float64...", flush=True)
    real_statistics = compute_fid_statistics(features_by_name["real"])

    # Use the actual FID computation; do not hard-code the identity result.
    print("Checking Real vs Real...", flush=True)
    self_fid = compute_fid_value(real_statistics, real_statistics)

    print(f"Real vs Real FID: {self_fid:.10f}", flush=True)

    if abs(self_fid) > args.self_check_atol:
        raise RuntimeError(
            f"Real-vs-Real check failed: FID={self_fid:.10f}, "
            f"tolerance={args.self_check_atol}.\n"
            "Inspect the metric implementation and numerical behaviour "
            "before reporting model FID values."
        )

    rows = []

    for name in ("dcgan", "stylegan2"):
        print(f"\nComputing {name} statistics in float64...", flush=True)

        fake_statistics = compute_fid_statistics(features_by_name[name])

        print(f"Computing Real vs {name} FID...", flush=True)
        raw_fid = compute_fid_value(real_statistics, fake_statistics)

        # A substantially negative FID is invalid.
        # Preserve tiny negative round-off in fid_raw, while reporting zero.
        if raw_fid < -args.self_check_atol:
            raise FloatingPointError(f"{name}: unexpectedly negative FID={raw_fid}.")

        numerical_zero_clamped = raw_fid < 0
        reported_fid = max(raw_fid, 0.0)

        rows.append(
            {
                "model": name,
                "reference": "real",
                "real_count": args.expected_count,
                "generated_count": args.expected_count,
                "feature_dim": 2048,
                "fid": reported_fid,
                "fid_raw": raw_fid,
                "numerical_zero_clamped": numerical_zero_clamped,
            }
        )

        print(
            f"{name}: FID={reported_fid:.6f} " f"(raw={raw_fid:.10f})",
            flush=True,
        )

        del fake_statistics

    report = {
        "schema_version": 1,
        "status": "complete",
        "completed_utc": utc_now(),
        "reference_split": reference_split,
        "metric": "Frechet Inception Distance",
        "implementation": (
            "torch_fidelity.metric_fid."
            "fid_features_to_statistics + fid_statistics_to_metric"
        ),
        "implementation_versions": {
            "torch-fidelity": installed_fidelity_version,
            "torch": str(torch.__version__),
            "numpy": version("numpy"),
            "scipy": version("scipy"),
        },
        "statistics_dtype": "float64",
        "statistics_device": "cpu",
        "covariance_ddof": 1,
        "feature_normalization": "none",
        "pca_applied": False,
        "batchwise_fid_averaging": False,
        "self_check": {
            "comparison": "real_vs_real",
            "fid_raw": self_fid,
            "absolute_tolerance": args.self_check_atol,
            "passed": True,
        },
        "results": rows,
        "extraction_config": extraction_config,
        "cache_provenance": provenance,
        "evaluation_source_sha256": file_sha256(Path(__file__)),
        "torch_threads": args.torch_threads,
        "elapsed_seconds": time.perf_counter() - started,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    atomic_write_fid_csv(csv_path, rows)
    atomic_write_json(json_path, report)

    print("\nFID evaluation complete.", flush=True)
    print(f"JSON: {json_path.resolve()}", flush=True)
    print(f"CSV:  {csv_path.resolve()}", flush=True)


def load_ssim_datasets(args, provenance, extraction_config):
    """Match current images/manifests to the already-cached features."""
    protocol, specs = load_evaluation_inputs(
        args.cohort_dir,
        args.data_root,
    )

    # The actual real-image preprocessing must still match extraction.
    dataset_source = Path(__file__).resolve().parent / "dataset.py"
    expected_source_hash = extraction_config.get("dataset_source_sha256")

    if file_sha256(dataset_source) != expected_source_hash:
        raise ValueError(
            "dataset.py differs from the version used for feature extraction.\n"
            "Check whether preprocessing changed before calculating SSIM."
        )

    datasets = {}

    for name in ("real", "dcgan", "stylegan2"):
        dataset = EvaluationImageDataset(name, specs[name])

        if len(dataset) != args.expected_count:
            raise ValueError(
                f"{name}: expected {args.expected_count} images, "
                f"got {len(dataset)}."
            )

        if dataset.manifest_sha256 != provenance[name]["manifest_sha256"]:
            raise ValueError(
                f"{name}: current manifest differs from the cached manifest."
            )

        # Check row identity explicitly: NN indices must resolve to the
        # same images used when extracting each feature row.
        cache_path = args.cache_dir / f"{name}_inception2048.pt"
        payload = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=True,
        )

        if payload["sample_ids"] != dataset.sample_ids:
            raise ValueError(f"{name}: sample order differs from cache.")

        if payload["relative_paths"] != dataset.relative_paths:
            raise ValueError(f"{name}: image path order differs from cache.")

        del payload

        # Validate actual image bytes without rerunning Inception.
        current_content_hash = dataset.content_sha256()

        if current_content_hash != provenance[name]["image_content_sha256"]:
            raise ValueError(
                f"{name}: image contents changed after feature extraction."
            )

        datasets[name] = dataset

    return protocol, datasets


@torch.no_grad()
def find_reference_neighbors(
    query_features,
    reference_features,
    device,
    chunk_size=128,
):
    """Find the nearest reference in the original Inception feature space.

    Uses float64 distances and bounded query chunks.
    No PCA, feature normalisation, or diagnosis-based filtering.
    Ties resolve to the first reference index.
    """
    if query_features.ndim != 2 or reference_features.ndim != 2:
        raise ValueError("Features must be two-dimensional.")

    if query_features.shape[1] != reference_features.shape[1]:
        raise ValueError("Feature dimensions do not match.")

    reference = reference_features.to(
        device=device,
        dtype=torch.float64,
    )

    neighbor_indices = []
    neighbor_distances = []

    for start in range(0, len(query_features), chunk_size):
        query = query_features[start : start + chunk_size].to(
            device=device,
            dtype=torch.float64,
        )

        distances = torch.cdist(
            query,
            reference,
            p=2,
            compute_mode="use_mm_for_euclid_dist",
        )

        if not torch.isfinite(distances).all().item():
            raise FloatingPointError("Non-finite nearest-neighbor distances.")

        _, indices = distances.min(dim=1)

        # Recompute the selected distances directly to avoid small
        # cancellation errors in the matrix-based distance formula.
        selected_distances = torch.linalg.vector_norm(
            query - reference[indices],
            ord=2,
            dim=1,
        )

        neighbor_indices.append(indices.cpu())
        neighbor_distances.append(selected_distances.cpu())

        print(
            f"Nearest neighbors: "
            f"{min(start + chunk_size, len(query_features))}/"
            f"{len(query_features)}",
            flush=True,
        )

        del distances, query

    indices = torch.cat(neighbor_indices)
    distances = torch.cat(neighbor_distances)

    # Independently verify a few queries using direct vector differences.
    for query_index in range(min(3, len(query_features))):
        query = query_features[query_index].to(
            device=device,
            dtype=torch.float64,
        )

        direct_distances = torch.linalg.vector_norm(
            reference - query,
            ord=2,
            dim=1,
        )

        true_minimum = direct_distances.min()
        chosen_distance = direct_distances[int(indices[query_index])]

        if not torch.isclose(
            chosen_distance,
            true_minimum,
            rtol=1e-8,
            atol=1e-8,
        ).item():
            raise RuntimeError(
                f"Nearest-neighbor validation failed for query {query_index}."
            )

    return indices, distances


def load_ssim_image(dataset, index):
    """Load the exact evaluation image as a 2D float64 array in [0, 1]."""
    pixels, returned_index = dataset[index]

    if returned_index != index:
        raise ValueError("Dataset returned an unexpected row index.")

    image = pixels.squeeze(0).numpy().astype(np.float64) / 255.0

    if image.shape != (256, 256):
        raise ValueError(f"Unexpected SSIM image shape: {image.shape}")

    if not np.isfinite(image).all():
        raise FloatingPointError("SSIM image contains NaN or Inf.")

    if image.min() < 0.0 or image.max() > 1.0:
        raise ValueError("SSIM image is outside [0, 1].")

    return image


def calculate_image_ssim(reference_image, generated_image):
    from skimage.metrics import structural_similarity

    value = structural_similarity(
        reference_image,
        generated_image,
        data_range=1.0,
        channel_axis=None,
        win_size=11,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
    )

    value = float(value)

    if not np.isfinite(value):
        raise FloatingPointError("SSIM is NaN or Inf.")

    # Negative SSIM can be valid; do not clamp it to zero.
    if value < -1.0 - 1e-8 or value > 1.0 + 1e-8:
        raise ValueError(f"Unexpected SSIM value: {value}")

    return value


def check_ssim_identity(datasets):
    """A numerical sanity check, not a quality score."""
    results = {}

    for name, dataset in datasets.items():
        image = load_ssim_image(dataset, 0)
        score = calculate_image_ssim(image, image)

        if not np.isclose(score, 1.0, rtol=0.0, atol=1e-10):
            raise RuntimeError(f"SSIM identity check failed for {name}: {score}")

        results[name] = score
        print(f"{name}: SSIM(image, image)={score:.10f}", flush=True)

    return results


def compute_reference_ssim(
    model_name,
    fake_dataset,
    real_dataset,
    neighbor_indices,
    neighbor_distances,
):
    from functools import lru_cache

    # Repeatedly selected references need not be decoded every time.
    @lru_cache(maxsize=128)
    def get_real_image(index):
        return load_ssim_image(real_dataset, index)

    rows = []

    for query_index in range(len(neighbor_indices)):
        reference_index = int(neighbor_indices[query_index].item())

        generated = load_ssim_image(fake_dataset, query_index)
        reference = get_real_image(reference_index)

        score = calculate_image_ssim(reference, generated)

        real_record = real_dataset.rows[reference_index]

        rows.append(
            {
                "model_name": model_name,
                "query_index": query_index,
                "sample_id": fake_dataset.sample_ids[query_index],
                "generated_relative_path": (fake_dataset.relative_paths[query_index]),
                "reference_index": reference_index,
                "reference_id": real_dataset.sample_ids[reference_index],
                "reference_relative_path": (
                    real_dataset.relative_paths[reference_index]
                ),
                "reference_subject_id": real_record["subject_id"],
                "reference_image_id": real_record["image_id"],
                "reference_slice_index": real_record["slice_index"],
                # This is the reference label, not a generated-image diagnosis.
                "reference_diagnosis": real_record["diagnosis"],
                "feature_l2_distance": float(neighbor_distances[query_index].item()),
                "ssim": score,
            }
        )

        if (query_index + 1) % 250 == 0 or (query_index + 1 == len(neighbor_indices)):
            print(
                f"{model_name}: SSIM " f"{query_index + 1}/{len(neighbor_indices)}",
                flush=True,
            )

    get_real_image.cache_clear()
    return rows


def summarize_reference_ssim(rows, reference_count):
    from collections import Counter

    scores = np.asarray(
        [row["ssim"] for row in rows],
        dtype=np.float64,
    )
    distances = np.asarray(
        [row["feature_l2_distance"] for row in rows],
        dtype=np.float64,
    )

    neighbor_counts = Counter(row["reference_index"] for row in rows)

    return {
        "query_count": len(rows),
        "reference_count": reference_count,
        "ssim": {
            "mean": float(scores.mean()),
            "std": float(scores.std(ddof=1)),
            "median": float(np.median(scores)),
            "p05": float(np.quantile(scores, 0.05)),
            "p95": float(np.quantile(scores, 0.95)),
            "minimum": float(scores.min()),
            "maximum": float(scores.max()),
            "fraction_gt_0_6": float((scores > 0.6).mean()),
        },
        "feature_l2_distance": {
            "mean": float(distances.mean()),
            "median": float(np.median(distances)),
            "p05": float(np.quantile(distances, 0.05)),
            "p95": float(np.quantile(distances, 0.95)),
        },
        "reference_selection": {
            "unique_references_selected": len(neighbor_counts),
            "fraction_of_reference_library_selected": (
                len(neighbor_counts) / reference_count
            ),
            "maximum_queries_selecting_one_reference": max(neighbor_counts.values()),
            "note": (
                "Reference selection frequency is descriptive; "
                "it is not a standalone mode-collapse metric."
            ),
        },
    }


def atomic_write_ssim_csv(path, rows):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")

    if not rows:
        raise ValueError("Cannot save an empty SSIM result.")

    try:
        with temporary.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(rows)

        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_ssim_pair_examples(
    rows,
    fake_dataset,
    real_dataset,
    output_dir,
    seed,
):
    """Save random examples and explicitly labelled low/high SSIM examples."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    example_count = min(4, len(rows))
    rng = np.random.default_rng(seed)

    random_indices = rng.choice(
        len(rows),
        size=example_count,
        replace=False,
    ).tolist()

    ordered = sorted(
        range(len(rows)),
        key=lambda index: (rows[index]["ssim"], index),
    )

    selections = {
        "random": random_indices,
        "lowest_ssim": ordered[:example_count],
        "highest_ssim": list(reversed(ordered[-example_count:])),
    }

    selection_record = {}

    for group_name, row_indices in selections.items():
        fig, axes = plt.subplots(
            len(row_indices),
            3,
            figsize=(10, 3.1 * len(row_indices)),
            squeeze=False,
        )

        selection_record[group_name] = []

        for position, row_index in enumerate(row_indices):
            row = rows[row_index]

            generated = load_ssim_image(
                fake_dataset,
                row["query_index"],
            )
            reference = load_ssim_image(
                real_dataset,
                row["reference_index"],
            )

            difference = np.abs(generated - reference)

            axes[position, 0].imshow(
                generated,
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            axes[position, 0].set_title(row["sample_id"], fontsize=9)

            axes[position, 1].imshow(
                reference,
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            axes[position, 1].set_title(
                f"{row['reference_id']} "
                f"(reference: {row['reference_diagnosis']})\n"
                f"SSIM={row['ssim']:.4f}, "
                f"L2={row['feature_l2_distance']:.3f}",
                fontsize=9,
            )

            axes[position, 2].imshow(
                difference,
                cmap="magma",
                vmin=0,
                vmax=1,
            )
            axes[position, 2].set_title(
                "Absolute pixel difference [0, 1]",
                fontsize=9,
            )

            for axis in axes[position]:
                axis.axis("off")

            selection_record[group_name].append(
                {
                    "sample_id": row["sample_id"],
                    "reference_id": row["reference_id"],
                    "ssim": row["ssim"],
                }
            )

        fig.suptitle(
            f"{fake_dataset.name}: {group_name}",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(output_dir / f"{group_name}.png", dpi=150)
        plt.close(fig)

    atomic_write_json(
        output_dir / "selection.json",
        {
            "random_seed": seed,
            "selection": selection_record,
            "difference_image_note": (
                "Pixel differences between distinct images; "
                "not a ground-truth anatomical error map."
            ),
        },
    )


def save_ssim_distribution(rows_by_model, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-1.0, 1.0, 81)

    for name, rows in rows_by_model.items():
        scores = [row["ssim"] for row in rows]

        axis.hist(
            scores,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            label=name,
        )

    axis.axvline(
        0.6,
        color="black",
        linestyle="--",
        linewidth=1,
        label="SSIM = 0.6",
    )
    axis.set_xlabel("Feature-nearest-reference SSIM")
    axis.set_ylabel("Density")
    axis.set_title("Full-image reference SSIM")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_ssim_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Match generated images to real reference neighbors "
            "and compute full-image SSIM."
        )
    )

    parser.add_argument(
        "--cohort-dir",
        type=Path,
        default=Path("evaluation/cohort_final_674672"),
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/groups/comp3710/ADNI/AD_NC"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--expected-count", type=int, default=5000)
    parser.add_argument(
        "--query-limit",
        type=int,
        default=None,
        help=(
            "Debug only: evaluate the first N generated images per model. "
            "The entire real reference library is still used."
        ),
    )
    parser.add_argument("--nn-chunk-size", type=int, default=128)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device for feature nearest-neighbor search; SSIM runs on CPU.",
    )
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--example-seed", type=int, default=2026)

    args = parser.parse_args(argv)

    if args.expected_count < 2:
        parser.error("--expected-count must be at least 2.")

    if args.nn_chunk_size < 1 or args.torch_threads < 1:
        parser.error("Chunk size and thread count must be positive.")

    if args.query_limit is not None:
        if not 2 <= args.query_limit <= args.expected_count:
            parser.error("--query-limit must be in [2, expected-count].")

    if args.example_seed < 0:
        parser.error("--example-seed must be non-negative.")

    if args.cache_dir is None:
        args.cache_dir = args.cohort_dir / "stage_b" / "features"

    if args.output_dir is None:
        name = "ssim" if args.query_limit is None else f"ssim_debug_{args.query_limit}"
        args.output_dir = args.cohort_dir / "stage_b" / name

    return args


def ssim_main(argv=None):
    import time

    args = parse_ssim_args(argv)

    if args.output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {args.output_dir}\n"
            "Choose another --output-dir; existing results are not overwritten."
        )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    device = torch.device(args.device)

    torch.set_num_threads(args.torch_threads)
    torch.backends.cuda.matmul.allow_tf32 = False

    (
        features_by_name,
        provenance,
        extraction_config,
        reference_split,
    ) = load_fid_feature_caches(
        args.cache_dir,
        args.expected_count,
    )

    protocol, datasets = load_ssim_datasets(
        args,
        provenance,
        extraction_config,
    )

    if protocol["real_reference"]["reference_split"] != reference_split:
        raise ValueError("Reference split differs from the feature summary.")

    identity_checks = check_ssim_identity(datasets)

    query_count = args.expected_count if args.query_limit is None else args.query_limit

    args.output_dir.mkdir(parents=True)
    report_path = args.output_dir / "ssim_results.json"

    report = {
        "schema_version": 1,
        "status": "incomplete",
        "started_utc": utc_now(),
        "debug_run": args.query_limit is not None,
        "reference_split": reference_split,
        "reference_count": args.expected_count,
        "query_count_per_model": query_count,
        "pairing": {
            "method": "nearest_reference_in_original_Inception_features",
            "distance": "euclidean",
            "distance_dtype": "float64",
            "feature_normalization": "none",
            "pca_applied": False,
            "diagnosis_filtering": False,
            "reference_reuse_allowed": True,
            "tie_break": "first_reference_index",
            "nn_chunk_size": args.nn_chunk_size,
            "nn_device": str(device),
        },
        "ssim_settings": {
            "image_shape": [256, 256],
            "image_range": [0.0, 1.0],
            "dtype": "float64",
            "data_range": 1.0,
            "channel_axis": None,
            "win_size": 11,
            "gaussian_weights": True,
            "sigma": 1.5,
            "use_sample_covariance": False,
            "region": "full_image_including_background",
            "registration_applied": False,
        },
        "identity_checks": identity_checks,
        "cache_provenance": provenance,
        "extraction_config": extraction_config,
        "versions": {
            "torch": str(torch.__version__),
            "numpy": version("numpy"),
            "scikit-image": version("scikit-image"),
            "Pillow": version("Pillow"),
        },
        "evaluation_source_sha256": file_sha256(Path(__file__)),
        "results": {},
    }
    atomic_write_json(report_path, report)

    started = time.perf_counter()
    rows_by_model = {}

    try:
        for name in ("dcgan", "stylegan2"):
            print(f"\nFinding {name} reference neighbors...", flush=True)

            indices, distances = find_reference_neighbors(
                features_by_name[name][:query_count],
                features_by_name["real"],
                device=device,
                chunk_size=args.nn_chunk_size,
            )

            print(f"Computing {name} SSIM...", flush=True)

            rows = compute_reference_ssim(
                model_name=name,
                fake_dataset=datasets[name],
                real_dataset=datasets["real"],
                neighbor_indices=indices,
                neighbor_distances=distances,
            )

            rows_by_model[name] = rows

            atomic_write_ssim_csv(
                args.output_dir / f"{name}_reference_pairs.csv",
                rows,
            )

            model_summary = summarize_reference_ssim(
                rows,
                reference_count=len(datasets["real"]),
            )
            report["results"][name] = model_summary
            atomic_write_json(report_path, report)

            save_ssim_pair_examples(
                rows,
                fake_dataset=datasets[name],
                real_dataset=datasets["real"],
                output_dir=args.output_dir / "figures" / name,
                seed=args.example_seed,
            )

            values = model_summary["ssim"]
            print(
                f"{name}: SSIM mean={values['mean']:.4f}, "
                f"std={values['std']:.4f}, "
                f"median={values['median']:.4f}, "
                f"fraction > 0.6={values['fraction_gt_0_6']:.2%}",
                flush=True,
            )

        comparison_rows = []

        for name, result in report["results"].items():
            comparison_rows.append(
                {
                    "model_name": name,
                    "query_count": result["query_count"],
                    "reference_count": result["reference_count"],
                    **result["ssim"],
                }
            )

        atomic_write_ssim_csv(
            args.output_dir / "ssim_summary.csv",
            comparison_rows,
        )

        save_ssim_distribution(
            rows_by_model,
            args.output_dir / "figures" / "ssim_distribution.png",
        )

        report["status"] = "complete"
        report["completed_utc"] = utc_now()
        report["elapsed_seconds"] = time.perf_counter() - started
        atomic_write_json(report_path, report)

    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        atomic_write_json(report_path, report)
        raise

    print(
        f"\nReference SSIM evaluation complete: " f"{args.output_dir.resolve()}",
        flush=True,
    )


SHARPNESS_KEYS = (
    "laplacian_full",
    "laplacian_interior",
    "tenengrad_full",
    "tenengrad_interior",
)


def make_sharpness_roi(image, settings):
    """Construct an approximate foreground interior, not a tissue label."""
    from scipy import ndimage as ndi

    threshold_mask = image > settings["foreground_threshold"]

    labels, component_count = ndi.label(
        threshold_mask,
        structure=np.ones((3, 3), dtype=np.uint8),
    )

    foreground = np.zeros_like(threshold_mask, dtype=bool)
    largest_component_fraction = 0.0

    if component_count:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        largest_label = int(sizes.argmax())

        foreground = labels == largest_label
        largest_component_fraction = int(foreground.sum()) / int(threshold_mask.sum())

        # Include enclosed dark structures in the foreground envelope.
        foreground = ndi.binary_fill_holes(foreground)

    # Padding explicitly treats pixels outside the image as background.
    padded = np.pad(
        foreground,
        pad_width=1,
        mode="constant",
        constant_values=False,
    )
    distance = ndi.distance_transform_edt(padded)[1:-1, 1:-1]

    interior = foreground & (distance > settings["erosion_pixels"])

    foreground_fraction = float(foreground.mean())
    interior_pixels = int(interior.sum())

    reasons = []

    if foreground_fraction < settings["minimum_foreground_fraction"]:
        reasons.append("foreground_too_small")

    if foreground_fraction > settings["maximum_foreground_fraction"]:
        reasons.append("foreground_too_large")

    if interior_pixels < settings["minimum_interior_pixels"]:
        reasons.append("insufficient_interior_pixels")

    information = {
        "roi_valid": not reasons,
        "roi_failure_reason": ";".join(reasons),
        "foreground_fraction": foreground_fraction,
        "interior_fraction": float(interior.mean()),
        "interior_pixels": interior_pixels,
        "threshold_component_count": int(component_count),
        "largest_component_fraction": float(largest_component_fraction),
    }

    return interior, information


def measure_sharpness(image, interior, roi_valid):
    """Compute derivative responses before applying the evaluation region."""
    from scipy import ndimage as ndi

    image = np.asarray(image, dtype=np.float64)

    if image.shape != (256, 256):
        raise ValueError(f"Unexpected image shape: {image.shape}")

    if not np.isfinite(image).all():
        raise FloatingPointError("Image contains NaN or Inf.")

    if image.min() < 0.0 or image.max() > 1.0:
        raise ValueError("Expected image values in [0, 1].")

    # Approximate 2D Laplacian: four-neighbor second derivative.
    laplacian = ndi.laplace(image, mode="reflect")

    # Normalised 3x3 Sobel responses.
    gradient_x = ndi.sobel(image, axis=1, mode="reflect") / 8.0
    gradient_y = ndi.sobel(image, axis=0, mode="reflect") / 8.0
    gradient_energy = gradient_x**2 + gradient_y**2

    values = {
        "laplacian_full": float(np.var(laplacian, ddof=0)),
        "laplacian_interior": None,
        "tenengrad_full": float(np.mean(gradient_energy)),
        "tenengrad_interior": None,
    }

    if roi_valid:
        values["laplacian_interior"] = float(np.var(laplacian[interior], ddof=0))
        values["tenengrad_interior"] = float(np.mean(gradient_energy[interior]))

    for key, value in values.items():
        if value is not None and (not np.isfinite(value) or value < 0.0):
            raise FloatingPointError(f"Invalid {key}: {value}")

    return values


def sharpness_implementation_hash():
    """Fingerprint the functions that define the measurement."""
    import inspect

    return object_sha256(
        {
            "roi_function": inspect.getsource(make_sharpness_roi),
            "metric_function": inspect.getsource(measure_sharpness),
        }
    )


def sharpness_versions():
    return {
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "Pillow": version("Pillow"),
        "torch": str(torch.__version__),
    }


def sharpness_numeric_summary(values):
    values = np.asarray(
        [value for value in values if value is not None],
        dtype=np.float64,
    )

    if len(values) == 0:
        return {"count": 0}

    return {
        "count": len(values),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else None,
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def summarize_sharpness_rows(rows):
    return {
        "image_count": len(rows),
        "valid_roi_count": sum(bool(row["roi_valid"]) for row in rows),
        "invalid_roi_count": sum(not row["roi_valid"] for row in rows),
        "metrics": {
            key: sharpness_numeric_summary([row[key] for row in rows])
            for key in SHARPNESS_KEYS
        },
        "foreground_fraction": sharpness_numeric_summary(
            [row["foreground_fraction"] for row in rows]
        ),
        "interior_fraction": sharpness_numeric_summary(
            [row["interior_fraction"] for row in rows]
        ),
    }


def sharpness_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_sharpness_roi_examples(examples, path):
    """Show the image, approximate interior, and an overlay."""
    if not examples:
        return

    plt = sharpness_pyplot()

    fig, axes = plt.subplots(
        len(examples),
        3,
        figsize=(10, 3 * len(examples)),
        squeeze=False,
    )

    for row_index, example in enumerate(examples):
        image = example["image"]
        interior = example["interior"]
        information = example["information"]

        axes[row_index, 0].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 0].set_title(example["sample_id"], fontsize=9)

        axes[row_index, 1].imshow(interior, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 1].set_title(
            f"Interior: {information['interior_fraction']:.1%}\n"
            f"Valid heuristic ROI: {information['roi_valid']}",
            fontsize=9,
        )

        axes[row_index, 2].imshow(image, cmap="gray", vmin=0, vmax=1)

        overlay = np.zeros((*image.shape, 4), dtype=np.float64)
        overlay[interior] = [1.0, 0.0, 0.0, 0.3]
        axes[row_index, 2].imshow(overlay)

        axes[row_index, 2].set_title(
            information["roi_failure_reason"] or "Interior overlay",
            fontsize=9,
        )

        for axis in axes[row_index]:
            axis.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_sharpness_distributions(rows_by_group, path):
    """Plot every observed value using empirical cumulative distributions."""
    plt = sharpness_pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for axis, key in zip(axes.ravel(), SHARPNESS_KEYS):
        for name, rows in rows_by_group.items():
            values = np.asarray(
                [row[key] for row in rows if row[key] is not None],
                dtype=np.float64,
            )

            if not len(values):
                continue

            values.sort()
            cumulative = np.arange(1, len(values) + 1) / len(values)

            axis.plot(
                values,
                cumulative,
                label=f"{name} (n={len(values)})",
                linewidth=1.5,
            )

        axis.set_title(key)
        axis.set_xlabel("Score (symlog scale)")
        axis.set_ylabel("Cumulative proportion")
        axis.set_xscale("symlog", linthresh=1e-8)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_sharpness_validation_dataset(args):
    """Select one validation slice per sampled patient."""
    from collections import defaultdict

    stage_a = read_json(args.cohort_dir / "protocol.json")

    if stage_a.get("status") != "complete":
        raise ValueError("Stage A is not complete.")

    snapshot = args.cohort_dir / "source_splits.json"

    verify_hash(
        snapshot,
        stage_a["real_reference"]["source_manifest_sha256"],
        "source split snapshot",
    )

    split_payload = read_json(snapshot)
    records = split_payload["splits"]["validation"]

    groups = defaultdict(list)
    for record in records:
        groups[record["subject_id"]].append(record)

    if args.calibration_count > len(groups):
        raise ValueError(
            f"Requested {args.calibration_count} validation patients, "
            f"but only {len(groups)} are available."
        )

    rng = random.Random(args.seed)
    patients = rng.sample(sorted(groups), args.calibration_count)

    selected = []
    for index, patient in enumerate(patients):
        candidates = sorted(
            groups[patient],
            key=lambda record: record["relative_path"],
        )
        record = dict(rng.choice(candidates))
        record["reference_id"] = f"validation_{index:04d}"
        selected.append(record)

    spec = {
        "root": args.data_root.resolve(),
        "rows": selected,
        "id_field": "reference_id",
        "manifest_sha256": object_sha256(selected),
        "kind": "real",
    }

    return EvaluationImageDataset("validation", spec)


def run_sharpness_calibration(args):
    from scipy import ndimage as ndi

    settings = {
        "foreground_threshold": args.foreground_threshold,
        "erosion_pixels": args.erosion_pixels,
        "minimum_foreground_fraction": 0.01,
        "maximum_foreground_fraction": 0.80,
        "minimum_interior_pixels": 128,
        "foreground_connectivity": 8,
        "keep_largest_component": True,
        "fill_enclosed_holes": True,
        "erosion_method": "euclidean_distance_greater_than_radius",
        "laplacian": "scipy_ndimage_laplace_2d",
        "laplacian_variance_ddof": 0,
        "tenengrad": "mean((sobel_x/8)^2 + (sobel_y/8)^2)",
        "derivative_boundary_mode": "reflect",
        "image_range": [0.0, 1.0],
        "image_dtype": "float64",
        "image_shape": [256, 256],
        "derivatives_before_masking": True,
    }

    dataset = load_sharpness_validation_dataset(args)

    # This is recorded once, before any perturbation is applied.
    content_hash = dataset.content_sha256()

    conditions = [
        "original",
        "blur_sigma_0.5",
        "blur_sigma_1.0",
        "blur_sigma_2.0",
        "noise_std_0.01",
        "noise_std_0.03",
    ]

    rows = []
    examples = []
    perturbation_examples = []

    for index in range(len(dataset)):
        image = load_ssim_image(dataset, index)
        interior, information = make_sharpness_roi(image, settings)

        rng = np.random.default_rng(args.seed + index)
        standard_noise = rng.standard_normal(image.shape)

        variants = {"original": image}
        clipping = {"original": 0.0}

        for sigma in (0.5, 1.0, 2.0):
            name = f"blur_sigma_{sigma}"
            variants[name] = ndi.gaussian_filter(
                image,
                sigma=sigma,
                mode="reflect",
                truncate=4.0,
            )
            clipping[name] = 0.0

        for std in (0.01, 0.03):
            name = f"noise_std_{std}"
            noisy = image + std * standard_noise

            clipping[name] = float(((noisy < 0.0) | (noisy > 1.0)).mean())
            variants[name] = np.clip(noisy, 0.0, 1.0)

        # Every variant uses the ORIGINAL image's identical interior mask.
        for condition in conditions:
            row = {
                "sample_id": dataset.sample_ids[index],
                "relative_path": dataset.relative_paths[index],
                "subject_id": dataset.rows[index]["subject_id"],
                "condition": condition,
                "noise_seed": args.seed + index,
                "perturbation_clipped_fraction": clipping[condition],
                **information,
                **measure_sharpness(
                    variants[condition],
                    interior,
                    information["roi_valid"],
                ),
            }
            rows.append(row)

        if index < 4:
            examples.append(
                {
                    "sample_id": dataset.sample_ids[index],
                    "image": image,
                    "interior": interior,
                    "information": information,
                }
            )
            perturbation_examples.append(variants)

        print(
            f"Calibration: {index + 1}/{len(dataset)}",
            flush=True,
        )

    atomic_write_ssim_csv(
        args.output_dir / "calibration_pairs.csv",
        rows,
    )
    atomic_write_ssim_csv(
        args.output_dir / "validation_manifest.csv",
        dataset.rows,
    )

    save_sharpness_roi_examples(
        examples,
        args.output_dir / "validation_roi_examples.png",
    )

    # Fixed examples: original, increasing blur, and added noise.
    plt = sharpness_pyplot()
    fig, axes = plt.subplots(
        len(perturbation_examples),
        len(conditions),
        figsize=(18, 3 * len(perturbation_examples)),
        squeeze=False,
    )

    for index, variants in enumerate(perturbation_examples):
        for column, condition in enumerate(conditions):
            axes[index, column].imshow(
                variants[condition],
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            axes[index, column].set_title(condition, fontsize=8)
            axes[index, column].axis("off")

    fig.tight_layout()
    fig.savefig(args.output_dir / "perturbation_examples.png", dpi=150)
    plt.close(fig)

    grouped = {
        condition: [row for row in rows if row["condition"] == condition]
        for condition in conditions
    }

    save_sharpness_distributions(
        grouped,
        args.output_dir / "calibration_distributions.png",
    )

    # Summarise matched per-image score changes relative to the original.
    baseline = {row["sample_id"]: row for row in grouped["original"]}

    paired_changes = {}

    for condition in conditions[1:]:
        paired_changes[condition] = {}

        for metric in SHARPNESS_KEYS:
            deltas = []

            for row in grouped[condition]:
                original = baseline[row["sample_id"]][metric]
                changed = row[metric]

                if original is not None and changed is not None:
                    deltas.append(changed - original)

            paired_changes[condition][metric] = {
                "delta_summary": sharpness_numeric_summary(deltas),
                "fraction_lower_than_original": (
                    float(np.mean(np.asarray(deltas) < 0.0)) if deltas else None
                ),
            }

    report = {
        "schema_version": 1,
        "status": "complete",
        "phase": "validation_calibration",
        "settings": settings,
        "implementation_sha256": sharpness_implementation_hash(),
        "dataset_source_sha256": file_sha256(
            Path(__file__).resolve().parent / "dataset.py"
        ),
        "versions": sharpness_versions(),
        "sample_count": len(dataset),
        "patient_sampling": "one_random_slice_per_sampled_validation_patient",
        "seed": args.seed,
        "validation_image_content_sha256": content_hash,
        "source_manifest_sha256": file_sha256(args.cohort_dir / "source_splits.json"),
        "mask_policy": "original_image_mask_shared_by_all_perturbations",
        "results": {
            condition: summarize_sharpness_rows(group_rows)
            for condition, group_rows in grouped.items()
        },
        "paired_changes": paired_changes,
        "completed_utc": utc_now(),
    }

    atomic_write_json(
        args.output_dir / "sharpness_protocol.json",
        report,
    )

    original_summary = report["results"]["original"]
    print(
        "Calibration complete. "
        f"Valid interiors: {original_summary['valid_roi_count']}/"
        f"{len(dataset)}",
        flush=True,
    )


def load_frozen_sharpness_protocol(args):
    if args.protocol is None:
        raise ValueError("--protocol is required for cohort evaluation.")

    protocol = read_json(args.protocol)

    if (
        protocol.get("schema_version") != 1
        or protocol.get("status") != "complete"
        or protocol.get("phase") != "validation_calibration"
    ):
        raise ValueError("Invalid or incomplete sharpness protocol.")

    if protocol["implementation_sha256"] != sharpness_implementation_hash():
        raise ValueError("Sharpness/ROI functions changed after calibration.")

    if protocol["versions"] != sharpness_versions():
        raise ValueError("Measurement dependencies differ from calibration.")

    current_dataset_hash = file_sha256(Path(__file__).resolve().parent / "dataset.py")

    if protocol["dataset_source_sha256"] != current_dataset_hash:
        raise ValueError("dataset.py changed after calibration.")

    if protocol["source_manifest_sha256"] != file_sha256(
        args.cohort_dir / "source_splits.json"
    ):
        raise ValueError("Split snapshot differs from calibration.")

    return protocol


def run_sharpness_cohort(args):
    frozen = load_frozen_sharpness_protocol(args)
    settings = frozen["settings"]

    # Reuse previous consistency checks, then release the unneeded features.
    features, provenance, extraction_config, reference_split = load_fid_feature_caches(
        args.cache_dir,
        args.expected_count,
    )
    del features

    _, datasets = load_ssim_datasets(
        args,
        provenance,
        extraction_config,
    )

    results = {}
    rows_by_group = {}
    invalid_rows = []

    for name in ("real", "dcgan", "stylegan2"):
        dataset = datasets[name]
        rows = []
        random_examples = []
        failed_examples = []

        rng = np.random.default_rng(args.seed)
        example_indices = set(
            rng.choice(len(dataset), size=min(4, len(dataset)), replace=False).tolist()
        )

        for index in range(len(dataset)):
            image = load_ssim_image(dataset, index)
            interior, information = make_sharpness_roi(image, settings)

            values = measure_sharpness(
                image,
                interior,
                information["roi_valid"],
            )

            row = {
                "group": name,
                "sample_id": dataset.sample_ids[index],
                "relative_path": dataset.relative_paths[index],
                **information,
                **values,
            }
            rows.append(row)

            example = {
                "sample_id": dataset.sample_ids[index],
                "image": image,
                "interior": interior,
                "information": information,
            }

            if index in example_indices:
                random_examples.append(example)

            if not information["roi_valid"]:
                invalid_rows.append(row)

                if len(failed_examples) < 4:
                    failed_examples.append(example)

            if (index + 1) % 500 == 0 or index + 1 == len(dataset):
                print(
                    f"{name}: sharpness {index + 1}/{len(dataset)}",
                    flush=True,
                )

        atomic_write_ssim_csv(
            args.output_dir / f"{name}_sharpness.csv",
            rows,
        )

        save_sharpness_roi_examples(
            random_examples,
            args.output_dir / f"{name}_random_roi_examples.png",
        )

        if failed_examples:
            save_sharpness_roi_examples(
                failed_examples,
                args.output_dir / f"{name}_failed_roi_examples.png",
            )

        results[name] = summarize_sharpness_rows(rows)
        rows_by_group[name] = rows

        print(
            f"{name}: valid ROI " f"{results[name]['valid_roi_count']}/{len(dataset)}",
            flush=True,
        )

    # Empty failures are explicitly recorded as an empty JSON list.
    atomic_write_json(
        args.output_dir / "invalid_roi_records.json",
        invalid_rows,
    )

    save_sharpness_distributions(
        rows_by_group,
        args.output_dir / "sharpness_distributions.png",
    )

    flat_summary = []

    for name, summary in results.items():
        for metric, statistics in summary["metrics"].items():
            flat_summary.append(
                {
                    "group": name,
                    "metric": metric,
                    "total_images": summary["image_count"],
                    "invalid_roi_count": summary["invalid_roi_count"],
                    "count": statistics["count"],
                    "mean": statistics.get("mean"),
                    "std": statistics.get("std"),
                    "median": statistics.get("median"),
                    "q25": statistics.get("q25"),
                    "q75": statistics.get("q75"),
                    "p05": statistics.get("p05"),
                    "p95": statistics.get("p95"),
                }
            )

    atomic_write_ssim_csv(
        args.output_dir / "sharpness_summary.csv",
        flat_summary,
    )

    atomic_write_json(
        args.output_dir / "sharpness_results.json",
        {
            "schema_version": 1,
            "status": "complete",
            "reference_split": reference_split,
            "settings": settings,
            "calibration_protocol": str(args.protocol.resolve()),
            "calibration_protocol_sha256": file_sha256(args.protocol),
            "cache_provenance": provenance,
            "implementation_sha256": sharpness_implementation_hash(),
            "versions": sharpness_versions(),
            "results": results,
            "invalid_roi_policy": (
                "Retain full-image metrics; record interior metrics as null. "
                "Do not replace or resample failed cases."
            ),
            "interpretation": (
                "Compare distributions with real images; "
                "higher values are not necessarily better."
            ),
            "completed_utc": utc_now(),
        },
    )


def parse_sharpness_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Calibrate and evaluate Laplacian/Tenengrad sharpness."
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=("calibrate", "cohort"),
    )
    parser.add_argument(
        "--cohort-dir",
        type=Path,
        default=Path("evaluation/cohort_final_674672"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/groups/comp3710/ADNI/AD_NC"),
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--protocol", type=Path, default=None)
    parser.add_argument("--expected-count", type=int, default=5000)
    parser.add_argument("--calibration-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)

    # Calibration-only parameters; cohort reads frozen values from JSON.
    parser.add_argument("--foreground-threshold", type=float, default=None)
    parser.add_argument("--erosion-pixels", type=float, default=None)

    args = parser.parse_args(argv)

    if args.seed < 0:
        parser.error("--seed must be non-negative.")

    if args.expected_count < 2 or args.calibration_count < 2:
        parser.error("Sample counts must be at least 2.")

    if args.phase == "cohort":
        if args.protocol is None:
            parser.error("--phase cohort requires --protocol.")

        if args.foreground_threshold is not None or args.erosion_pixels is not None:
            parser.error(
                "Cohort evaluation uses the frozen protocol; "
                "ROI parameters can only be supplied during calibration."
            )
    else:
        if args.foreground_threshold is None:
            args.foreground_threshold = 0.05

        if args.erosion_pixels is None:
            args.erosion_pixels = 3.0

        if not (
            np.isfinite(args.foreground_threshold)
            and 0.0 < args.foreground_threshold < 1.0
        ):
            parser.error("Foreground threshold must be between 0 and 1.")

        if not (np.isfinite(args.erosion_pixels) and args.erosion_pixels >= 2.0):
            parser.error("Erosion radius must be finite and at least 2 pixels.")

    if args.cache_dir is None:
        args.cache_dir = args.cohort_dir / "stage_b" / "features"

    if args.output_dir is None:
        directory = (
            "sharpness_calibration" if args.phase == "calibrate" else "sharpness"
        )
        args.output_dir = args.cohort_dir / "stage_b" / directory

    return args


def sharpness_main(argv=None):
    args = parse_sharpness_args(argv)

    if args.output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {args.output_dir}\n"
            "Use a new --output-dir rather than overwriting an experiment."
        )

    if not args.cohort_dir.is_dir():
        raise NotADirectoryError(args.cohort_dir)

    if not args.data_root.is_dir():
        raise NotADirectoryError(args.data_root)

    args.output_dir.mkdir(parents=True)

    status_path = args.output_dir / "run_status.json"
    status = {
        "status": "incomplete",
        "phase": args.phase,
        "started_utc": utc_now(),
    }
    atomic_write_json(status_path, status)

    try:
        if args.phase == "calibrate":
            run_sharpness_calibration(args)
        else:
            run_sharpness_cohort(args)

        status["status"] = "complete"
        status["completed_utc"] = utc_now()
        atomic_write_json(status_path, status)

    except Exception as error:
        status["status"] = "failed"
        status["error"] = f"{type(error).__name__}: {error}"
        atomic_write_json(status_path, status)
        raise

    print(
        f"\nSharpness {args.phase} complete: " f"{args.output_dir.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "fid":
        fid_main(sys.argv[2:])

    elif len(sys.argv) > 1 and sys.argv[1] == "ssim":
        ssim_main(sys.argv[2:])

    elif len(sys.argv) > 1 and sys.argv[1] == "sharpness":
        sharpness_main(sys.argv[2:])

    else:
        main()
