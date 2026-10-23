import argparse
import hashlib
import time

from collections import Counter, defaultdict
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

import evaluation as ev

MODELS = ("dcgan", "stylegan2")


def numeric_summary(values):
    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        return None

    if not np.isfinite(values).all():
        raise FloatingPointError("Summary contains non-finite values.")

    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def write_csv(path, rows):
    # Existing helper requires at least one row.
    if rows:
        ev.atomic_write_ssim_csv(path, rows)


def make_split_dataset(name, records, data_root):
    rows = []

    for index, record in enumerate(records):
        row = dict(record)
        row["reference_id"] = f"{name}_{index:06d}"
        rows.append(row)

    if not rows:
        raise ValueError(f"Empty split: {name}")

    paths = [row["relative_path"] for row in rows]

    if len(paths) != len(set(paths)):
        raise ValueError(f"Duplicate paths in {name}.")

    required = {
        "relative_path",
        "subject_id",
        "image_id",
        "slice_index",
        "diagnosis",
    }

    for row in rows:
        if not required.issubset(row):
            raise ValueError(f"Incomplete record in {name}: {row}")

        if not str(row["subject_id"]):
            raise ValueError(f"Empty subject ID in {name}.")

    spec = {
        "root": Path(data_root).resolve(),
        "rows": rows,
        "id_field": "reference_id",
        "manifest_sha256": ev.object_sha256(rows),
        "kind": "real",
    }

    return ev.EvaluationImageDataset(name, spec)


def population_summary(dataset):
    return {
        "images": len(dataset),
        "patients": len({str(row["subject_id"]) for row in dataset.rows}),
        "diagnoses": dict(Counter(row["diagnosis"] for row in dataset.rows)),
    }


def verify_training_pool(args, stage_a, split_payload):
    """
    Check the saved training configurations against the split snapshot.

    These older configs do not necessarily contain the training manifest's
    SHA256, so this checks configuration compatibility, not retrospective
    cryptographic proof of the checkpoint's training membership.
    """
    split_config = split_payload["config"]
    manifest_name = Path(stage_a["arguments"]["manifest"].replace("\\", "/")).name

    result = {}

    for model_name in MODELS:
        config_path = args.cohort_dir / model_name / "generation_config.json"
        generated = ev.read_json(config_path)
        training = generated["training_config"]

        training_manifest_name = Path(training["manifest_path"].replace("\\", "/")).name

        if training_manifest_name != manifest_name:
            raise ValueError(f"{model_name}: training manifest name differs.")

        if set(training["included_classes"]) != set(split_config["included_classes"]):
            raise ValueError(f"{model_name}: included_classes differ.")

        if int(training["split_seed"]) != int(split_config["seed"]):
            raise ValueError(f"{model_name}: split seed differs.")

        if abs(float(training["val_ratio"]) - float(split_config["val_ratio"])) > 1e-12:
            raise ValueError(f"{model_name}: val_ratio differs.")

        result[model_name] = {
            "generation_config_sha256": ev.file_sha256(config_path),
            "checkpoint_sha256": generated["checkpoint_sha256"],
            "checkpoint_epoch": generated["checkpoint_epoch"],
            "weights_key": generated["weights_key"],
            "manifest_path": training["manifest_path"],
            "verification": "saved_training_configuration_compatible",
        }

    return result


def check_extraction_environment(config, weights_path, device):
    ev.verify_hash(
        weights_path,
        config["weights_sha256"],
        "Inception weights",
    )

    current_versions = {
        "torch": str(torch.__version__),
        "torchvision": version("torchvision"),
        "torch-fidelity": version("torch-fidelity"),
        "numpy": version("numpy"),
        "Pillow": version("Pillow"),
    }

    for name, current in current_versions.items():
        saved = config["versions"][name]

        if current != saved:
            raise ValueError(
                f"{name} version mismatch: saved={saved}, "
                f"current={current}. Use the feature-extraction environment."
            )

    if device.type != config["device_type"]:
        raise ValueError("Use the same device type as the existing feature cache.")

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable.")

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_float32_matmul_precision("highest")


@torch.no_grad()
def nearest_neighbors(query, reference, k, device, chunk_size=64):
    """
    Exhaustive feature-space search against every reference image.

    float64 reduces cancellation errors in distance computation.
    Only one query chunk's distance matrix is held at a time.
    """
    if query.ndim != 2 or reference.ndim != 2:
        raise ValueError("Expected two feature matrices.")

    if query.shape[1] != reference.shape[1]:
        raise ValueError("Feature dimensions differ.")

    if not 1 <= k <= len(reference):
        raise ValueError("Invalid number of neighbors.")

    reference_device = reference.to(device=device, dtype=torch.float64)

    all_distances = []
    all_indices = []

    for start in range(0, len(query), chunk_size):
        q = query[start : start + chunk_size].to(device=device, dtype=torch.float64)

        distances = torch.cdist(
            q,
            reference_device,
            p=2,
            compute_mode="use_mm_for_euclid_dist",
        )

        if not torch.isfinite(distances).all():
            raise FloatingPointError("Non-finite NN distances.")

        _, indices = torch.topk(
            distances,
            k=k,
            dim=1,
            largest=False,
            sorted=True,
        )

        # Recompute selected distances directly, avoiding the
        # subtractive cancellation of the matrix formula.
        selected = reference_device[indices]
        exact_distances = torch.linalg.vector_norm(
            q[:, None, :] - selected,
            ord=2,
            dim=2,
        )

        order = exact_distances.argsort(dim=1)
        indices = indices.gather(1, order)
        exact_distances = exact_distances.gather(1, order)

        all_distances.append(exact_distances.cpu())
        all_indices.append(indices.cpu())

        if start == 0 or start + chunk_size >= len(query):
            print(
                f"NN search: {min(start + chunk_size, len(query))}" f"/{len(query)}",
                flush=True,
            )

    return (
        torch.cat(all_distances).numpy(),
        torch.cat(all_indices).numpy(),
    )


def decoded_pixel_hashes(dataset):
    """
    Hash the exact preprocessed uint8 pixel array, not JPEG/PNG file bytes.
    Thus different file encodings do not hide identical decoded pixels.
    """
    hashes = []

    for index in range(len(dataset)):
        pixels, returned_index = dataset[index]

        if returned_index != index:
            raise ValueError("Image index mismatch.")

        array = np.ascontiguousarray(pixels.numpy())
        digest = hashlib.sha256(array.tobytes()).hexdigest()
        hashes.append(digest)

        if (index + 1) % 1000 == 0 or index + 1 == len(dataset):
            print(
                f"{dataset.name}: pixel hashes " f"{index + 1}/{len(dataset)}",
                flush=True,
            )

    return hashes


def make_hash_lookup(hashes):
    lookup = defaultdict(list)

    for index, digest in enumerate(hashes):
        lookup[digest].append(index)

    return lookup


def pixel_metrics(query_image, train_image, foreground_threshold):
    difference = np.abs(query_image - train_image)

    # Union support includes foreground present in either image,
    # so misalignment is not hidden by intersecting the two masks.
    support = (query_image > foreground_threshold) | (
        train_image > foreground_threshold
    )

    return {
        "ssim": ev.calculate_image_ssim(train_image, query_image),
        "mae_full": float(difference.mean()),
        "mae_support": (float(difference[support].mean()) if support.any() else None),
        "support_fraction": float(support.mean()),
    }


def pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_pair_panel(name, selected_rows, generated_dataset, train_dataset, path):
    plt = pyplot()

    figure, axes = plt.subplots(
        len(selected_rows),
        3,
        figsize=(11, 3.2 * len(selected_rows)),
        squeeze=False,
    )

    for position, row in enumerate(selected_rows):
        query_index = int(row["query_index"])
        train_index = int(row["train_index"])

        query = ev.load_ssim_image(generated_dataset, query_index)
        train = ev.load_ssim_image(train_dataset, train_index)
        difference = np.abs(query - train)

        axes[position, 0].imshow(query, cmap="gray", vmin=0, vmax=1)
        axes[position, 1].imshow(train, cmap="gray", vmin=0, vmax=1)
        axes[position, 2].imshow(difference, cmap="magma", vmin=0, vmax=1)

        axes[position, 0].set_title(f"{name}: {row['sample_id']}")
        axes[position, 1].set_title(
            f"Train: {row['train_subject_id']} / "
            f"{row['train_image_id']} / "
            f"slice {row['train_slice_index']}"
        )
        axes[position, 2].set_title(
            f"|difference|, scale [0, 1]\n"
            f"d={row['feature_distance']:.3f}, "
            f"SSIM={row['ssim']:.4f}"
        )

        for axis in axes[position]:
            axis.axis("off")

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_distance_plot(distributions, threshold, path):
    plt = pyplot()
    figure, axis = plt.subplots(figsize=(9, 6))

    for name, values in distributions.items():
        values = np.sort(np.asarray(values, dtype=np.float64))

        if not len(values):
            continue

        probability = np.arange(1, len(values) + 1) / len(values)
        axis.plot(values, probability, label=name)

    axis.axvline(
        threshold,
        color="black",
        linestyle="--",
        label="Validation lower-tail review threshold",
    )
    axis.set_xlabel("Nearest training-image feature distance")
    axis.set_ylabel("Empirical cumulative probability")
    axis.set_title("All groups use the same complete training reference")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def audit_generated_group(
    name,
    dataset,
    features,
    train_dataset,
    train_features,
    train_hash_lookup,
    threshold,
    args,
):
    distances, indices = nearest_neighbors(
        features,
        train_features,
        k=args.top_k,
        device=args.device,
        chunk_size=args.query_chunk,
    )

    hashes = decoded_pixel_hashes(dataset)

    @lru_cache(maxsize=128)
    def training_image(index):
        return ev.load_ssim_image(train_dataset, index)

    candidates = []
    queries = []
    exact_matches = []
    nn_rows = []
    best_ssim_rows = []

    for query_index in range(len(dataset)):
        query_image = ev.load_ssim_image(dataset, query_index)
        sample_id = dataset.sample_ids[query_index]
        local_rows = []

        for rank in range(args.top_k):
            train_index = int(indices[query_index, rank])
            record = train_dataset.rows[train_index]

            metrics = pixel_metrics(
                query_image,
                training_image(train_index),
                args.foreground_threshold,
            )

            row = {
                "model": name,
                "query_index": query_index,
                "sample_id": sample_id,
                "generated_relative_path": dataset.relative_paths[query_index],
                "feature_rank": rank + 1,
                "feature_distance": float(distances[query_index, rank]),
                "train_index": train_index,
                "train_relative_path": record["relative_path"],
                "train_subject_id": record["subject_id"],
                "train_image_id": record["image_id"],
                "train_slice_index": record["slice_index"],
                "train_diagnosis": record["diagnosis"],
                **metrics,
            }

            candidates.append(row)
            local_rows.append(row)

        nearest = local_rows[0]
        best_ssim = max(local_rows, key=lambda row: row["ssim"])
        matches = train_hash_lookup.get(hashes[query_index], [])

        for train_index in matches:
            record = train_dataset.rows[train_index]

            exact_matches.append(
                {
                    "model": name,
                    "sample_id": sample_id,
                    "query_index": query_index,
                    "generated_relative_path": dataset.relative_paths[query_index],
                    "train_index": train_index,
                    "train_relative_path": record["relative_path"],
                    "train_subject_id": record["subject_id"],
                    "train_image_id": record["image_id"],
                    "train_slice_index": record["slice_index"],
                }
            )

        queries.append(
            {
                "model": name,
                "query_index": query_index,
                "sample_id": sample_id,
                "nn_train_index": nearest["train_index"],
                "nn_train_subject_id": nearest["train_subject_id"],
                "nn_feature_distance": nearest["feature_distance"],
                "nn_ssim": nearest["ssim"],
                "nn_mae_full": nearest["mae_full"],
                "nn_mae_support": nearest["mae_support"],
                "best_candidate_ssim": best_ssim["ssim"],
                "best_ssim_train_index": best_ssim["train_index"],
                "exact_pixel_match_count": len(matches),
                "review_flag": (nearest["feature_distance"] < threshold),
            }
        )

        nn_rows.append(nearest)
        best_ssim_rows.append(best_ssim)

        if (query_index + 1) % 500 == 0:
            print(
                f"{name}: candidate SSIM " f"{query_index + 1}/{len(dataset)}",
                flush=True,
            )

    output = args.output_dir / name
    output.mkdir(parents=True, exist_ok=True)

    write_csv(output / "candidate_pairs.csv", candidates)
    write_csv(output / "per_generated_image.csv", queries)

    # JSON also represents the no-match case unambiguously as [].
    ev.atomic_write_json(output / "exact_pixel_matches.json", exact_matches)

    count = min(args.examples, len(dataset))
    rng = np.random.default_rng(args.seed)
    random_indices = rng.choice(len(dataset), size=count, replace=False)

    selections = {
        "random_pairs": [nn_rows[int(i)] for i in random_indices],
        "smallest_distance_pairs": sorted(
            nn_rows, key=lambda row: row["feature_distance"]
        )[:count],
        "highest_candidate_ssim_pairs": sorted(
            best_ssim_rows,
            key=lambda row: row["ssim"],
            reverse=True,
        )[:count],
    }

    for selection_name, rows in selections.items():
        save_pair_panel(
            name,
            rows,
            dataset,
            train_dataset,
            output / f"{selection_name}.png",
        )
        write_csv(output / f"{selection_name}.csv", rows)

    image_usage = Counter(row["nn_train_index"] for row in queries)
    patient_usage = Counter(str(row["nn_train_subject_id"]) for row in queries)

    result = {
        "generated_count": len(queries),
        "training_reference_count": len(train_dataset),
        "feature_top_k": args.top_k,
        "nearest_feature_distance": numeric_summary(distances[:, 0]),
        "ssim_at_feature_nn": numeric_summary([row["nn_ssim"] for row in queries]),
        "best_ssim_among_feature_candidates": numeric_summary(
            [row["best_candidate_ssim"] for row in queries]
        ),
        "nn_mae_full": numeric_summary([row["nn_mae_full"] for row in queries]),
        "nn_mae_support": numeric_summary(
            [
                row["nn_mae_support"]
                for row in queries
                if row["nn_mae_support"] is not None
            ]
        ),
        "empty_support_count": sum(row["nn_mae_support"] is None for row in queries),
        "generated_images_with_exact_pixel_match": sum(
            row["exact_pixel_match_count"] > 0 for row in queries
        ),
        "exact_pixel_match_pairs": len(exact_matches),
        "review_flag_count": sum(row["review_flag"] for row in queries),
        "review_flag_fraction": float(np.mean([row["review_flag"] for row in queries])),
        "unique_nearest_training_images": len(image_usage),
        "unique_nearest_training_patients": len(patient_usage),
        "maximum_queries_per_training_image": max(image_usage.values()),
        "maximum_queries_per_training_patient": max(patient_usage.values()),
    }

    return result, distances[:, 0]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training-image memorization and nearest-neighbor audit."
    )

    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--inception-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query-chunk", type=int, default=64)
    parser.add_argument("--review-quantile", type=float, default=0.01)
    parser.add_argument("--foreground-threshold", type=float, default=0.05)
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--expected-count", type=int, default=5000)

    args = parser.parse_args()

    if (
        min(
            args.batch_size,
            args.top_k,
            args.query_chunk,
            args.examples,
            args.expected_count,
        )
        < 1
    ):
        parser.error("Batch size/counts must be positive.")

    if args.num_workers < 0:
        parser.error("num-workers must be non-negative.")

    if not 0 < args.review_quantile < 0.5:
        parser.error("review-quantile must be between 0 and 0.5.")

    if not 0 <= args.foreground_threshold < 1:
        parser.error("foreground-threshold must be in [0, 1).")

    args.cohort_dir = args.cohort_dir.resolve()
    args.data_root = args.data_root.resolve()
    args.inception_weights = args.inception_weights.resolve()
    args.cache_dir = args.cohort_dir / "stage_b" / "features"

    if args.output_dir is None:
        args.output_dir = args.cohort_dir / "stage_c" / "memorization"

    args.output_dir = args.output_dir.resolve()
    args.device = torch.device(args.device)

    return args


def run_audit(args, report):
    (
        existing_features,
        provenance,
        extraction_config,
        reference_split,
    ) = ev.load_fid_feature_caches(
        args.cache_dir,
        args.expected_count,
    )

    if reference_split != "test":
        raise ValueError("This audit expects the Stage A real reference to be test.")

    stage_a, datasets = ev.load_ssim_datasets(args, provenance, extraction_config)

    snapshot_path = args.cohort_dir / "source_splits.json"

    ev.verify_hash(
        snapshot_path,
        stage_a["real_reference"]["source_manifest_sha256"],
        "source split snapshot",
    )

    split_payload = ev.read_json(snapshot_path)

    report["training_pool_evidence"] = verify_training_pool(
        args, stage_a, split_payload
    )

    train = make_split_dataset(
        "audit_train",
        split_payload["splits"]["train"],
        args.data_root,
    )
    validation = make_split_dataset(
        "audit_validation",
        split_payload["splits"]["validation"],
        args.data_root,
    )
    real = datasets["real"]

    if args.top_k > len(train):
        raise ValueError("top-k exceeds the training reference size.")

    train_subjects = {str(row["subject_id"]) for row in train.rows}
    validation_subjects = {str(row["subject_id"]) for row in validation.rows}
    test_subjects = {str(row["subject_id"]) for row in real.rows}

    train_validation_overlap = train_subjects & validation_subjects

    if train_validation_overlap:
        raise ValueError(
            "Validation controls overlap training patients: "
            f"{sorted(train_validation_overlap)}"
        )

    train_paths = set(train.relative_paths)

    if train_paths & set(validation.relative_paths):
        raise ValueError("Train/validation image paths overlap.")

    if train_paths & set(real.relative_paths):
        raise ValueError("Train/test image paths overlap.")

    report["populations"] = {
        "train": population_summary(train),
        "validation": population_summary(validation),
        "selected_test": population_summary(real),
        "train_validation_patient_overlap": 0,
        "train_selected_test_patient_overlap": len(train_subjects & test_subjects),
    }

    check_extraction_environment(
        extraction_config,
        args.inception_weights,
        args.device,
    )

    extractor = ev.build_feature_extractor(args.inception_weights, args.device)

    # Check that the current extraction helper still agrees numerically
    # with the already-cached real-image features.
    count = min(args.batch_size, len(real))
    check_spec = {
        "root": real.root,
        "rows": real.rows[:count],
        "id_field": real.id_field,
        "manifest_sha256": ev.object_sha256(real.rows[:count]),
        "kind": "real",
    }
    check_dataset = ev.EvaluationImageDataset("cache_consistency_check", check_spec)

    current_check = ev.extract_features(check_dataset, extractor, args, args.device)
    cached_check = existing_features["real"][:count]

    torch.testing.assert_close(
        current_check,
        cached_check,
        rtol=1e-4,
        atol=1e-4,
    )

    report["feature_consistency_max_abs_difference"] = float(
        (current_check - cached_check).abs().max()
    )

    # Keep actual new extraction provenance instead of claiming that
    # the current script is the old feature-extraction source.
    new_extraction_config = dict(extraction_config)
    new_extraction_config.update(
        {
            "evaluation_source_sha256": ev.file_sha256(Path(ev.__file__)),
            "audit_source_sha256": ev.file_sha256(Path(__file__)),
            "batch_size": args.batch_size,
            "gpu_name": (
                torch.cuda.get_device_name(args.device)
                if args.device.type == "cuda"
                else None
            ),
        }
    )

    feature_dir = args.output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    added_features = {}

    for dataset in (train, validation):
        payload, _ = ev.extract_or_load_features(
            dataset=dataset,
            cache_path=feature_dir / f"{dataset.name}_inception2048.pt",
            extraction_config=new_extraction_config,
            get_extractor=lambda: extractor,
            args=args,
            device=args.device,
        )

        added_features[dataset.name] = payload["features"]

        report.setdefault("new_feature_caches", {})[dataset.name] = {
            "fingerprint": payload["fingerprint"],
            "sample_count": len(dataset),
        }

    del extractor

    if args.device.type == "cuda":
        torch.cuda.empty_cache()

    train_features = added_features["audit_train"]
    validation_features = added_features["audit_validation"]

    # Identity sanity check: training features searched against themselves.
    sanity_distances, _ = nearest_neighbors(
        train_features[:8],
        train_features,
        k=1,
        device=args.device,
        chunk_size=args.query_chunk,
    )

    if float(sanity_distances.max()) > 1e-8:
        raise ValueError("Nearest-neighbor identity check failed.")

    identity_image = ev.load_ssim_image(train, 0)
    identity_ssim = ev.calculate_image_ssim(identity_image, identity_image)

    if abs(identity_ssim - 1.0) > 1e-10:
        raise ValueError("SSIM identity check failed.")

    report["sanity_checks"] = {
        "self_nn_max_distance": float(sanity_distances.max()),
        "self_ssim": identity_ssim,
    }

    train_hashes = decoded_pixel_hashes(train)
    train_lookup = make_hash_lookup(train_hashes)
    validation_hashes = decoded_pixel_hashes(validation)
    test_hashes = decoded_pixel_hashes(real)

    validation_distances, validation_indices = nearest_neighbors(
        validation_features,
        train_features,
        k=1,
        device=args.device,
        chunk_size=args.query_chunk,
    )
    test_distances, test_indices = nearest_neighbors(
        existing_features["real"],
        train_features,
        k=1,
        device=args.device,
        chunk_size=args.query_chunk,
    )

    # Exclude exact duplicated pixels from the unseen-image calibration
    # reference, and explicitly report how many were excluded.
    validation_eligible = np.array(
        [digest not in train_lookup for digest in validation_hashes], dtype=bool
    )

    if int(validation_eligible.sum()) < 100:
        raise ValueError("Fewer than 100 eligible validation controls.")

    validation_control_distances = validation_distances[validation_eligible, 0]

    threshold = float(
        np.quantile(
            validation_control_distances,
            args.review_quantile,
        )
    )

    control_rows = []

    for name, dataset, hashes, distances, indices in (
        (
            "validation",
            validation,
            validation_hashes,
            validation_distances,
            validation_indices,
        ),
        (
            "selected_test",
            real,
            test_hashes,
            test_distances,
            test_indices,
        ),
    ):
        for index, record in enumerate(dataset.rows):
            nearest_index = int(indices[index, 0])
            train_record = train.rows[nearest_index]

            control_rows.append(
                {
                    "group": name,
                    "reference_id": dataset.sample_ids[index],
                    "relative_path": record["relative_path"],
                    "subject_id": record["subject_id"],
                    "diagnosis": record["diagnosis"],
                    "patient_seen_in_training": (
                        str(record["subject_id"]) in train_subjects
                    ),
                    "exact_training_pixel_matches": len(
                        train_lookup.get(hashes[index], [])
                    ),
                    "nearest_train_index": nearest_index,
                    "nearest_train_path": train_record["relative_path"],
                    "nearest_train_subject_id": train_record["subject_id"],
                    "feature_distance": float(distances[index, 0]),
                    "used_for_threshold": (
                        name == "validation" and bool(validation_eligible[index])
                    ),
                }
            )

    write_csv(args.output_dir / "real_control_neighbors.csv", control_rows)

    unseen_test_mask = np.array(
        [
            str(row["subject_id"]) not in train_subjects and digest not in train_lookup
            for row, digest in zip(real.rows, test_hashes)
        ],
        dtype=bool,
    )

    report["calibration"] = {
        "reference": "validation queries against complete train reference",
        "review_quantile": args.review_quantile,
        "review_threshold": threshold,
        "rule": "nearest_feature_distance < review_threshold",
        "interpretation": "review flag, not a memorization decision",
        "validation_controls_used": int(validation_eligible.sum()),
        "validation_exact_duplicate_images_excluded": int((~validation_eligible).sum()),
        "validation_distance": numeric_summary(validation_control_distances),
        "test_all_distance": numeric_summary(test_distances[:, 0]),
        "test_unseen_patient_nonduplicate_count": int(unseen_test_mask.sum()),
        "test_unseen_patient_distance": numeric_summary(
            test_distances[unseen_test_mask, 0]
        ),
    }

    report["training_pixel_duplicates"] = {
        "unique_pixel_arrays": len(train_lookup),
        "extra_duplicate_records": len(train) - len(train_lookup),
        "maximum_records_per_pixel_array": max(
            len(indices) for indices in train_lookup.values()
        ),
    }

    distributions = {
        "Validation, unseen patients": validation_control_distances,
        "Test, all selected": test_distances[:, 0],
        "Test, unseen patients": test_distances[unseen_test_mask, 0],
    }

    report["models"] = {}

    for name in MODELS:
        result, nearest_distances = audit_generated_group(
            name=name,
            dataset=datasets[name],
            features=existing_features[name],
            train_dataset=train,
            train_features=train_features,
            train_hash_lookup=train_lookup,
            threshold=threshold,
            args=args,
        )

        report["models"][name] = result
        distributions[name] = nearest_distances

        # Save partial progress after each completed model.
        ev.atomic_write_json(args.output_dir / "audit_results.json", report)

    save_distance_plot(
        distributions,
        threshold,
        args.output_dir / "nearest_train_distance_ecdf.png",
    )

    summary_rows = []

    for name, result in report["models"].items():
        summary_rows.append(
            {
                "model": name,
                "generated_count": result["generated_count"],
                "training_reference_count": result["training_reference_count"],
                "exact_pixel_match_images": result[
                    "generated_images_with_exact_pixel_match"
                ],
                "nn_distance_median": result["nearest_feature_distance"]["median"],
                "nn_distance_p01": result["nearest_feature_distance"]["p01"],
                "nn_ssim_median": result["ssim_at_feature_nn"]["median"],
                "best_candidate_ssim_max": result["best_ssim_among_feature_candidates"][
                    "max"
                ],
                "review_flag_count": result["review_flag_count"],
                "review_flag_fraction": result["review_flag_fraction"],
                "unique_nn_training_images": result["unique_nearest_training_images"],
                "unique_nn_training_patients": result[
                    "unique_nearest_training_patients"
                ],
            }
        )

    write_csv(args.output_dir / "audit_summary.csv", summary_rows)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Freeze settings for this output directory.
    protocol = {
        "schema_version": 1,
        "audit_source_sha256": ev.file_sha256(Path(__file__)),
        "evaluation_source_sha256": ev.file_sha256(Path(ev.__file__)),
        "stage_a_protocol_sha256": ev.file_sha256(args.cohort_dir / "protocol.json"),
        "top_k": args.top_k,
        "review_quantile": args.review_quantile,
        "foreground_threshold": args.foreground_threshold,
        "expected_count": args.expected_count,
        "seed": args.seed,
        "examples": args.examples,
        "feature_distance": "Euclidean, unnormalised Inception 2048",
        "pairing": "feature top-k, then candidate SSIM and MAE",
        "alignment": "none",
        "pixel_comparison": "preprocessed uint8 grayscale 256x256",
        "ssim": {
            "data_range": 1.0,
            "gaussian_weights": True,
            "sigma": 1.5,
            "win_size": 11,
            "use_sample_covariance": False,
        },
    }

    protocol_path = args.output_dir / "audit_protocol.json"

    if protocol_path.exists():
        if ev.read_json(protocol_path) != protocol:
            raise ValueError("Audit settings/source changed. Use a new --output-dir.")
    else:
        ev.atomic_write_json(protocol_path, protocol)

    started = time.perf_counter()
    report = {
        "schema_version": 1,
        "status": "running",
        "started_utc": ev.utc_now(),
        "protocol": protocol,
        "existing_feature_summary_sha256": ev.file_sha256(
            args.cache_dir / "feature_summary.json"
        ),
        "limitations": [
            "Finite generated cohort; unobserved outputs are not audited.",
            "Inception features may miss medical or transformed copies.",
            "Candidate SSIM is not an exhaustive global SSIM search.",
            "Review flags are not confirmed memorization or attack success.",
            "Slices from one patient are correlated; no IID confidence claim.",
            "Exact matching applies to the exported uint8 pixel representation.",
        ],
    }

    report_path = args.output_dir / "audit_results.json"

    try:
        run_audit(args, report)
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["finished_utc"] = ev.utc_now()
        ev.atomic_write_json(report_path, report)

    print(
        f"\nMemorization audit complete: {args.output_dir}",
        flush=True,
    )

    for name, result in report["models"].items():
        print(
            f"{name}: "
            f"exact-match images="
            f"{result['generated_images_with_exact_pixel_match']}, "
            f"NN distance median="
            f"{result['nearest_feature_distance']['median']:.4f}, "
            f"review flags="
            f"{result['review_flag_count']}/"
            f"{result['generated_count']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
