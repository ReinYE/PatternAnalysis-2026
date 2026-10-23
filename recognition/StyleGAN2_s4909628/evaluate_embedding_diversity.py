import argparse
import inspect
import time

from collections import Counter
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits

import evaluation as ev

GROUPS = ("real", "dcgan", "stylegan2")

COLORS = {
    "real": "#4477AA",
    "dcgan": "#EE6677",
    "stylegan2": "#228833",
}

PAIR_COLORS = {
    "real_all": "#4477AA",
    "real_cross_patient": "#66CCEE",
    "dcgan": "#EE6677",
    "stylegan2": "#228833",
}


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to write: {path}")

    ev.atomic_write_ssim_csv(path, rows)


def numeric_summary(values):
    values = np.asarray(values, dtype=np.float64)

    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Empty or non-finite summary values.")

    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def save_figure(figure, stem):
    figure.savefig(stem.with_suffix(".png"), dpi=180)
    figure.savefig(stem.with_suffix(".pdf"))
    plt.close(figure)


def set_embedding_axes(axis):
    axis.set_xlabel("t-SNE 1")
    axis.set_ylabel("t-SNE 2")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.15)


def plot_embedding(coordinates, metadata, stem, seed):
    labels = np.array([row["group"] for row in metadata])

    # Shuffle plotting order so the last model does not always cover
    # the other groups where points overlap.
    order = np.random.default_rng(seed).permutation(len(metadata))

    figure, axis = plt.subplots(figsize=(9, 8))

    axis.scatter(
        coordinates[order, 0],
        coordinates[order, 1],
        c=[COLORS[label] for label in labels[order]],
        s=5,
        alpha=0.45,
        linewidths=0,
        rasterized=True,
    )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=COLORS[name],
            label=f"{name} (n={int((labels == name).sum())})",
        )
        for name in GROUPS
    ]

    axis.legend(handles=handles)
    axis.set_title("Joint t-SNE: real and generated images")
    set_embedding_axes(axis)
    figure.tight_layout()
    save_figure(figure, stem.with_name(stem.name + "_joint"))

    # Three panels show subsets of the SAME fitted embedding.
    # They do not fit three separate t-SNE models.
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharex=True, sharey=True)

    for axis, name in zip(axes, GROUPS):
        mask = labels == name

        axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c="#DDDDDD",
            s=3,
            alpha=0.20,
            linewidths=0,
            rasterized=True,
        )
        axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            c=COLORS[name],
            s=5,
            alpha=0.55,
            linewidths=0,
            rasterized=True,
        )
        axis.set_title(name)
        set_embedding_axes(axis)

    figure.suptitle("Group views of the same joint embedding")
    figure.tight_layout()
    save_figure(figure, stem.with_name(stem.name + "_panels"))

    # Only real images have diagnosis ground truth.
    real_mask = labels == "real"
    diagnoses = np.array([row["diagnosis"] for row in metadata])

    figure, axis = plt.subplots(figsize=(9, 8))
    axis.scatter(
        coordinates[~real_mask, 0],
        coordinates[~real_mask, 1],
        c="#CCCCCC",
        s=4,
        alpha=0.20,
        linewidths=0,
        label="Generated: diagnosis unavailable",
        rasterized=True,
    )

    for diagnosis, color in (("AD", "#CC3311"), ("NC", "#0077BB")):
        mask = real_mask & (diagnoses == diagnosis)

        axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            c=color,
            s=7,
            alpha=0.55,
            linewidths=0,
            label=f"Real {diagnosis}",
            rasterized=True,
        )

    axis.set_title("Real diagnosis labels on the joint embedding")
    axis.legend()
    set_embedding_axes(axis)
    figure.tight_layout()
    save_figure(figure, stem.with_name(stem.name + "_real_diagnosis"))


def run_embedding(args, features, records):
    output = args.output_dir / "embedding"
    output.mkdir(parents=True, exist_ok=True)

    matrices = []
    metadata = []

    for name in GROUPS:
        matrices.append(features[name].numpy())

        id_field = "reference_id" if name == "real" else "sample_id"

        for index, row in enumerate(records[name]):
            metadata.append(
                {
                    "group": name,
                    "source_index": index,
                    "sample_id": row[id_field],
                    "relative_path": row["relative_path"],
                    "subject_id": row.get("subject_id", ""),
                    "diagnosis": (row.get("diagnosis", "") if name == "real" else ""),
                    "image_id": row.get("image_id", ""),
                    "slice_index": row.get("slice_index", ""),
                }
            )

    matrix = np.concatenate(matrices, axis=0).astype(np.float32, copy=False)

    if not np.isfinite(matrix).all():
        raise FloatingPointError("Non-finite embedding input.")

    if args.pca_components >= min(matrix.shape):
        raise ValueError("Too many PCA components.")

    if max(args.perplexities) >= len(matrix):
        raise ValueError("Perplexity must be smaller than sample count.")

    write_csv(output / "embedding_manifest.csv", metadata)

    print(
        f"Joint PCA: {matrix.shape} -> " f"({len(matrix)}, {args.pca_components})",
        flush=True,
    )

    pca = PCA(
        n_components=args.pca_components,
        whiten=False,
        svd_solver="randomized",
        random_state=args.seed,
    )

    with threadpool_limits(limits=args.threads):
        reduced = pca.fit_transform(matrix)

    if not np.isfinite(reduced).all():
        raise FloatingPointError("Non-finite PCA output.")

    np.savez_compressed(
        output / "joint_pca.npz",
        features=reduced,
        components=pca.components_,
        mean=pca.mean_,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )

    summary = {
        "sample_count": len(matrix),
        "per_group_count": {name: len(features[name]) for name in GROUPS},
        "input_feature_dimension": int(matrix.shape[1]),
        "pca_components": args.pca_components,
        "pca_whiten": False,
        "feature_standardization": "none",
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "runs": [],
    }

    for perplexity in args.perplexities:
        for tsne_seed in args.tsne_seeds:
            started = time.perf_counter()
            name = f"tsne_p{perplexity:g}_seed{tsne_seed}"

            # Numeric equivalent of sklearn's learning_rate="auto",
            # also compatible with older sklearn releases.
            learning_rate = max(len(reduced) / 12.0 / 4.0, 50.0)

            kwargs = {
                "n_components": 2,
                "perplexity": perplexity,
                "early_exaggeration": 12.0,
                "learning_rate": learning_rate,
                "init": "pca",
                "metric": "euclidean",
                "method": "barnes_hut",
                "angle": 0.5,
                "random_state": tsne_seed,
                "n_jobs": args.threads,
                "verbose": 1,
            }

            # sklearn renamed n_iter to max_iter in version 1.5.
            signature = inspect.signature(TSNE).parameters
            iteration_key = "max_iter" if "max_iter" in signature else "n_iter"
            kwargs[iteration_key] = args.iterations

            print(f"\nRunning {name}", flush=True)

            with threadpool_limits(limits=args.threads):
                estimator = TSNE(**kwargs)
                coordinates = estimator.fit_transform(reduced)

            if not np.isfinite(coordinates).all():
                raise FloatingPointError(f"Non-finite coordinates: {name}")

            if not np.isfinite(estimator.kl_divergence_):
                raise FloatingPointError(f"Non-finite t-SNE objective: {name}")

            coordinate_rows = [
                {
                    **row,
                    "tsne_x": float(coordinates[index, 0]),
                    "tsne_y": float(coordinates[index, 1]),
                }
                for index, row in enumerate(metadata)
            ]

            write_csv(output / f"{name}_coordinates.csv", coordinate_rows)
            plot_embedding(
                coordinates,
                metadata,
                output / name,
                args.seed,
            )

            summary["runs"].append(
                {
                    "name": name,
                    "parameters": kwargs,
                    "iterations_recorded": int(estimator.n_iter_),
                    "kl_divergence": float(estimator.kl_divergence_),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )

            ev.atomic_write_json(output / "embedding_summary.json", summary)

    return summary


def sample_unique_pairs(dataset, count, seed, cross_patient=False):
    """
    Uniform rejection sampling over unordered image pairs.

    Each pair has i < j. No self-pairs and no repeated pairs.
    For real_cross_patient, reject pairs from the same subject.
    """
    size = len(dataset)
    possible = size * (size - 1) // 2

    subjects = None

    if cross_patient:
        subjects = [str(row["subject_id"]) for row in dataset.rows]
        patient_counts = Counter(subjects)

        possible -= sum(n * (n - 1) // 2 for n in patient_counts.values())

    if count > possible:
        raise ValueError(f"Requested {count} pairs, but only {possible} are eligible.")

    # This sampler is intended for a sparse sample of the pair space.
    if count > possible * 0.1:
        raise ValueError(
            "Requested over 10% of eligible pairs. "
            "Reduce --pairs; this rejection sampler is for sparse sampling."
        )

    rng = np.random.default_rng(seed)
    seen = set()
    pairs = []

    while len(pairs) < count:
        i, j = (int(x) for x in rng.integers(0, size, size=2))

        if i == j:
            continue

        if i > j:
            i, j = j, i

        if cross_patient and subjects[i] == subjects[j]:
            continue

        if (i, j) in seen:
            continue

        seen.add((i, j))
        pairs.append((i, j))

    return pairs


def save_pair_examples(dataset, rows, output, group, seed, count):
    count = min(count, len(rows))
    rng = np.random.default_rng(seed)

    selections = {
        "random": [
            rows[int(index)] for index in rng.choice(len(rows), count, replace=False)
        ],
        "highest": sorted(rows, key=lambda row: row["ssim"], reverse=True)[:count],
        "lowest": sorted(rows, key=lambda row: row["ssim"])[:count],
    }

    for label, chosen in selections.items():
        figure, axes = plt.subplots(count, 2, figsize=(7, 3.2 * count), squeeze=False)

        for position, row in enumerate(chosen):
            for column, side in enumerate(("a", "b")):
                image = ev.load_ssim_image(dataset, int(row[f"index_{side}"]))
                axes[position, column].imshow(image, cmap="gray", vmin=0, vmax=1)
                axes[position, column].axis("off")
                axes[position, column].set_title(
                    f"{row[f'id_{side}']}\n" f"Pair SSIM={row['ssim']:.4f}",
                    fontsize=9,
                )

        figure.suptitle(f"{group}: {label} sampled pairs")
        figure.tight_layout(rect=(0, 0, 1, 0.97))
        figure.savefig(output / f"{group}_{label}_pairs.png", dpi=160)
        plt.close(figure)


def compute_pair_group(args, group, dataset, cross_patient, seed, output):
    pairs = sample_unique_pairs(
        dataset,
        args.pairs,
        seed,
        cross_patient=cross_patient,
    )

    @lru_cache(maxsize=128)
    def load_image(index):
        return ev.load_ssim_image(dataset, index)

    image = load_image(pairs[0][0])
    identity_ssim = ev.calculate_image_ssim(image, image)

    if abs(identity_ssim - 1.0) > 1e-10:
        raise ValueError("SSIM identity check failed.")

    # Save sampled identities before computing their scores.
    pair_manifest = []

    for pair_index, (i, j) in enumerate(pairs):
        record_a = dataset.rows[i]
        record_b = dataset.rows[j]
        real = dataset.kind == "real"

        pair_manifest.append(
            {
                "group": group,
                "pair_index": pair_index,
                "index_a": i,
                "index_b": j,
                "id_a": dataset.sample_ids[i],
                "id_b": dataset.sample_ids[j],
                "path_a": dataset.relative_paths[i],
                "path_b": dataset.relative_paths[j],
                "subject_a": record_a.get("subject_id", ""),
                "subject_b": record_b.get("subject_id", ""),
                "diagnosis_a": record_a.get("diagnosis", ""),
                "diagnosis_b": record_b.get("diagnosis", ""),
                "same_patient": (
                    str(record_a["subject_id"]) == str(record_b["subject_id"])
                    if real
                    else ""
                ),
                "same_diagnosis": (
                    record_a["diagnosis"] == record_b["diagnosis"] if real else ""
                ),
                # Recorded metadata, not an assertion of anatomical alignment.
                "slice_index_gap": (
                    abs(int(record_a["slice_index"]) - int(record_b["slice_index"]))
                    if real
                    else ""
                ),
            }
        )

    manifest_path = output / f"{group}_pair_manifest.csv"
    write_csv(manifest_path, pair_manifest)

    result_rows = []

    for position, row in enumerate(pair_manifest):
        score = ev.calculate_image_ssim(
            load_image(row["index_a"]),
            load_image(row["index_b"]),
        )
        result_rows.append({**row, "ssim": score})

        if (position + 1) % 500 == 0 or position + 1 == len(pair_manifest):
            print(
                f"{group}: SSIM {position + 1}/{len(pair_manifest)}",
                flush=True,
            )

    load_image.cache_clear()

    write_csv(output / f"{group}_pair_ssim.csv", result_rows)

    save_pair_examples(
        dataset,
        result_rows,
        output,
        group,
        seed,
        args.examples,
    )

    scores = [row["ssim"] for row in result_rows]
    summary = {
        "group": group,
        "pair_seed": seed,
        "cross_patient_only": cross_patient,
        "source_image_count": len(dataset),
        "unique_images_in_pairs": len({index for pair in pairs for index in pair}),
        "pair_manifest_sha256": ev.file_sha256(manifest_path),
        "identity_ssim": identity_ssim,
        "ssim": numeric_summary(scores),
    }

    if dataset.kind == "real":
        summary["same_patient_pair_count"] = sum(
            row["same_patient"] for row in result_rows
        )
        summary["diagnosis_pair_counts"] = dict(
            Counter(
                "/".join(sorted((row["diagnosis_a"], row["diagnosis_b"])))
                for row in result_rows
            )
        )

    return result_rows, summary


def run_diversity(args, provenance, extraction_config):
    output = args.output_dir / "pairwise_ssim"
    output.mkdir(parents=True, exist_ok=True)

    # Revalidates image identities, preprocessing and image-byte hashes
    # against the existing Stage B caches. No Inception rerun.
    _, datasets = ev.load_ssim_datasets(args, provenance, extraction_config)

    specifications = (
        ("real_all", datasets["real"], False),
        ("real_cross_patient", datasets["real"], True),
        ("dcgan", datasets["dcgan"], False),
        ("stylegan2", datasets["stylegan2"], False),
    )

    summaries = {}
    values = {}

    for offset, (group, dataset, cross_patient) in enumerate(specifications):
        rows, summary = compute_pair_group(
            args,
            group,
            dataset,
            cross_patient,
            args.seed + 1000 + offset,
            output,
        )

        summaries[group] = summary
        values[group] = np.array([row["ssim"] for row in rows])

        ev.atomic_write_json(output / "pairwise_ssim_results.json", summaries)

    write_csv(
        output / "pairwise_ssim_summary.csv",
        [
            {
                "group": group,
                "unique_images_in_pairs": item["unique_images_in_pairs"],
                **item["ssim"],
            }
            for group, item in summaries.items()
        ],
    )

    figure, axis = plt.subplots(figsize=(9, 6))

    for group, scores in values.items():
        sorted_scores = np.sort(scores)
        probability = np.arange(1, len(scores) + 1) / len(scores)

        axis.plot(
            sorted_scores,
            probability,
            color=PAIR_COLORS[group],
            label=group,
        )

    axis.set_xlabel("Within-group pairwise SSIM")
    axis.set_ylabel("Empirical cumulative probability")
    axis.set_title("Random distinct image pairs; no nearest-neighbor matching")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    save_figure(figure, output / "pairwise_ssim_ecdf")

    return summaries


def parse_args():
    parser = argparse.ArgumentParser(
        description="Joint t-SNE and within-group pairwise SSIM."
    )

    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--phase",
        choices=("embedding", "diversity", "all"),
        default="all",
    )
    parser.add_argument("--expected-count", type=int, default=5000)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--perplexities", type=float, nargs="+", default=[30.0, 50.0])
    parser.add_argument("--tsne-seeds", type=int, nargs="+", default=[2026])
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--pairs", type=int, default=5000)
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)

    args = parser.parse_args()

    if args.phase in ("diversity", "all") and args.data_root is None:
        parser.error("--data-root is required for pairwise SSIM.")

    if (
        min(
            args.expected_count,
            args.pca_components,
            args.pairs,
            args.examples,
            args.threads,
        )
        < 1
    ):
        parser.error("Counts and threads must be positive.")

    if args.pca_components < 2:
        parser.error("pca-components must be at least 2.")

    if args.iterations < 500:
        parser.error("Use at least 500 t-SNE iterations.")

    if any(not np.isfinite(p) or p <= 0 for p in args.perplexities):
        parser.error("Perplexities must be finite and positive.")

    if args.seed < 0 or any(seed < 0 for seed in args.tsne_seeds):
        parser.error("Seeds must be non-negative.")

    if len(set(args.perplexities)) != len(args.perplexities):
        parser.error("Duplicate perplexities.")

    if len(set(args.tsne_seeds)) != len(args.tsne_seeds):
        parser.error("Duplicate t-SNE seeds.")

    args.cohort_dir = args.cohort_dir.resolve()
    args.cache_dir = args.cohort_dir / "stage_b" / "features"

    if args.data_root is not None:
        args.data_root = args.data_root.resolve()

    args.output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else args.cohort_dir / "stage_c" / "embedding_diversity"
    )

    return args


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)

    features, provenance, extraction_config, reference_split = (
        ev.load_fid_feature_caches(
            args.cache_dir,
            args.expected_count,
        )
    )

    records = {}

    for name in GROUPS:
        payload = torch.load(
            args.cache_dir / f"{name}_inception2048.pt",
            map_location="cpu",
            weights_only=True,
        )
        records[name] = payload["records"]
        del payload

    args.output_dir.mkdir(parents=True, exist_ok=True)

    protocol = {
        "schema_version": 1,
        "script_sha256": ev.file_sha256(Path(__file__)),
        "evaluation_source_sha256": ev.file_sha256(Path(ev.__file__)),
        "feature_cache_sha256": {
            name: provenance[name]["cache_sha256"] for name in GROUPS
        },
        "reference_split": reference_split,
        "count_per_group": args.expected_count,
        "joint_embedding": True,
        "pca_components": args.pca_components,
        "pca_seed": args.seed,
        "pca_whiten": False,
        "feature_standardization": "none",
        "perplexities": args.perplexities,
        "tsne_seeds": args.tsne_seeds,
        "iterations": args.iterations,
        "threads": args.threads,
        "pairs_per_group": args.pairs,
        "pair_seed_base": args.seed + 1000,
        "pair_sampling": (
            "uniform unordered distinct image pairs without replacement; "
            "additional real cross-patient reference"
        ),
        "examples_per_selection": args.examples,
        "ssim": {
            "data_range": 1.0,
            "gaussian_weights": True,
            "sigma": 1.5,
            "win_size": 11,
            "use_sample_covariance": False,
            "region": "full 256x256 image",
            "alignment": "none",
        },
        "versions": {
            package: version(package)
            for package in (
                "numpy",
                "scipy",
                "scikit-learn",
                "scikit-image",
                "matplotlib",
                "torch",
                "Pillow",
                "threadpoolctl",
            )
        },
        "interpretation_limits": [
            "t-SNE is exploratory; 2D distances and areas are not coverage metrics.",
            "Generated images have no diagnosis ground-truth labels.",
            "Random-pair SSIM depends on slice position and black background.",
            "Lower pairwise SSIM can reflect artifacts, not useful diversity.",
            "Pairs reuse images and patients; summary SD is not a confidence interval.",
        ],
    }

    protocol_path = args.output_dir / "protocol.json"

    if protocol_path.exists():
        if ev.read_json(protocol_path) != protocol:
            raise ValueError(
                "Settings, source code, environment or inputs changed. "
                "Use a new --output-dir to preserve the earlier experiment."
            )
    else:
        ev.atomic_write_json(protocol_path, protocol)

    requested = ("embedding", "diversity") if args.phase == "all" else (args.phase,)

    for phase in requested:
        started = time.perf_counter()
        status = {
            "phase": phase,
            "status": "running",
            "started_utc": ev.utc_now(),
        }
        status_path = args.output_dir / f"{phase}_status.json"
        ev.atomic_write_json(status_path, status)

        try:
            if phase == "embedding":
                run_embedding(args, features, records)
            else:
                run_diversity(args, provenance, extraction_config)

            status["status"] = "complete"
        except Exception as error:
            status["status"] = "failed"
            status["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            status["finished_utc"] = ev.utc_now()
            status["elapsed_seconds"] = time.perf_counter() - started
            ev.atomic_write_json(status_path, status)

    print(f"\nCompleted: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
