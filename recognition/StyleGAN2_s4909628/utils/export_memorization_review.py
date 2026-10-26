import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import evaluation as ev


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError("No review rows.")

    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_images(name, root, rows, id_field, kind):
    """
    Reuse the exact image loading and preprocessing from evaluation.py.
    Only the selected rows are instantiated.
    """
    spec = {
        "root": root,
        "rows": rows,
        "id_field": id_field,
        "manifest_sha256": ev.object_sha256(rows),
        "kind": kind,
    }
    dataset = ev.EvaluationImageDataset(name, spec)

    return [ev.load_ssim_image(dataset, index) for index in range(len(dataset))]


def save_page(sample_id, generated, references, pairs, output_path):
    columns = len(pairs) + 1

    figure, axes = plt.subplots(
        2,
        columns,
        figsize=(3.2 * columns, 7.0),
        squeeze=False,
    )

    axes[0, 0].imshow(generated, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title(f"Generated\n{sample_id}", fontsize=10)

    axes[1, 0].text(
        0.02,
        0.95,
        "Top row: original images\n"
        "Bottom row: absolute differences\n\n"
        "No registration\n"
        "No brightness adjustment\n"
        "Difference scale fixed at [0, 1]\n\n"
        "Inspect all candidates.\n"
        "A review flag is not proof\n"
        "of memorization.",
        transform=axes[1, 0].transAxes,
        va="top",
        fontsize=10,
    )

    for column, (reference, pair) in enumerate(zip(references, pairs), start=1):
        axes[0, column].imshow(reference, cmap="gray", vmin=0, vmax=1)

        axes[0, column].set_title(
            f"Feature rank {pair['feature_rank']}\n"
            f"Patient {pair['train_subject_id']} | "
            f"scan {pair['train_image_id']}\n"
            f"Slice {pair['train_slice_index']}",
            fontsize=9,
        )

        difference = np.abs(generated - reference)

        axes[1, column].imshow(difference, cmap="magma", vmin=0, vmax=1)

        support_mae = pair["mae_support"]
        support_text = f"{float(support_mae):.4f}" if support_mae else "N/A"

        axes[1, column].set_title(
            f"d={float(pair['feature_distance']):.4f}\n"
            f"SSIM={float(pair['ssim']):.4f}\n"
            f"Support MAE={support_text}",
            fontsize=9,
        )

    for axis in axes.flat:
        axis.axis("off")

    figure.suptitle(
        f"Training-neighbor review: {sample_id}",
        fontsize=14,
    )

    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=("dcgan", "stylegan2"),
        default="stylegan2",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    cohort = args.cohort_dir.resolve()
    data_root = args.data_root.resolve()
    audit_dir = cohort / "stage_c" / "memorization"
    model_dir = audit_dir / args.model

    audit = ev.read_json(audit_dir / "audit_results.json")

    if audit.get("status") != "complete":
        raise ValueError("The memorization audit is not complete.")

    # Verify that current Stage A metadata matches this audit.
    ev.verify_hash(
        cohort / "protocol.json",
        audit["protocol"]["stage_a_protocol_sha256"],
        "Stage A protocol",
    )

    # Real images must use the same preprocessing implementation.
    stage_a = ev.read_json(cohort / "protocol.json")
    ev.verify_hash(
        Path(ev.__file__).resolve().parent / "dataset.py",
        stage_a["source_hashes"]["dataset.py"],
        "dataset.py",
    )

    # Existing helper checks the Stage A manifests.
    _, specs = ev.load_evaluation_inputs(cohort, data_root)

    snapshot_path = cohort / "source_splits.json"
    ev.verify_hash(
        snapshot_path,
        stage_a["real_reference"]["source_manifest_sha256"],
        "source split snapshot",
    )
    training_records = ev.read_json(snapshot_path)["splits"]["train"]

    query_path = model_dir / "per_generated_image.csv"
    candidate_path = model_dir / "candidate_pairs.csv"

    queries = read_csv(query_path)
    candidates = read_csv(candidate_path)

    flagged = [row for row in queries if row["review_flag"].strip().lower() == "true"]
    flagged.sort(key=lambda row: float(row["nn_feature_distance"]))

    expected_flags = audit["models"][args.model]["review_flag_count"]

    if len(flagged) != expected_flags:
        raise ValueError("Flag count differs from audit_results.json.")

    if not flagged:
        print(f"{args.model}: no flagged samples to export.")
        return

    if len({row["sample_id"] for row in flagged}) != len(flagged):
        raise ValueError("Duplicate flagged sample IDs.")

    grouped = defaultdict(list)
    for row in candidates:
        grouped[row["sample_id"]].append(row)

    top_k = int(audit["protocol"]["top_k"])
    generated_spec = specs[args.model]

    # Validate identities before creating output.
    plans = []

    for query in flagged:
        sample_id = query["sample_id"]
        query_index = int(query["query_index"])
        generated_record = generated_spec["rows"][query_index]

        if generated_record["sample_id"] != sample_id:
            raise ValueError(f"Generated identity mismatch: {sample_id}")

        pairs = sorted(
            grouped[sample_id],
            key=lambda row: int(row["feature_rank"]),
        )

        if [int(row["feature_rank"]) for row in pairs] != list(range(1, top_k + 1)):
            raise ValueError(f"Missing/duplicate candidates: {sample_id}")

        selected_train = []

        for pair in pairs:
            if pair["model"] != args.model:
                raise ValueError("Candidate model mismatch.")

            if int(pair["query_index"]) != query_index:
                raise ValueError("Candidate query index mismatch.")

            if pair["generated_relative_path"] != generated_record["relative_path"]:
                raise ValueError("Generated path mismatch.")

            train_index = int(pair["train_index"])
            record = dict(training_records[train_index])

            for csv_key, record_key in (
                ("train_relative_path", "relative_path"),
                ("train_subject_id", "subject_id"),
                ("train_image_id", "image_id"),
                ("train_slice_index", "slice_index"),
            ):
                if str(pair[csv_key]) != str(record[record_key]):
                    raise ValueError(f"Training identity mismatch: {sample_id}")

            record["reference_id"] = f"audit_train_{train_index:06d}"
            selected_train.append(record)

        plans.append((query, generated_record, selected_train, pairs))

    output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else model_dir / "manual_review"
    )

    # Do not overwrite an already-filled manual review table.
    if output.exists():
        raise FileExistsError(
            f"Output already exists: {output}\n"
            "Use another --output-dir to preserve existing review notes."
        )

    pages_dir = output / "pages"
    pages_dir.mkdir(parents=True)

    review_rows = []

    for position, (query, generated_record, train_rows, pairs) in enumerate(
        plans, start=1
    ):
        sample_id = query["sample_id"]

        generated = load_images(
            "review_generated",
            generated_spec["root"],
            [generated_record],
            "sample_id",
            "generated",
        )[0]

        references = load_images(
            "review_train",
            data_root,
            train_rows,
            "reference_id",
            "real",
        )

        page_name = f"{position:03d}_{sample_id}.png"
        save_page(
            sample_id,
            generated,
            references,
            pairs,
            pages_dir / page_name,
        )

        for pair in pairs:
            review_rows.append(
                {
                    "page": f"pages/{page_name}",
                    "model": args.model,
                    "sample_id": sample_id,
                    "feature_rank": pair["feature_rank"],
                    "train_subject_id": pair["train_subject_id"],
                    "train_image_id": pair["train_image_id"],
                    "train_slice_index": pair["train_slice_index"],
                    "train_relative_path": pair["train_relative_path"],
                    "feature_distance": pair["feature_distance"],
                    "ssim": pair["ssim"],
                    "mae_support": pair["mae_support"],
                    "reviewed": "",
                    "assessment": "",
                    "observed_similarities": "",
                    "observed_differences": "",
                    "follow_up": "",
                    "reviewer": "",
                    "review_date": "",
                }
            )

        print(
            f"Exported {position}/{len(plans)}: {page_name}",
            flush=True,
        )

    write_csv(output / "manual_review.csv", review_rows)

    ev.atomic_write_json(
        output / "export_summary.json",
        {
            "status": "complete",
            "created_utc": ev.utc_now(),
            "model": args.model,
            "flagged_samples": len(plans),
            "candidate_pairs": len(review_rows),
            "query_csv_sha256": ev.file_sha256(query_path),
            "candidate_csv_sha256": ev.file_sha256(candidate_path),
            "audit_results_sha256": ev.file_sha256(audit_dir / "audit_results.json"),
            "script_sha256": ev.file_sha256(Path(__file__)),
            "manual_review_completed": False,
            "allowed_assessments": [
                "no_obvious_copy",
                "uncertain",
                "suspected_copy",
            ],
        },
    )

    print(f"\nExport complete: {output}")
    print("Fill manual_review.csv after visually inspecting every pair.")


if __name__ == "__main__":
    main()
