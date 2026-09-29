import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
import json
import re
from pathlib import Path
import random
import math
from collections import Counter


@dataclass(frozen=True)
class SliceRecord:
    relative_path: str
    subject_id: str
    image_id: str
    slice_index: int
    diagnosis: str
    source_split: str


# Map each MIR image ID to each patient ID using using metadata path strings in the meta_data_with_label.json file.
def load_subject_mapping(metadata_path):
    metadata_path = Path(metadata_path)

    with metadata_path.open("r", encoding="utf-8-sig") as file:
        metadata = json.load(file)

    if not isinstance(metadata, dict) or not metadata:
        raise ValueError("Metadata must be a non-empty dictionary.")

    # PTID is site_S_RID; capture only the four- or five-digit patient RID.
    subject_pattern = re.compile(r"(?<!\d)\d{3}_S_(\d{4,5})(?!\d)")
    fields = ("c1", "c2", "c3", "c4", "c5", "raw", "masked")
    subject_mapping = {}

    for image_id, record in metadata.items():
        if not isinstance(record, dict):
            raise ValueError(f"Invalid metadata record for image {image_id}.")

        subject_ids = set()

        for field in fields:
            if field not in record:
                continue
            path_text = record[field]
            if not isinstance(path_text, str):
                raise ValueError(f"Image {image_id}: {field} must be a string.")

            matches = set(subject_pattern.findall(path_text))

            if len(matches) != 1:
                raise ValueError(
                    f"Image {image_id}: cannot identify one patient in {field}."
                )

            subject_ids.update(matches)

        if len(subject_ids) != 1:
            raise ValueError(
                f"Image {image_id}: missing or conflicting patient IDs: "
                f"{sorted(subject_ids)}"
            )

        subject_mapping[image_id] = next(iter(subject_ids))

    return subject_mapping


# Collect JPEG slice records and associate them with patient IDs.
def collect_record(data_root, source_split, included_classes, subject_mapping):
    if source_split not in ("train", "test"):
        raise ValueError(f"The source {source_split!r} dose not exit.")

    for diagnosis in included_classes:
        if diagnosis not in ("AD", "NC"):
            raise ValueError(f"Unknown diagnosis folder: {diagnosis!r}.")

    filename_pattern = re.compile(r"(\d+)_(\d+)")
    records = []
    seen_slices = set()

    for diagnosis in included_classes:
        class_dir = data_root / source_split / diagnosis

        if not class_dir.is_dir():
            raise FileNotFoundError(f"Directory dose not exit: {class_dir}")

        for image_path in sorted(class_dir.iterdir()):
            match = filename_pattern.fullmatch(image_path.stem)
            if match is None:
                raise ValueError(f"Unexpected JPEG filename: {image_path.name}")
            image_id = match.group(1)
            slice_index = int(match.group(2))

            if image_id not in subject_mapping:
                raise ValueError(
                    f"No patient mapping for image {image_id}: {image_path}"
                )

            subject_id = subject_mapping[image_id]

            slice_key = (image_id, slice_index)

            if slice_key in seen_slices:
                raise ValueError(
                    f"Duplicate image/slice pair {slice_key}: {image_path}"
                )

            seen_slices.add(slice_key)

            records.append(
                SliceRecord(
                    relative_path=image_path.relative_to(data_root).as_posix(),
                    subject_id=subject_id,
                    image_id=image_id,
                    slice_index=slice_index,
                    diagnosis=diagnosis,
                    source_split=source_split,
                )
            )

        if len(records) == 0:
            raise RuntimeError(f"No JPEG images found in: {class_dir}")

    records.sort(
        key=lambda record: (record.subject_id, record.image_id, record.slice_index)
    )

    return records


def split_by_patient(records, val_ratio, seed=42):
    if any(r.source_split != "train" for r in records):
        raise ValueError(
            "split_by_patient only accepts records from the official train split."
        )
    if not 0 < val_ratio < 1:
        raise ValueError(f"Invalid validation ratio: {val_ratio!r}")

    subject_ids = sorted({r.subject_id for r in records})

    n = len(subject_ids)
    if n < 2:
        raise ValueError("At least two patients!")

    rng = random.Random(seed)
    rng.shuffle(subject_ids)
    n_val = max(1, min(n - 1, math.floor(n * val_ratio + 0.5)))

    val_subjects = set(subject_ids[:n_val])

    train_records = [r for r in records if r.subject_id not in val_subjects]

    val_records = [r for r in records if r.subject_id in val_subjects]

    return {"train": train_records, "validation": val_records}


# Validate patient isolation and preservation of original records.
def validate_splits(splits, source_train, source_test):
    expected_names = {"train", "validation", "test"}
    if set(splits) != expected_names:
        raise ValueError("splits must contain exactly: train, validation, test.")

    groups = {
        "source_train": (source_train, "train"),
        "source_test": (source_test, "test"),
        "train": (splits["train"], "train"),
        "validation": (splits["validation"], "train"),
        "test": (splits["test"], "test"),
    }

    for name, (records, expected_source) in groups.items():
        if not records:
            raise ValueError(f"{name} contains no records.")

        slice_keys = [(r.image_id, r.slice_index) for r in records]

        if len(slice_keys) != len(set(slice_keys)):
            raise ValueError(f"{name} contains duplicate image/slice pairs.")

        paths = [r.relative_path for r in records]

        if len(paths) != len(set(paths)):
            raise ValueError(f"{name} contains duplicate image paths.")

        for record in records:
            if not record.subject_id:
                raise ValueError(
                    f"{name}: missing patient ID for " f"{record.relative_path}."
                )

            if record.source_split != expected_source:
                raise ValueError(
                    f"{name}: {record.relative_path} has source_split="
                    f"{record.source_split!r}; "
                    f"expected {expected_source!r}."
                )

    patient_sets = {
        name: {r.subject_id for r in records} for name, records in splits.items()
    }

    slice_sets = {
        name: {(r.image_id, r.slice_index) for r in records}
        for name, records in splits.items()
    }

    pairs = (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    )

    for left, right in pairs:
        shared_patients = patient_sets[left] & patient_sets[right]

        if shared_patients:
            raise ValueError(
                f"Patient leakage between {left} and {right}: "
                f"{sorted(shared_patients)}"
            )

        shared_slices = slice_sets[left] & slice_sets[right]

        if shared_slices:
            raise ValueError(
                f"Shared image slices between {left} and {right}: "
                f"{sorted(shared_slices)}"
            )

    combined_train = splits["train"] + splits["validation"]

    if Counter(combined_train) != Counter(source_train):
        raise ValueError(
            "train + validation must contain exactly " "the original training records."
        )

    if Counter(splits["test"]) != Counter(source_test):
        raise ValueError("test must contain exactly the original test records.")

    summary = {}

    for name, records in splits.items():
        diagnoses = sorted({r.diagnosis for r in records})

        summary[name] = {
            "patients": len(patient_sets[name]),
            "scans": len({r.image_id for r in records}),
            "slices": len(records),
            "patients_by_diagnosis": {
                diagnosis: len(
                    {r.subject_id for r in records if r.diagnosis == diagnosis}
                )
                for diagnosis in diagnoses
            },
            "slices_by_diagnosis": dict(Counter(r.diagnosis for r in records)),
        }

    return summary
