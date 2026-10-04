import sys
import json
import re
import random
import math
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from dataclasses import dataclass, asdict
from pathlib import Path
from collections import Counter
from PIL import Image, ImageOps


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

        count_before = len(records)

        for image_path in sorted(class_dir.iterdir()):
            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in {".jpg", ".jpeg"}:
                continue

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

        if len(records) == count_before:
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
            print(
                f"Patient leakage between {left} and {right}: "
                f"{sorted(shared_patients)}",
                file=sys.stderr,
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


def create_or_load_splits(
    data_root,
    metadata_path,
    manifest_path,
    included_classes=("AD",),
    val_ratio=0.2,
    seed=42,
):
    data_root = Path(data_root)
    metadata_path = Path(metadata_path)
    manifest_path = Path(manifest_path)

    if not isinstance(included_classes, (tuple, list)) or not included_classes:
        raise ValueError("included_classes must be a non-empty tuple or list.")

    if any(name not in ("AD", "NC") for name in included_classes):
        raise ValueError("included_classes can only contain AD and NC.")

    if len(included_classes) != len(set(included_classes)):
        raise ValueError("included_classes contains duplicate classes.")

    if not 0 < val_ratio < 1:
        raise ValueError(f"Invalid validation ratio: {val_ratio!r}")

    if type(seed) is not int:
        raise ValueError("seed must be an integer.")

    included_classes = tuple(sorted(included_classes))

    config = {
        "included_classes": list(included_classes),
        "val_ratio": val_ratio,
        "seed": seed,
    }

    subject_mapping = load_subject_mapping(metadata_path)

    source_train = collect_record(data_root, "train", included_classes, subject_mapping)

    source_test = collect_record(data_root, "test", included_classes, subject_mapping)

    loaded = manifest_path.exists()

    if loaded:
        with manifest_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        if not isinstance(payload, dict):
            raise ValueError("The split manifest must be a dictionary.")

        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported split manifest version.")

        if payload.get("config") != config:
            raise ValueError(
                "The saved split configuration does not match this run. "
                "Use the original configuration or a different manifest_path."
            )

        saved_splits = payload.get("splits")

        if not isinstance(saved_splits, dict) or set(saved_splits) != {
            "train",
            "validation",
            "test",
        }:
            raise ValueError("The manifest must contain train, validation and test.")

        splits = {}

        for name, items in saved_splits.items():
            if not isinstance(items, list):
                raise ValueError(f"{name} must contain a list of records.")

            records = []

            for item in items:
                if not isinstance(item, dict):
                    raise ValueError(f"Invalid record in {name}")

                records.append(SliceRecord(**item))

            splits[name] = records
    else:
        splits = split_by_patient(source_train, val_ratio, seed)

        splits["test"] = source_test.copy()

    summary = validate_splits(splits, source_train, source_test)

    if not loaded:
        payload = {
            "schema_version": 1,
            "config": config,
            "splits": {
                name: [asdict(record) for record in records]
                for name, records in splits.items()
            },
        }

        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        with manifest_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")

    action = "Loaded" if loaded else "Created"
    print(f"{action} split manifest: {manifest_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    return splits


def preprocess_image(image):
    image = image.convert("L")

    width, height = image.size
    side = max(width, height)

    width_padding = side - width
    height_padding = side - height

    left = width_padding // 2
    right = width_padding - left

    top = height_padding // 2
    bottom = height_padding - top

    # padding
    image = ImageOps.expand(image, border=(left, top, right, bottom), fill=0)

    # convert to array [-1, 1]
    pixels = np.array(image, dtype=np.float32)
    pixels = (pixels - 127.5) / 127.5

    # convert to tensor, adding C = 1 --> [C, H, W]
    tensor = torch.from_numpy(pixels).unsqueeze(0)

    return tensor


class ADNIDataset(Dataset):
    def __init__(
        self, records, data_root="/home/groups/comp3710/ADNI/AD_NC", image_size=256
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.records = tuple(records)
        self.image_size = image_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image_path = self.data_root / record.relative_path

        with Image.open(image_path) as image:
            tensor = preprocess_image(image)

        valid_shape = torch.Size([1, self.image_size, self.image_size])

        if tensor.shape != valid_shape:
            raise ValueError(
                f"Invalid image shape for {image_path}: got {tuple(tensor.shape)}, wanted {valid_shape}."
            )

        return tensor


def build_dataloaders(
    data_root,
    metadata_path,
    manifest_path,
    included_classes=("AD",),
    val_ratio=0.2,
    split_seed=42,
    image_size=256,
    batch_size=16,
    num_workers=0,
    pin_memory=False,
    loader_seed=42,
    drop_last_train=True,
):
    splits = create_or_load_splits(
        data_root=data_root,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        included_classes=included_classes,
        val_ratio=val_ratio,
        seed=split_seed,
    )

    train_dataset = ADNIDataset(
        data_root=data_root, records=splits["train"], image_size=image_size
    )
    if drop_last_train and len(train_dataset) < batch_size:
        raise RuntimeError(
            f"Training dataset contains {len(train_dataset)} images, "
            f"fewer than batch_size={batch_size}; "
            "drop_last_train=True would produce no batches."
        )

    test_dataset = ADNIDataset(
        data_root=data_root, records=splits["test"], image_size=image_size
    )

    val_dataset = ADNIDataset(
        data_root=data_root, records=splits["validation"], image_size=image_size
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(loader_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=drop_last_train,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    return (
        train_dataset,
        train_loader,
        test_dataset,
        test_loader,
        val_dataset,
        val_loader,
    )
