"""Validate and split a YOLO detection dataset safely.

Expected raw layout (nested folders are supported):

    dataset/
      images/
        example.jpg
      labels/
        example.txt

Each label line must be: class_id x_center y_center width height
with normalized coordinates in [0, 1]. Empty label files are accepted as
explicit negative images. Missing label files are errors by default.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import yaml
from PIL import Image, UnidentifiedImageError

LOGGER = logging.getLogger("prepare_dataset")
SCRIPT_DIR = Path(__file__).resolve().parent

SPLIT_NAMES = ("train", "val", "test")
IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


class DatasetValidationError(ValueError):
    """Raised when the source dataset is unsafe or invalid."""


@dataclass(frozen=True, slots=True)
class Sample:
    """A fully validated image/label pair."""

    image_path: Path
    label_path: Path | None
    relative_image: Path
    width: int
    height: int
    class_ids: tuple[int, ...]

    @property
    def relative_label(self) -> Path:
        return self.relative_image.with_suffix(".txt")

    @property
    def class_signature(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.class_ids)))

    @property
    def box_count(self) -> int:
        return len(self.class_ids)


@dataclass(frozen=True, slots=True)
class SplitRatios:
    train: float
    val: float
    test: float

    def as_dict(self) -> dict[str, float]:
        return {"train": self.train, "val": self.val, "test": self.test}

    def validate(self) -> None:
        values = self.as_dict()
        if any(value < 0 for value in values.values()):
            raise DatasetValidationError("Split ratios cannot be negative.")
        if not math.isclose(sum(values.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise DatasetValidationError(
                f"Split ratios must sum to 1.0; got {sum(values.values()):.12g}."
            )
        if self.train <= 0 or self.val <= 0 or self.test <= 0:
            raise DatasetValidationError(
                "Train, validation, and test ratios must all be greater than zero."
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate, stratify, and safely build a YOLO dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--images", type=Path, default=SCRIPT_DIR / "dataset/images")
    parser.add_argument("--labels", type=Path, default=SCRIPT_DIR / "dataset/labels")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "data")
    parser.add_argument("--data-yaml", type=Path, default=SCRIPT_DIR / "data.yaml")
    parser.add_argument("--train", type=float, default=0.70, dest="train_ratio")
    parser.add_argument("--val", type=float, default=0.20, dest="val_ratio")
    parser.add_argument("--test", type=float, default=0.10, dest="test_ratio")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--missing-label",
        choices=("error", "empty", "skip"),
        default="error",
        help=(
            "How to handle an image without a label file. Use 'empty' only when "
            "the image was reviewed and is truly a negative example."
        ),
    )
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
        help="How source files are placed in the prepared dataset.",
    )
    parser.add_argument(
        "--group-regex",
        default=None,
        help=(
            "Optional regex applied to each relative image path. Matching files "
            "with the same first capture group stay in one split. This prevents "
            "frame leakage when images come from the same video or sequence."
        ),
    )
    parser.add_argument(
        "--check-duplicates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Hash images and reject exact duplicate file content.",
    )
    parser.add_argument(
        "--verify-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fully decode every image before copying it.",
    )
    parser.add_argument(
        "--strict-class-coverage",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail if any split contains no example of one of the configured classes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output dataset only after a new build succeeds.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(levelname)s: %(message)s",
    )


def load_class_names(data_yaml: Path) -> dict[int, str]:
    data_yaml = data_yaml.expanduser().resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    with data_yaml.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    names = raw.get("names")
    if isinstance(names, list):
        result = {index: str(name) for index, name in enumerate(names)}
    elif isinstance(names, dict):
        try:
            result = {int(index): str(name) for index, name in names.items()}
        except (TypeError, ValueError) as exc:
            raise DatasetValidationError(
                f"The 'names' keys in {data_yaml} must be integer class IDs."
            ) from exc
    else:
        raise DatasetValidationError(
            f"{data_yaml} must contain a 'names' list or mapping."
        )

    if not result:
        raise DatasetValidationError("At least one class name is required.")

    expected = list(range(len(result)))
    if sorted(result) != expected:
        raise DatasetValidationError(
            f"Class IDs must be contiguous from 0 to {len(result) - 1}; "
            f"got {sorted(result)}."
        )
    return dict(sorted(result.items()))


def discover_images(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")

    images = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise DatasetValidationError(f"No supported images found under {root}")
    return images


def ensure_unique_relative_stems(images: Iterable[Path], image_root: Path) -> None:
    by_stem: dict[str, list[Path]] = defaultdict(list)
    for image in images:
        relative = image.relative_to(image_root)
        key = relative.with_suffix("").as_posix().casefold()
        by_stem[key].append(image)

    collisions = [paths for paths in by_stem.values() if len(paths) > 1]
    if collisions:
        details = "\n".join(
            "  - " + ", ".join(str(path) for path in group) for group in collisions[:20]
        )
        raise DatasetValidationError(
            "Multiple images map to the same YOLO label path. Rename or remove "
            f"the collisions:\n{details}"
        )


def validate_image(path: Path, fully_decode: bool) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            width, height = image.size
            if fully_decode:
                image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise DatasetValidationError(f"Unreadable image {path}: {exc}") from exc

    if width < 2 or height < 2:
        raise DatasetValidationError(
            f"Image dimensions must be at least 2x2; got {width}x{height}: {path}"
        )
    return width, height


def parse_label_file(path: Path, class_names: Mapping[int, str]) -> tuple[int, ...]:
    class_ids: list[int] = []
    seen_boxes: set[tuple[int, float, float, float, float]] = set()

    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise DatasetValidationError(
            f"Could not read label file {path}: {exc}"
        ) from exc

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) != 5:
            raise DatasetValidationError(
                f"{path}:{line_number}: expected 5 values "
                "(class x_center y_center width height), "
                f"got {len(parts)}."
            )

        try:
            class_value = float(parts[0])
            coordinates = tuple(float(value) for value in parts[1:])
        except ValueError as exc:
            raise DatasetValidationError(
                f"{path}:{line_number}: every value must be numeric."
            ) from exc

        if not class_value.is_integer():
            raise DatasetValidationError(
                f"{path}:{line_number}: class ID must be an integer, got {parts[0]!r}."
            )
        class_id = int(class_value)
        if class_id not in class_names:
            raise DatasetValidationError(
                f"{path}:{line_number}: class ID {class_id} is not configured; "
                f"valid IDs are {list(class_names)}."
            )

        if not all(math.isfinite(value) for value in coordinates):
            raise DatasetValidationError(
                f"{path}:{line_number}: coordinates must be finite numbers."
            )

        x_center, y_center, width, height = coordinates
        if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
            raise DatasetValidationError(
                f"{path}:{line_number}: box center must be within [0, 1]."
            )
        if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            raise DatasetValidationError(
                f"{path}:{line_number}: box width and height must be in (0, 1]."
            )

        epsilon = 1e-6
        x1 = x_center - width / 2.0
        y1 = y_center - height / 2.0
        x2 = x_center + width / 2.0
        y2 = y_center + height / 2.0
        if x1 < -epsilon or y1 < -epsilon or x2 > 1.0 + epsilon or y2 > 1.0 + epsilon:
            raise DatasetValidationError(
                f"{path}:{line_number}: box extends outside the image after converting "
                "from center/size coordinates."
            )

        canonical = (class_id, x_center, y_center, width, height)
        if canonical in seen_boxes:
            LOGGER.warning("Duplicate box in %s at line %d", path, line_number)
        seen_boxes.add(canonical)
        class_ids.append(class_id)

    return tuple(class_ids)


def build_samples(
    images: Sequence[Path],
    image_root: Path,
    label_root: Path,
    class_names: Mapping[int, str],
    missing_label_policy: str,
    verify_images: bool,
) -> tuple[list[Sample], list[Path], list[Path]]:
    samples: list[Sample] = []
    missing_labels: list[Path] = []
    skipped_images: list[Path] = []

    for index, image_path in enumerate(images, start=1):
        if index == 1 or index % 250 == 0 or index == len(images):
            LOGGER.info("Validating image %d/%d", index, len(images))

        relative_image = image_path.relative_to(image_root)
        label_path = label_root / relative_image.with_suffix(".txt")
        width, height = validate_image(image_path, verify_images)

        if label_path.is_file():
            class_ids = parse_label_file(label_path, class_names)
            effective_label_path: Path | None = label_path
        else:
            missing_labels.append(label_path)
            if missing_label_policy == "error":
                continue
            if missing_label_policy == "skip":
                skipped_images.append(image_path)
                continue
            class_ids = ()
            effective_label_path = None

        samples.append(
            Sample(
                image_path=image_path,
                label_path=effective_label_path,
                relative_image=relative_image,
                width=width,
                height=height,
                class_ids=class_ids,
            )
        )

    if missing_labels and missing_label_policy == "error":
        preview = "\n".join(f"  - {path}" for path in missing_labels[:25])
        extra = (
            ""
            if len(missing_labels) <= 25
            else f"\n  ... and {len(missing_labels) - 25} more"
        )
        raise DatasetValidationError(
            f"Found {len(missing_labels)} image(s) without label files. "
            "This is an error because an absent annotation is not automatically a "
            f"negative image.\n{preview}{extra}\n"
            "Create an empty .txt file for each reviewed negative image, or explicitly "
            "use --missing-label empty/skip."
        )

    if not samples:
        raise DatasetValidationError(
            "No usable image/label pairs remain after validation."
        )
    return samples, missing_labels, skipped_images


def find_orphan_labels(
    label_root: Path,
    samples: Sequence[Sample],
    image_root: Path,
) -> list[Path]:
    del image_root  # Kept in the signature for readability at the call site.
    expected = {sample.relative_label.as_posix().casefold() for sample in samples}
    return sorted(
        label
        for label in label_root.rglob("*.txt")
        if label.relative_to(label_root).as_posix().casefold() not in expected
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def reject_exact_duplicates(samples: Sequence[Sample]) -> None:
    by_hash: dict[str, list[Sample]] = defaultdict(list)
    for index, sample in enumerate(samples, start=1):
        if index == 1 or index % 250 == 0 or index == len(samples):
            LOGGER.info("Hashing image %d/%d", index, len(samples))
        by_hash[sha256_file(sample.image_path)].append(sample)

    duplicates = [group for group in by_hash.values() if len(group) > 1]
    if not duplicates:
        return

    details: list[str] = []
    for group in duplicates[:20]:
        labels_match = len({sample.class_ids for sample in group}) == 1
        status = "same labels" if labels_match else "CONFLICTING labels"
        paths = ", ".join(str(sample.image_path) for sample in group)
        details.append(f"  - [{status}] {paths}")

    raise DatasetValidationError(
        f"Found {sum(len(group) - 1 for group in duplicates)} exact duplicate image(s). "
        "Remove them or keep each duplicate set in a single source record before "
        "splitting:\n" + "\n".join(details)
    )


def allocate_target_counts(total: int, ratios: Mapping[str, float]) -> dict[str, int]:
    raw = {split: total * ratios[split] for split in SPLIT_NAMES}
    counts = {split: math.floor(raw[split]) for split in SPLIT_NAMES}
    remaining = total - sum(counts.values())

    order = sorted(
        SPLIT_NAMES,
        key=lambda split: (raw[split] - counts[split], ratios[split]),
        reverse=True,
    )
    for split in order[:remaining]:
        counts[split] += 1

    if total >= len(SPLIT_NAMES):
        for empty_split in [split for split in SPLIT_NAMES if counts[split] == 0]:
            donor = max(SPLIT_NAMES, key=lambda split: counts[split])
            if counts[donor] <= 1:
                break
            counts[donor] -= 1
            counts[empty_split] += 1
    return counts


def stratified_split(
    samples: Sequence[Sample],
    ratios: Mapping[str, float],
    seed: int,
) -> dict[str, list[Sample]]:
    """Split individual samples exactly while balancing class-presence signatures."""

    rng = random.Random(seed)
    targets = allocate_target_counts(len(samples), ratios)
    remaining = targets.copy()
    result = {split: [] for split in SPLIT_NAMES}

    buckets: dict[tuple[int, ...], list[Sample]] = defaultdict(list)
    for sample in samples:
        buckets[sample.class_signature].append(sample)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    assigned_by_signature: Counter[tuple[tuple[int, ...], str]] = Counter()
    categories = sorted(
        buckets, key=lambda signature: (len(buckets[signature]), signature)
    )

    for signature in categories:
        bucket = buckets[signature]
        desired = {split: len(bucket) * ratios[split] for split in SPLIT_NAMES}
        for sample in bucket:
            candidates = [split for split in SPLIT_NAMES if remaining[split] > 0]
            if not candidates:
                raise RuntimeError(
                    "Internal split allocation error: no capacity remains."
                )

            rng.shuffle(candidates)

            def score(split: str) -> tuple[float, float]:
                signature_deficit = (
                    desired[split] - assigned_by_signature[(signature, split)]
                )
                capacity_fraction = remaining[split] / max(targets[split], 1)
                return (signature_deficit, capacity_fraction)

            chosen = max(candidates, key=score)
            result[chosen].append(sample)
            remaining[chosen] -= 1
            assigned_by_signature[(signature, chosen)] += 1

    if any(remaining.values()):
        raise RuntimeError(
            f"Internal split allocation error: remaining capacity {remaining}"
        )
    for files in result.values():
        rng.shuffle(files)
    return result


def compile_group_regex(pattern: str | None) -> re.Pattern[str] | None:
    if pattern is None:
        return None
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise DatasetValidationError(f"Invalid --group-regex: {exc}") from exc
    return compiled


def group_id_for_sample(sample: Sample, pattern: re.Pattern[str]) -> str:
    relative = sample.relative_image.as_posix()
    match = pattern.search(relative)
    if match is None:
        raise DatasetValidationError(
            f"--group-regex did not match image path: {relative}"
        )
    return match.group(1) if match.lastindex else match.group(0)


def grouped_split(
    samples: Sequence[Sample],
    ratios: Mapping[str, float],
    seed: int,
    pattern: re.Pattern[str],
) -> dict[str, list[Sample]]:
    """Keep related samples together while greedily balancing size and classes."""

    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[group_id_for_sample(sample, pattern)].append(sample)

    if len(groups) < len(SPLIT_NAMES):
        raise DatasetValidationError(
            f"--group-regex produced only {len(groups)} group(s); at least "
            f"{len(SPLIT_NAMES)} are needed for train/val/test."
        )

    rng = random.Random(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)

    global_signature_counts = Counter(sample.class_signature for sample in samples)
    target_images = {split: len(samples) * ratios[split] for split in SPLIT_NAMES}
    target_signatures = {
        (signature, split): count * ratios[split]
        for signature, count in global_signature_counts.items()
        for split in SPLIT_NAMES
    }

    def rarity(item: tuple[str, list[Sample]]) -> tuple[float, int]:
        _, group_samples = item
        score = sum(
            1.0 / global_signature_counts[s.class_signature] for s in group_samples
        )
        return (score, len(group_samples))

    group_items.sort(key=rarity, reverse=True)

    split_groups: dict[str, list[tuple[str, list[Sample]]]] = {
        split: [] for split in SPLIT_NAMES
    }
    image_counts = Counter({split: 0 for split in SPLIT_NAMES})
    signature_counts: Counter[tuple[tuple[int, ...], str]] = Counter()

    for group_index, item in enumerate(group_items):
        group_name, group_samples = item
        group_signature_counts = Counter(s.class_signature for s in group_samples)
        remaining_groups = len(group_items) - group_index
        empty_splits = [split for split in SPLIT_NAMES if not split_groups[split]]
        candidates = list(SPLIT_NAMES)
        if empty_splits and remaining_groups == len(empty_splits):
            candidates = empty_splits
        rng.shuffle(candidates)

        def incremental_cost(split: str) -> tuple[float, float]:
            old_total = image_counts[split]
            new_total = old_total + len(group_samples)
            target_total = max(target_images[split], 1.0)
            total_delta = ((new_total - target_images[split]) / target_total) ** 2 - (
                (old_total - target_images[split]) / target_total
            ) ** 2

            signature_delta = 0.0
            for signature, amount in group_signature_counts.items():
                old_value = signature_counts[(signature, split)]
                new_value = old_value + amount
                target = max(target_signatures[(signature, split)], 1.0)
                signature_delta += (
                    (new_value - target_signatures[(signature, split)]) / target
                ) ** 2 - (
                    (old_value - target_signatures[(signature, split)]) / target
                ) ** 2

            overflow = max(0.0, new_total - target_images[split]) / target_total
            return (3.0 * total_delta + signature_delta + 4.0 * overflow**2, overflow)

        chosen = min(candidates, key=incremental_cost)
        split_groups[chosen].append((group_name, group_samples))
        image_counts[chosen] += len(group_samples)
        for signature, amount in group_signature_counts.items():
            signature_counts[(signature, chosen)] += amount

    result = {
        split: [
            sample
            for _, group_samples in split_groups[split]
            for sample in group_samples
        ]
        for split in SPLIT_NAMES
    }
    for files in result.values():
        rng.shuffle(files)
    return result


def validate_split_coverage(
    split_files: Mapping[str, Sequence[Sample]],
    class_names: Mapping[int, str],
    strict: bool,
) -> list[str]:
    warnings: list[str] = []
    for split, samples in split_files.items():
        present = {class_id for sample in samples for class_id in sample.class_ids}
        missing = [
            class_names[class_id] for class_id in class_names if class_id not in present
        ]
        if missing:
            message = (
                f"Split '{split}' contains no boxes for class(es): {', '.join(missing)}"
            )
            if strict:
                raise DatasetValidationError(message)
            warnings.append(message)
            LOGGER.warning(message)
    return warnings


def place_file(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source.resolve())
    else:  # Defensive; argparse already validates this.
        raise ValueError(f"Unknown copy mode: {mode}")


def write_manifest(
    root: Path,
    split_files: Mapping[str, Sequence[Sample]],
    class_names: Mapping[int, str],
) -> None:
    manifest_path = root / "manifest.csv"
    fieldnames = [
        "split",
        "image",
        "label",
        "width",
        "height",
        "boxes",
        "class_ids",
        "class_names",
        "source_image",
        "source_label",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split in SPLIT_NAMES:
            for sample in split_files[split]:
                unique_classes = sorted(set(sample.class_ids))
                writer.writerow(
                    {
                        "split": split,
                        "image": (
                            Path(split) / "images" / sample.relative_image
                        ).as_posix(),
                        "label": (
                            Path(split) / "labels" / sample.relative_label
                        ).as_posix(),
                        "width": sample.width,
                        "height": sample.height,
                        "boxes": sample.box_count,
                        "class_ids": ",".join(map(str, unique_classes)),
                        "class_names": ",".join(class_names[i] for i in unique_classes),
                        "source_image": str(sample.image_path),
                        "source_label": str(sample.label_path)
                        if sample.label_path
                        else "",
                    }
                )


def build_report(
    split_files: Mapping[str, Sequence[Sample]],
    class_names: Mapping[int, str],
    ratios: Mapping[str, float],
    seed: int,
    missing_labels: Sequence[Path],
    skipped_images: Sequence[Path],
    orphan_labels: Sequence[Path],
    warnings: Sequence[str],
    group_regex: str | None,
    copy_mode: str,
) -> dict[str, object]:
    total_images = sum(len(files) for files in split_files.values())
    per_split: dict[str, object] = {}

    for split in SPLIT_NAMES:
        files = split_files[split]
        class_image_counts = Counter()
        class_box_counts = Counter()
        negatives = 0
        for sample in files:
            if not sample.class_ids:
                negatives += 1
            for class_id in set(sample.class_ids):
                class_image_counts[class_id] += 1
            class_box_counts.update(sample.class_ids)

        per_split[split] = {
            "images": len(files),
            "actual_ratio": len(files) / total_images if total_images else 0.0,
            "negative_images": negatives,
            "boxes": sum(sample.box_count for sample in files),
            "class_image_counts": {
                class_names[class_id]: class_image_counts[class_id]
                for class_id in class_names
            },
            "class_box_counts": {
                class_names[class_id]: class_box_counts[class_id]
                for class_id in class_names
            },
        }

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_images": total_images,
        "class_names": class_names,
        "requested_ratios": ratios,
        "seed": seed,
        "copy_mode": copy_mode,
        "group_regex": group_regex,
        "splits": per_split,
        "missing_label_count": len(missing_labels),
        "skipped_image_count": len(skipped_images),
        "orphan_label_count": len(orphan_labels),
        "orphan_labels_preview": [str(path) for path in orphan_labels[:100]],
        "warnings": list(warnings),
    }


def build_staging_dataset(
    staging: Path,
    split_files: Mapping[str, Sequence[Sample]],
    class_names: Mapping[int, str],
    report: Mapping[str, object],
    copy_mode: str,
) -> None:
    for split in SPLIT_NAMES:
        for sample in split_files[split]:
            image_destination = staging / split / "images" / sample.relative_image
            label_destination = staging / split / "labels" / sample.relative_label
            place_file(sample.image_path, image_destination, copy_mode)
            label_destination.parent.mkdir(parents=True, exist_ok=True)
            if sample.label_path is None:
                label_destination.write_text("", encoding="utf-8")
            else:
                place_file(sample.label_path, label_destination, copy_mode)

    write_manifest(staging, split_files, class_names)
    with (staging / "dataset_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def commit_staging_dataset(staging: Path, output: Path, overwrite: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        staging.replace(output)
        return

    if not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output}. Use --overwrite to replace it."
        )

    backup = output.parent / f".{output.name}.backup-{uuid.uuid4().hex[:8]}"
    output.replace(backup)
    try:
        staging.replace(output)
    except Exception:
        if not output.exists() and backup.exists():
            backup.replace(output)
        raise
    else:
        shutil.rmtree(backup)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    try:
        ratios = SplitRatios(args.train_ratio, args.val_ratio, args.test_ratio)
        ratios.validate()
        ratio_map = ratios.as_dict()

        image_root = args.images.expanduser().resolve()
        label_root = args.labels.expanduser().resolve()
        output = args.output.expanduser().resolve()

        if not label_root.is_dir():
            raise FileNotFoundError(f"Label directory not found: {label_root}")
        if output.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output}. Use --overwrite to replace it."
            )

        class_names = load_class_names(args.data_yaml)
        images = discover_images(image_root)
        ensure_unique_relative_stems(images, image_root)
        LOGGER.info(
            "Found %d image(s) and %d configured class(es)",
            len(images),
            len(class_names),
        )

        samples, missing_labels, skipped_images = build_samples(
            images=images,
            image_root=image_root,
            label_root=label_root,
            class_names=class_names,
            missing_label_policy=args.missing_label,
            verify_images=args.verify_images,
        )
        if len(samples) < len(SPLIT_NAMES):
            raise DatasetValidationError(
                f"At least {len(SPLIT_NAMES)} usable images are required; got {len(samples)}."
            )

        if args.check_duplicates:
            reject_exact_duplicates(samples)

        group_pattern = compile_group_regex(args.group_regex)
        if group_pattern is None:
            split_files = stratified_split(samples, ratio_map, args.seed)
        else:
            split_files = grouped_split(samples, ratio_map, args.seed, group_pattern)

        warnings = validate_split_coverage(
            split_files, class_names, strict=args.strict_class_coverage
        )
        orphan_labels = find_orphan_labels(label_root, samples, image_root)
        if orphan_labels:
            message = f"Found {len(orphan_labels)} orphan label file(s) with no matching image."
            warnings.append(message)
            LOGGER.warning(message)

        report = build_report(
            split_files=split_files,
            class_names=class_names,
            ratios=ratio_map,
            seed=args.seed,
            missing_labels=missing_labels,
            skipped_images=skipped_images,
            orphan_labels=orphan_labels,
            warnings=warnings,
            group_regex=args.group_regex,
            copy_mode=args.copy_mode,
        )

        staging = output.parent / f".{output.name}.building-{uuid.uuid4().hex[:8]}"
        try:
            staging.mkdir(parents=True, exist_ok=False)
            build_staging_dataset(
                staging=staging,
                split_files=split_files,
                class_names=class_names,
                report=report,
                copy_mode=args.copy_mode,
            )
            commit_staging_dataset(staging, output, args.overwrite)
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

        LOGGER.info("Dataset prepared successfully: %s", output)
        for split in SPLIT_NAMES:
            LOGGER.info("  %-5s %d image(s)", split + ":", len(split_files[split]))
        LOGGER.info("Manifest: %s", output / "manifest.csv")
        LOGGER.info("Report:   %s", output / "dataset_report.json")
        return 0

    except KeyboardInterrupt:
        LOGGER.warning("Dataset preparation interrupted by user.")
        return 130
    except (DatasetValidationError, FileNotFoundError, FileExistsError, OSError) as exc:
        LOGGER.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())