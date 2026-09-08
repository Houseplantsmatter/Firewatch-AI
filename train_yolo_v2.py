"""Version 2 configuration, preflight validation, and training orchestration."""

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml


class ConfigError(Exception):
    """Raised when the training configuration is invalid."""


class DatasetError(Exception):
    """Raised when the YOLO dataset layout or labels are invalid."""


class RuntimeEnvironmentError(Exception):
    """Raised when the selected runtime environment cannot run training."""


TOP_LEVEL_KEYS = {"version", "model", "data", "project", "name", "train", "evaluation"}
TRAIN_KEYS = {
    "epochs", "imgsz", "batch", "device", "workers", "patience", "optimizer", "lr0", "lrf",
    "cos_lr", "pretrained", "amp", "cache", "seed", "deterministic", "save_period", "val", "plots",
    "degrees", "translate", "scale", "shear", "perspective", "flipud", "fliplr", "hsv_h", "hsv_s",
    "hsv_v", "mosaic", "mixup", "close_mosaic",
}
EVALUATION_KEYS = {"split", "imgsz", "batch", "device", "half", "plots"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
OPTIMIZERS = {
    "auto": "auto",
    "adam": "Adam",
    "adamax": "Adamax",
    "adamw": "AdamW",
    "nadam": "NAdam",
    "radam": "RAdam",
    "rmsprop": "RMSProp",
    "sgd": "SGD",
}


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _require_exact_keys(section: Any, expected: Set[str], location: str) -> Dict[str, Any]:
    if not isinstance(section, dict):
        raise ConfigError("{} must be a mapping".format(location))
    actual = set(section)
    missing = expected - actual
    unknown = actual - expected
    if missing:
        raise ConfigError("{} is missing required keys: {}".format(location, ", ".join(sorted(missing))))
    if unknown:
        raise ConfigError("{} has unknown keys: {}".format(location, ", ".join(sorted(unknown))))
    return section


def _require_non_empty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("{} must be a non-empty string".format(location))
    return value


def _require_positive_integer(value: Any, location: str) -> None:
    if not _is_integer(value) or value <= 0:
        raise ConfigError("{} must be a positive integer".format(location))


def _require_nonnegative_integer(value: Any, location: str) -> None:
    if not _is_integer(value) or value < 0:
        raise ConfigError("{} must be a nonnegative integer".format(location))


def _require_probability(value: Any, location: str) -> None:
    if not _is_number(value) or not 0 <= value <= 1:
        raise ConfigError("{} must be a finite number between 0 and 1".format(location))


def _require_boolean(value: Any, location: str) -> None:
    if not isinstance(value, bool):
        raise ConfigError("{} must be a boolean".format(location))


def _resolve_path(value: str, base: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def load_config(path: Path) -> Dict[str, Any]:
    """Load, validate, and normalize a version 2 training configuration."""
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as file_handle:
            loaded = yaml.safe_load(file_handle)
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError("Unable to load configuration {}: {}".format(path, error)) from error

    config = _require_exact_keys(loaded, TOP_LEVEL_KEYS, "configuration")
    if config["version"] != 2 or not _is_integer(config["version"]):
        raise ConfigError("version must be integer 2")
    for key in ("model", "data", "project", "name"):
        _require_non_empty_string(config[key], key)

    train = _require_exact_keys(config["train"], TRAIN_KEYS, "train")
    for key in ("epochs", "imgsz", "batch"):
        _require_positive_integer(train[key], "train.{}".format(key))
    for key in ("device", "workers", "patience", "seed", "save_period", "close_mosaic"):
        _require_nonnegative_integer(train[key], "train.{}".format(key))
    optimizer = _require_non_empty_string(train["optimizer"], "train.optimizer")
    canonical_optimizer = OPTIMIZERS.get(optimizer.casefold())
    if canonical_optimizer is None:
        raise ConfigError(
            "train.optimizer must be one of: {}".format(", ".join(OPTIMIZERS.values()))
        )
    if not _is_number(train["lr0"]) or train["lr0"] <= 0:
        raise ConfigError("train.lr0 must be a positive finite number")
    _require_probability(train["lrf"], "train.lrf")
    for key in ("degrees", "shear"):
        if not _is_number(train[key]) or train[key] < 0:
            raise ConfigError("train.{} must be a nonnegative finite number".format(key))
    for key in ("translate", "scale", "perspective", "flipud", "fliplr", "hsv_h", "hsv_s", "hsv_v", "mosaic", "mixup"):
        _require_probability(train[key], "train.{}".format(key))
    for key in ("cos_lr", "pretrained", "amp", "cache", "deterministic", "val", "plots"):
        _require_boolean(train[key], "train.{}".format(key))

    evaluation = _require_exact_keys(config["evaluation"], EVALUATION_KEYS, "evaluation")
    if evaluation["split"] != "test":
        raise ConfigError("evaluation.split must equal exactly 'test'")
    for key in ("imgsz", "batch"):
        _require_positive_integer(evaluation[key], "evaluation.{}".format(key))
    _require_nonnegative_integer(evaluation["device"], "evaluation.device")
    for key in ("half", "plots"):
        _require_boolean(evaluation[key], "evaluation.{}".format(key))

    result = dict(config)
    result["model"] = _resolve_path(config["model"], path.parent)
    result["data"] = _resolve_path(config["data"], path.parent)
    result["project"] = _resolve_path(config["project"], path.parent)
    result["train"] = dict(train)
    result["train"]["optimizer"] = canonical_optimizer
    result["evaluation"] = dict(evaluation)
    return result


def validate_label(path: Path, class_ids: Set[int]) -> Tuple[int, Dict[int, int]]:
    """Validate a YOLO label file and return its box and class counts."""
    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise DatasetError("Unable to read label file {}: {}".format(path, error)) from error

    boxes = 0
    class_counts = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise DatasetError("{} line {} must contain exactly five fields".format(path, line_number))
        try:
            class_id = int(fields[0])
        except ValueError as error:
            raise DatasetError("{} line {} has a noninteger class ID".format(path, line_number)) from error
        if class_id not in class_ids:
            raise DatasetError("{} line {} has undeclared class ID {}".format(path, line_number, class_id))
        try:
            coordinates = [float(value) for value in fields[1:]]
        except ValueError as error:
            raise DatasetError("{} line {} has a nonnumeric coordinate".format(path, line_number)) from error
        if not all(math.isfinite(value) for value in coordinates):
            raise DatasetError("{} line {} has a nonfinite coordinate".format(path, line_number))
        if not all(0 <= value <= 1 for value in coordinates):
            raise DatasetError("{} line {} has a coordinate outside [0, 1]".format(path, line_number))
        if coordinates[2] <= 0 or coordinates[3] <= 0:
            raise DatasetError("{} line {} has nonpositive width or height".format(path, line_number))
        boxes += 1
        class_counts[class_id] = class_counts.get(class_id, 0) + 1
    return boxes, class_counts


def _class_ids_from_names(names: Any, data_path: Path) -> Set[int]:
    if isinstance(names, list):
        if not names or any(not isinstance(name, str) or not name.strip() for name in names):
            raise DatasetError("{} has an invalid names list".format(data_path))
        return set(range(len(names)))
    if isinstance(names, dict):
        if not names:
            raise DatasetError("{} has an empty names mapping".format(data_path))
        class_ids = set()
        for class_id, name in names.items():
            if not _is_integer(class_id) or class_id < 0 or not isinstance(name, str) or not name.strip():
                raise DatasetError("{} has an invalid names mapping".format(data_path))
            class_ids.add(class_id)
        if class_ids != set(range(len(class_ids))):
            raise DatasetError(
                "{} names mapping class IDs must be contiguous from 0".format(data_path)
            )
        return class_ids
    raise DatasetError("{} must define names as a non-empty mapping or list".format(data_path))


def validate_dataset(data_path: Path) -> Dict[str, Dict[str, Any]]:
    """Validate the YOLO dataset described by *data_path* and return split summaries."""
    data_path = Path(data_path).resolve()
    try:
        with data_path.open("r", encoding="utf-8") as file_handle:
            data = yaml.safe_load(file_handle)
    except (OSError, yaml.YAMLError) as error:
        raise DatasetError("Unable to load dataset YAML {}: {}".format(data_path, error)) from error
    if not isinstance(data, dict):
        raise DatasetError("{} must contain a mapping".format(data_path))
    class_ids = _class_ids_from_names(data.get("names"), data_path)
    if "path" not in data:
        root = data_path.parent.resolve()
    else:
        root_value = data["path"]
        if not isinstance(root_value, str) or not root_value.strip():
            raise DatasetError("{} path must be a non-empty string when provided".format(data_path))
        root = _resolve_path(root_value, data_path.parent)

    summaries = {}
    for split in ("train", "val", "test"):
        split_value = data.get(split)
        if not isinstance(split_value, str) or not split_value.strip():
            raise DatasetError("{} must define a non-empty {} split".format(data_path, split))
        image_dir = _resolve_path(split_value, root)
        if not _is_within(image_dir, root):
            raise DatasetError("{} split is outside the dataset root: {}".format(split, image_dir))
        if image_dir.name != "images":
            raise DatasetError("{} split must point to a directory named images: {}".format(split, image_dir))
        labels_dir = image_dir.parent / "labels"
        if not image_dir.is_dir() or not labels_dir.is_dir():
            raise DatasetError("{} split requires sibling images and labels directories".format(split))
        images = sorted(image for image in image_dir.rglob("*")
                        if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise DatasetError("{} split has no supported images in {}".format(split, image_dir))

        expected_labels = {
            image.relative_to(image_dir).with_suffix(".txt") for image in images
        }
        orphan_labels = sorted(
            label for label in labels_dir.rglob("*")
            if label.is_file()
            and label.suffix.lower() == ".txt"
            and label.relative_to(labels_dir) not in expected_labels
        )
        if orphan_labels:
            raise DatasetError(
                "Orphan label has no corresponding supported image: {}".format(orphan_labels[0])
            )

        boxes = 0
        negative_images = 0
        class_counts = {}
        for image in images:
            label_path = labels_dir / image.relative_to(image_dir).with_suffix(".txt")
            if not label_path.is_file():
                raise DatasetError("Missing label file {} for image {}".format(label_path, image))
            label_boxes, label_counts = validate_label(label_path, class_ids)
            boxes += label_boxes
            if not label_boxes:
                negative_images += 1
            for class_id, count in label_counts.items():
                class_counts[class_id] = class_counts.get(class_id, 0) + count
        summaries[split] = {
            "images": len(images),
            "boxes": boxes,
            "negative_images": negative_images,
            "class_counts": class_counts,
        }
    return summaries


def validate_cuda(device: Any) -> Dict[str, Any]:
    """Validate a CUDA device selection and return its normalized details."""
    if _is_integer(device):
        index = device
    elif isinstance(device, str) and device.strip().isdigit():
        index = int(device.strip())
    else:
        raise RuntimeEnvironmentError(
            "Requested device {!r} is invalid; use a nonnegative CUDA device index".format(device)
        )
    if index < 0:
        raise RuntimeEnvironmentError(
            "Requested device {} is invalid; use a nonnegative CUDA device index".format(index)
        )

    try:
        import torch
    except ImportError as error:
        raise RuntimeEnvironmentError(
            "PyTorch is required to validate requested device {}; install the training dependencies".format(index)
        ) from error

    if not torch.cuda.is_available():
        raise RuntimeEnvironmentError(
            "Requested device {} is unavailable because CUDA is not available".format(index)
        )
    device_count = torch.cuda.device_count()
    if index >= device_count:
        raise RuntimeEnvironmentError(
            "Requested device {} is unavailable; CUDA exposes {} device(s)".format(index, device_count)
        )
    properties = torch.cuda.get_device_properties(index)
    return {
        "index": index,
        "name": torch.cuda.get_device_name(index),
        "vram_gb": properties.total_memory / float(1024 ** 3),
    }


def build_train_args(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build an isolated Ultralytics training argument mapping from *config*."""
    arguments = dict(config["train"])
    arguments.update({
        "data": str(config["data"]),
        "project": str(config["project"]),
        "name": config["name"],
        "exist_ok": False,
    })
    return arguments


def _load_yolo_class():
    """Load Ultralytics only when a training run is actually requested."""
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeEnvironmentError(
            "Ultralytics is required to start training; install the training dependencies"
        ) from error

    return YOLO


def _best_checkpoint(model: Any) -> Path:
    best = getattr(getattr(model, "trainer", None), "best", None)
    if not best:
        raise RuntimeEnvironmentError("Training did not report a best checkpoint")
    best_path = Path(best)
    if not best_path.is_file():
        raise RuntimeEnvironmentError("Best checkpoint is missing: {}".format(best_path))
    return best_path


def run_training(
    config: Dict[str, Any],
    resume: Optional[Path] = None,
    resume_batch: Optional[int] = None,
) -> Path:
    """Run a fresh or explicit-resume training operation and return its best weights."""
    if resume is not None:
        checkpoint = Path(resume)
        if not checkpoint.is_file():
            raise ConfigError("Resume checkpoint does not exist: {}".format(checkpoint))
        yolo_class = _load_yolo_class()
        model = yolo_class(str(checkpoint))
        resume_arguments = {"resume": True, "device": config["train"]["device"]}
        if resume_batch is not None:
            resume_arguments["batch"] = resume_batch
        model.train(**resume_arguments)
        return _best_checkpoint(model)

    output_directories = (
        Path(config["project"]) / config["name"],
        Path(config["project"]) / (config["name"] + "_test"),
    )
    for output_directory in output_directories:
        if output_directory.exists():
            raise RuntimeEnvironmentError(
                "Output directory already exists: {}. Choose a different --name or explicitly "
                "resume with --resume CHECKPOINT.".format(output_directory)
            )

    model_path = Path(config["model"])
    if not model_path.is_file():
        raise ConfigError("Configured model does not exist: {}".format(model_path))
    yolo_class = _load_yolo_class()
    model = yolo_class(str(model_path))
    model.train(**build_train_args(config))
    best_path = _best_checkpoint(model)

    evaluation_model = yolo_class(str(best_path))
    evaluation_args = dict(config["evaluation"])
    evaluation_args.update({
        "data": str(config["data"]),
        "project": str(config["project"]),
        "name": config["name"] + "_test",
        "exist_ok": False,
    })
    metrics = evaluation_model.val(**evaluation_args)
    print("Precision: {}".format(metrics.box.mp))
    print("Recall: {}".format(metrics.box.mr))
    print("mAP50: {}".format(metrics.box.map50))
    print("mAP50-95: {}".format(metrics.box.map))
    print("Best checkpoint: {}".format(best_path))
    return best_path


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _nonnegative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a nonnegative integer") from error
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def _nonempty_string(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("must be a non-empty string")
    return value


def _apply_overrides(config: Dict[str, Any], arguments: argparse.Namespace) -> Dict[str, Any]:
    updated = dict(config)
    updated["train"] = dict(config["train"])
    updated["evaluation"] = dict(config["evaluation"])
    if arguments.epochs is not None:
        updated["train"]["epochs"] = arguments.epochs
    if arguments.batch is not None:
        updated["train"]["batch"] = arguments.batch
        updated["evaluation"]["batch"] = arguments.batch
    if arguments.device is not None:
        updated["train"]["device"] = arguments.device
        updated["evaluation"]["device"] = arguments.device
    if arguments.name is not None:
        updated["name"] = arguments.name
    return updated


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Fire and Smoke YOLO V2 model")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().with_name("train_config_v2.yaml"))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--epochs", type=_positive_int)
    parser.add_argument("--batch", type=_positive_int)
    parser.add_argument("--device", type=_nonnegative_int)
    parser.add_argument("--name", type=_nonempty_string)
    parser.add_argument("--resume", type=Path)
    return parser


def _print_preflight(cuda: Dict[str, Any], summaries: Dict[str, Dict[str, Any]]) -> None:
    print("GPU {}: {} ({:.2f} GB VRAM)".format(cuda["index"], cuda["name"], cuda["vram_gb"]))
    for split in ("train", "val", "test"):
        summary = summaries[split]
        print("{}: images={}, boxes={}, negative_images={}, class_counts={}".format(
            split, summary["images"], summary["boxes"], summary["negative_images"], summary["class_counts"]
        ))


def _oom_recovery_message(
    config: Dict[str, Any], arguments: argparse.Namespace, run_directory: Path
) -> str:
    if arguments.resume is None:
        requested_batch = config["train"]["batch"]
        recommended_batch = max(1, requested_batch // 2)
        checkpoint = run_directory / "weights" / "last.pt"
        return (
            "ERROR: CUDA out of memory during a fresh run with requested train/evaluation "
            "batch {}. If the checkpoint exists, retry: train_yolo_v2.py --resume \"{}\" "
            "--batch {}. If no checkpoint exists, retry with a distinct --name: "
            "train_yolo_v2.py --name DISTINCT_NAME --batch {}."
        ).format(requested_batch, checkpoint, recommended_batch, recommended_batch)

    checkpoint = Path(arguments.resume).resolve()
    if arguments.batch is not None:
        recommended_batch = max(1, arguments.batch // 2)
        return (
            "ERROR: CUDA out of memory while resuming with explicitly requested batch {}. "
            "Retry: train_yolo_v2.py --resume \"{}\" --batch {}."
        ).format(arguments.batch, checkpoint, recommended_batch)

    conservative_batch = max(1, config["train"]["batch"] // 2)
    return (
        "ERROR: CUDA out of memory while resuming; the checkpoint batch is unknown. "
        "A conservative retry suggestion is: train_yolo_v2.py --resume \"{}\" --batch {}."
    ).format(checkpoint, conservative_batch)


def main(argv: Optional[List[str]] = None) -> int:
    """Run the command-line training workflow and return its process status."""
    arguments = _build_parser().parse_args(argv)
    run_directory = None
    try:
        config = _apply_overrides(load_config(arguments.config), arguments)
        if arguments.resume is None:
            run_directory = Path(config["project"]) / config["name"]
        else:
            checkpoint = Path(arguments.resume).resolve()
            run_directory = checkpoint.parent.parent if checkpoint.parent.name == "weights" else checkpoint.parent
        summaries = validate_dataset(config["data"])
        cuda = validate_cuda(config["train"]["device"])
        _print_preflight(cuda, summaries)
        if arguments.check_only:
            print("Preflight checks passed.")
            return 0
        run_training(config, arguments.resume, resume_batch=arguments.batch)
        return 0
    except (ConfigError, DatasetError, RuntimeEnvironmentError) as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        return 2
    except RuntimeError as error:
        if "cuda" in str(error).lower() and "out of memory" in str(error).lower():
            print(_oom_recovery_message(config, arguments, run_directory), file=sys.stderr)
            return 3
        raise
    except KeyboardInterrupt:
        if run_directory is None:
            print("Training interrupted before the run directory was planned.", file=sys.stderr)
        else:
            print(
                "Training interrupted; checkpoints are preserved in {}.".format(run_directory),
                file=sys.stderr,
            )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
