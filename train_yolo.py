"""Train, resume, or evaluate an Ultralytics fire/smoke detector."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

LOGGER = logging.getLogger("train_yolo")
SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class ConfigurationError(ValueError):
    """Raised when a training or dataset configuration is invalid."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train, resume, or evaluate a YOLO fire/smoke detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "train_config.yaml")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        type=Path,
        metavar="LAST.PT",
        help="Resume an interrupted run from a last.pt checkpoint.",
    )
    mode.add_argument(
        "--evaluate",
        type=Path,
        metavar="WEIGHTS.PT",
        help="Evaluate weights without training.",
    )

    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--test-after-training",
        action="store_true",
        help="Evaluate best.pt on the test split after training. Normally do this only for a final model.",
    )

    # Common, convenient overrides. --set supports every other Ultralytics key.
    parser.add_argument("--model", help="Pretrained model name or local weights path.")
    parser.add_argument("--data", type=Path, help="Dataset YAML path.")
    parser.add_argument(
        "--device", help="auto, cpu, mps, 0, -1, or comma-separated CUDA IDs."
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", help="Integer, fraction, or 'auto'.")
    parser.add_argument("--workers", help="Integer or 'auto'.")
    parser.add_argument("--project", type=Path)
    parser.add_argument("--name")
    parser.add_argument(
        "--cache",
        choices=("false", "ram", "disk"),
        help="Dataset caching mode.",
    )
    parser.add_argument(
        "--exist-ok",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow reuse of an existing project/name directory.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override any flat Ultralytics training setting; repeat as needed.",
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


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigurationError(f"Configuration must be a YAML mapping: {path}")
    return dict(data)


def parse_yaml_scalar(value: str) -> Any:
    try:
        parsed = yaml.safe_load(value)
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"Could not parse override value {value!r}: {exc}"
        ) from exc
    if isinstance(parsed, (dict, list)):
        return parsed
    return parsed


def apply_cli_overrides(
    config: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    result = dict(config)
    direct = {
        "model": args.model,
        "data": str(args.data) if args.data is not None else None,
        "device": args.device,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "project": str(args.project) if args.project is not None else None,
        "name": args.name,
    }
    for key, value in direct.items():
        if value is not None:
            result[key] = value

    if args.batch is not None:
        result["batch"] = parse_yaml_scalar(args.batch)
    if args.workers is not None:
        result["workers"] = parse_yaml_scalar(args.workers)
    if args.cache is not None:
        result["cache"] = False if args.cache == "false" else args.cache
    if args.exist_ok is not None:
        result["exist_ok"] = args.exist_ok

    for override in args.set:
        if "=" not in override:
            raise ConfigurationError(
                f"Invalid --set value {override!r}; expected KEY=VALUE."
            )
        key, raw_value = override.split("=", 1)
        key = key.strip()
        if not key or "." in key:
            raise ConfigurationError(
                "--set keys must be non-empty, flat configuration names without dots."
            )
        result[key] = parse_yaml_scalar(raw_value.strip())
    return result


def resolve_input_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def resolve_model_reference(value: str | Path, base: Path) -> str:
    text = str(value)
    path = Path(text).expanduser()

    # A path with directories is always treated as a required local file.
    if path.is_absolute() or path.parent != Path("."):
        resolved = resolve_input_path(path, base)
        if not resolved.is_file():
            raise FileNotFoundError(f"Model file not found: {resolved}")
        return str(resolved)

    # Prefer a same-directory local file; otherwise keep a standard model name
    # such as yolo11s.pt so Ultralytics can resolve/download it.
    local_candidate = (base / path).resolve()
    if local_candidate.is_file():
        return str(local_candidate)
    return text


def parse_device(value: Any) -> Any:
    if value is None or (isinstance(value, str) and value.strip().lower() == "auto"):
        if torch.cuda.is_available():
            return 0
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    if isinstance(value, (int, list)):
        return value
    if not isinstance(value, str):
        raise ConfigurationError(f"Unsupported device value: {value!r}")

    text = value.strip().lower()
    if "," in text:
        try:
            return [int(part.strip()) for part in text.split(",")]
        except ValueError as exc:
            raise ConfigurationError(f"Invalid CUDA device list: {value!r}") from exc
    if text.lstrip("-").isdigit():
        return int(text)
    return text


def validate_device(device: Any) -> None:
    uses_cuda = isinstance(device, int) or (
        isinstance(device, list) and all(isinstance(item, int) for item in device)
    )
    if uses_cuda and not torch.cuda.is_available():
        raise ConfigurationError(
            f"CUDA device {device!r} was requested, but torch.cuda.is_available() is False."
        )
    if device == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise ConfigurationError("MPS was requested, but it is not available.")


def normalize_batch(value: Any, device: Any) -> int | float:
    if isinstance(value, str) and value.strip().lower() == "auto":
        if isinstance(device, list):
            raise ConfigurationError(
                "Set an explicit total batch size when using multiple CUDA devices."
            )
        if isinstance(device, int):
            return -1
        return 8 if device == "mps" else 4
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(
            "batch must be an integer, a utilization fraction, or 'auto'."
        )
    if isinstance(value, int) and value == -1:
        return value
    if value <= 0:
        raise ConfigurationError("batch must be positive, or -1 for CUDA auto-batch.")
    return value


def normalize_workers(value: Any) -> int:
    if isinstance(value, str) and value.strip().lower() == "auto":
        cpu_count = os.cpu_count() or 2
        if os.name == "nt":
            return min(4, max(1, cpu_count // 2))
        return min(8, max(1, cpu_count - 1))
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigurationError("workers must be a non-negative integer or 'auto'.")
    return value


def normalize_cache(value: Any) -> bool | str:
    if value is None or value is False:
        return False
    if value is True:
        return "ram"
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"false", "none", "off"}:
            return False
        if text in {"ram", "disk"}:
            return text
    raise ConfigurationError("cache must be false, 'ram', or 'disk'.")


def normalize_training_config(
    config: dict[str, Any], config_path: Path, *, require_model: bool
) -> dict[str, Any]:
    config_dir = config_path.expanduser().resolve().parent
    required_keys = ["data"] + (["model"] if require_model else [])
    required = [key for key in required_keys if key not in config]
    if required:
        raise ConfigurationError(
            f"Missing required config key(s): {', '.join(required)}"
        )

    result = dict(config)
    if require_model:
        result["model"] = resolve_model_reference(result["model"], config_dir)
    result["data"] = str(resolve_input_path(result["data"], config_dir))

    project_value = result.get("project", "runs/detect")
    result["project"] = str(resolve_input_path(project_value, config_dir))

    name = str(result.get("name", "train")).strip()
    if not name or Path(name).name != name:
        raise ConfigurationError("name must be a single non-empty directory name.")
    result["name"] = name

    device = parse_device(result.get("device", "auto"))
    validate_device(device)
    result["device"] = device
    result["batch"] = normalize_batch(result.get("batch", 16), device)
    result["workers"] = normalize_workers(result.get("workers", "auto"))
    result["cache"] = normalize_cache(result.get("cache", False))

    epochs = result.get("epochs", 100)
    imgsz = result.get("imgsz", 640)
    patience = result.get("patience", 100)
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ConfigurationError("epochs must be a positive integer.")
    if isinstance(imgsz, bool) or not isinstance(imgsz, int) or imgsz < 32:
        raise ConfigurationError("imgsz must be an integer of at least 32.")
    if isinstance(patience, bool) or not isinstance(patience, int) or patience < 0:
        raise ConfigurationError("patience must be a non-negative integer.")

    for key in (
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "fliplr",
        "flipud",
        "mosaic",
        "mixup",
    ):
        if key in result and not isinstance(result[key], (int, float)):
            raise ConfigurationError(f"{key} must be numeric.")

    return result


def parse_class_names(raw: Any, data_yaml: Path) -> dict[int, str]:
    if isinstance(raw, list):
        names = {index: str(name) for index, name in enumerate(raw)}
    elif isinstance(raw, dict):
        try:
            names = {int(index): str(name) for index, name in raw.items()}
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"Class IDs in {data_yaml} must be integers."
            ) from exc
    else:
        raise ConfigurationError(f"{data_yaml} must contain a 'names' list or mapping.")

    if not names:
        raise ConfigurationError(f"{data_yaml} must define at least one class name.")
    if sorted(names) != list(range(len(names))):
        raise ConfigurationError(
            f"Class IDs in {data_yaml} must be contiguous from zero; got {sorted(names)}."
        )
    return dict(sorted(names.items()))


def is_remote_path(value: str) -> bool:
    lowered = value.lower()
    return lowered.startswith(("http://", "https://", "s3://", "gs://", "azure://"))


def resolve_dataset_entry(
    value: str, dataset_root: Path, yaml_dir: Path
) -> Path | None:
    del yaml_dir  # Split entries are resolved relative to the dataset root (`path`).
    if is_remote_path(value):
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (dataset_root / path).resolve()


def validate_dataset_yaml(
    data_yaml: Path, required_splits: Sequence[str]
) -> dict[str, Any]:
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")
    with data_yaml.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigurationError(f"Dataset YAML must contain a mapping: {data_yaml}")

    names = parse_class_names(data.get("names"), data_yaml)
    yaml_dir = data_yaml.parent
    root_value = data.get("path", ".")
    dataset_root = resolve_input_path(root_value, yaml_dir)

    split_info: dict[str, Any] = {}
    for split in required_splits:
        raw_entry = data.get(split)
        if raw_entry in (None, ""):
            raise ConfigurationError(
                f"Dataset YAML has no '{split}' entry: {data_yaml}"
            )
        entries = raw_entry if isinstance(raw_entry, list) else [raw_entry]
        resolved_entries: list[str] = []
        image_count = 0
        for entry in entries:
            if not isinstance(entry, str):
                raise ConfigurationError(
                    f"Dataset '{split}' entries must be strings; got {entry!r}."
                )
            resolved = resolve_dataset_entry(entry, dataset_root, yaml_dir)
            if resolved is None:
                resolved_entries.append(entry)
                continue
            if not resolved.exists():
                raise FileNotFoundError(
                    f"Dataset '{split}' path does not exist: {resolved}"
                )
            resolved_entries.append(str(resolved))
            if resolved.is_dir():
                image_count += sum(
                    1
                    for path in resolved.rglob("*")
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                )
            elif resolved.suffix.lower() == ".txt":
                image_count += sum(
                    1
                    for line in resolved.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            else:
                image_count += 1

        if image_count == 0 and all(
            not is_remote_path(item) for item in resolved_entries
        ):
            raise ConfigurationError(f"Dataset split '{split}' contains no images.")
        split_info[split] = {"entries": resolved_entries, "image_count": image_count}

    return {
        "data_yaml": str(data_yaml),
        "dataset_root": str(dataset_root),
        "class_names": names,
        "splits": split_info,
    }


def seed_process(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return str(value)


def metrics_payload(metrics: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    results_dict = getattr(metrics, "results_dict", None)
    if isinstance(results_dict, Mapping):
        payload["results"] = dict(results_dict)

    box = getattr(metrics, "box", None)
    if box is not None:
        for attribute in ("map", "map50", "map75", "maps", "mp", "mr"):
            if hasattr(box, attribute):
                payload[attribute] = getattr(box, attribute)

    speed = getattr(metrics, "speed", None)
    if isinstance(speed, Mapping):
        payload["speed_ms"] = dict(speed)

    summary_method = getattr(metrics, "summary", None)
    if callable(summary_method):
        try:
            payload["summary"] = summary_method()
        except Exception as exc:
            LOGGER.debug("Could not export metrics.summary(): %s", exc)
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=json_default)
        handle.write("\n")


def trainer_save_dir(model: Any, fallback: Path) -> Path:
    trainer = getattr(model, "trainer", None)
    save_dir = getattr(trainer, "save_dir", None)
    return Path(save_dir).resolve() if save_dir is not None else fallback.resolve()


def validator_save_dir(model: Any, fallback: Path) -> Path:
    validator = getattr(model, "validator", None)
    save_dir = getattr(validator, "save_dir", None)
    return Path(save_dir).resolve() if save_dir is not None else fallback.resolve()


def log_environment(ultralytics_version: str, device: Any) -> None:
    LOGGER.info("Python:      %s", sys.version.split()[0])
    LOGGER.info("Ultralytics: %s", ultralytics_version)
    LOGGER.info("PyTorch:     %s", torch.__version__)
    LOGGER.info("Device:      %s", device)
    if torch.cuda.is_available():
        LOGGER.info("CUDA:        %s", torch.version.cuda)
        LOGGER.info("GPU:         %s", torch.cuda.get_device_name(0))


def run_evaluation(
    weights: Path,
    config: Mapping[str, Any],
    split: str,
    output_project: Path | None = None,
) -> Path:
    from ultralytics import YOLO

    data_yaml = Path(str(config["data"]))
    validate_dataset_yaml(data_yaml, required_splits=(split,))

    model = YOLO(str(weights))
    batch = config["batch"]
    if batch == -1 or isinstance(batch, float):
        batch = 16 if torch.cuda.is_available() else 4

    base_project = output_project or (Path(str(config["project"])) / "evaluation")
    run_name = f"{weights.stem}_{split}"
    LOGGER.info("Evaluating %s on split '%s'", weights, split)
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=int(config.get("imgsz", 640)),
        batch=batch,
        device=config["device"],
        workers=config["workers"],
        plots=True,
        project=str(base_project),
        name=run_name,
        exist_ok=False,
        verbose=True,
    )
    fallback = base_project / run_name
    save_dir = validator_save_dir(model, fallback)
    write_json(save_dir / "metrics.json", metrics_payload(metrics))

    box = getattr(metrics, "box", None)
    if box is not None:
        LOGGER.info("mAP50-95: %.4f", float(box.map))
        LOGGER.info("mAP50:    %.4f", float(box.map50))
        LOGGER.info("mAP75:    %.4f", float(box.map75))
    LOGGER.info("Evaluation output: %s", save_dir)
    return save_dir


def run_training(config: dict[str, Any], test_after_training: bool) -> Path:
    from ultralytics import YOLO

    data_yaml = Path(str(config["data"]))
    validate_dataset_yaml(data_yaml, required_splits=("train", "val"))

    seed = int(config.get("seed", 0))
    deterministic = bool(config.get("deterministic", True))
    seed_process(seed, deterministic)

    model_reference = str(config["model"])
    train_kwargs = dict(config)
    train_kwargs.pop("model", None)

    LOGGER.info("Starting new training run with model: %s", model_reference)
    LOGGER.info("Dataset: %s", data_yaml)
    model = YOLO(model_reference)
    model.train(**train_kwargs)

    fallback = Path(str(config["project"])) / str(config["name"])
    save_dir = trainer_save_dir(model, fallback)
    weights_dir = save_dir / "weights"
    best_weights = weights_dir / "best.pt"
    last_weights = weights_dir / "last.pt"

    resolved_payload = dict(config)
    resolved_payload["run_save_dir"] = str(save_dir)
    resolved_payload["best_weights"] = str(best_weights)
    resolved_payload["last_weights"] = str(last_weights)
    with (save_dir / "resolved_train_config.yaml").open(
        "w", encoding="utf-8"
    ) as handle:
        yaml.safe_dump(resolved_payload, handle, sort_keys=False, allow_unicode=True)

    LOGGER.info("Training output: %s", save_dir)
    LOGGER.info("Best weights:   %s", best_weights)
    LOGGER.info("Last weights:   %s", last_weights)

    if not best_weights.is_file():
        raise RuntimeError(
            f"Training completed, but best.pt was not found: {best_weights}"
        )

    if test_after_training:
        run_evaluation(
            weights=best_weights,
            config=config,
            split="test",
            output_project=save_dir / "evaluation",
        )
    return best_weights


def run_resume(checkpoint: Path, config: Mapping[str, Any]) -> Path:
    from ultralytics import YOLO

    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
    if checkpoint.name.lower() != "last.pt":
        LOGGER.warning(
            "Resuming is normally done from last.pt so optimizer and scheduler state are current."
        )

    seed_process(int(config.get("seed", 0)), bool(config.get("deterministic", True)))
    model = YOLO(str(checkpoint))
    LOGGER.info("Resuming training from: %s", checkpoint)
    model.train(
        resume=True,
        device=config["device"],
        workers=config["workers"],
        cache=config["cache"],
    )

    fallback = checkpoint.parent.parent
    save_dir = trainer_save_dir(model, fallback)
    best_weights = save_dir / "weights" / "best.pt"
    LOGGER.info("Resumed run output: %s", save_dir)
    LOGGER.info("Best weights:       %s", best_weights)
    return best_weights


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    try:
        config_path = args.config.expanduser().resolve()
        raw_config = load_yaml_mapping(config_path)
        overridden = apply_cli_overrides(raw_config, args)
        config = normalize_training_config(
            overridden,
            config_path,
            require_model=args.resume is None and args.evaluate is None,
        )

        import ultralytics

        log_environment(ultralytics.__version__, config["device"])

        if args.evaluate is not None:
            weights = args.evaluate.expanduser().resolve()
            if not weights.is_file():
                raise FileNotFoundError(f"Evaluation weights not found: {weights}")
            run_evaluation(weights, config, args.split)
        elif args.resume is not None:
            best_weights = run_resume(args.resume, config)
            if args.test_after_training:
                run_evaluation(
                    best_weights,
                    config,
                    "test",
                    output_project=best_weights.parent.parent / "evaluation",
                )
        else:
            run_training(config, args.test_after_training)
        return 0

    except KeyboardInterrupt:
        LOGGER.warning("Operation interrupted by user.")
        return 130
    except (
        ConfigurationError,
        FileNotFoundError,
        ImportError,
        RuntimeError,
        OSError,
        yaml.YAMLError,
    ) as exc:
        LOGGER.error("%s", exc)
        return 2
    except Exception as exc:  # Guard the CLI from third-party trainer failures.
        if args.log_level == "DEBUG":
            LOGGER.exception("YOLO operation failed")
        else:
            LOGGER.error("YOLO operation failed (%s): %s", type(exc).__name__, exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())