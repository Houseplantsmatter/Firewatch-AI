"""Run memory-efficient YOLO inference and write structured reports."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, IO, Mapping, Sequence
from urllib.parse import urlparse

LOGGER = logging.getLogger("detect")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = SCRIPT_DIR / "runs/detect/fire_smoke_yolo11s/weights/best.pt"
DEFAULT_SOURCE = SCRIPT_DIR / "image copy.png"

CSV_FIELDS = [
    "item_index",
    "source",
    "image_width",
    "image_height",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "x_center",
    "y_center",
    "box_width",
    "box_height",
    "x1_normalized",
    "y1_normalized",
    "x2_normalized",
    "y2_normalized",
]


@dataclass(frozen=True, slots=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    x_center: float
    y_center: float
    box_width: float
    box_height: float
    x1_normalized: float
    y1_normalized: float
    x2_normalized: float
    y2_normalized: float


class ReportWriter:
    """Incrementally writes reports so long videos do not accumulate in memory."""

    def __init__(self, run_dir: Path, enabled: bool) -> None:
        self.run_dir = run_dir
        self.enabled = enabled
        self._jsonl: IO[str] | None = None
        self._csv: IO[str] | None = None
        self._csv_writer: csv.DictWriter[str] | None = None
        self._items_written = 0

    def open(self) -> None:
        if not self.enabled:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = (self.run_dir / "predictions.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        )
        self._csv = (self.run_dir / "detections.csv").open(
            "w", encoding="utf-8", newline=""
        )
        self._csv_writer = csv.DictWriter(self._csv, fieldnames=CSV_FIELDS)
        self._csv_writer.writeheader()

    def write_item(self, item: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        assert self._jsonl is not None
        assert self._csv_writer is not None
        self._jsonl.write(json.dumps(item, ensure_ascii=False) + "\n")

        for detection in item["detections"]:
            row = {
                "item_index": item["item_index"],
                "source": item["source"],
                "image_width": item["image_width"],
                "image_height": item["image_height"],
                **detection,
            }
            self._csv_writer.writerow(row)
        self._items_written += 1
        if self._items_written % 50 == 0:
            self._jsonl.flush()
            if self._csv is not None:
                self._csv.flush()

    def write_summary(self, summary: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        with (self.run_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    def close(self) -> None:
        if self._jsonl is not None:
            self._jsonl.close()
        if self._csv is not None:
            self._csv.close()

    def __enter__(self) -> "ReportWriter":
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def probability(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected a number, got {value!r}.") from exc
    if not 0.0 <= result <= 1.0:
        raise argparse.ArgumentTypeError("Value must be between 0 and 1.")
    return result


def positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected an integer, got {value!r}."
        ) from exc
    if result < 1:
        raise argparse.ArgumentTypeError("Value must be at least 1.")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect fire and smoke in an image, directory, video, URL, or webcam.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("source", nargs="?", default=str(DEFAULT_SOURCE))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--conf", type=probability, default=0.25)
    parser.add_argument("--iou", type=probability, default=0.45)
    parser.add_argument("--imgsz", type=positive_int, default=640)
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, mps, or one CUDA ID."
    )
    parser.add_argument("--batch", type=positive_int, default=1)
    parser.add_argument("--max-det", type=positive_int, default=300)
    parser.add_argument("--vid-stride", type=positive_int, default=1)
    parser.add_argument(
        "--classes", nargs="+", type=int, help="Optional class IDs to keep."
    )
    parser.add_argument("--agnostic-nms", action="store_true")
    parser.add_argument(
        "--show", action="store_true", help="Open a live display window."
    )
    parser.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save annotated images/video.",
    )
    parser.add_argument(
        "--reports",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write predictions.jsonl, detections.csv, and summary.json.",
    )
    parser.add_argument(
        "--save-txt", action="store_true", help="Also save YOLO text predictions."
    )
    parser.add_argument(
        "--save-conf",
        action="store_true",
        help="Include confidence in saved YOLO text.",
    )
    parser.add_argument(
        "--show-details", action="store_true", help="Print every bounding box."
    )
    parser.add_argument("--project", type=Path, default=SCRIPT_DIR / "runs/predict")
    parser.add_argument("--name", default="fire_smoke")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Reuse project/name instead of creating a numbered run directory.",
    )
    parser.add_argument("--line-width", type=positive_int)
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


def is_url(value: str) -> bool:
    return urlparse(value).scheme.lower() in {
        "http",
        "https",
        "rtsp",
        "rtmp",
        "tcp",
        "udp",
    }


def resolve_source(value: str) -> str | int:
    if is_url(value) or value.lower() in {"screen", "screenshot"}:
        return value

    path = Path(value).expanduser()
    if path.exists():
        return str(path.resolve())

    if value.isdecimal():
        return int(value)

    if any(character in value for character in "*?["):
        matches = glob.glob(value, recursive=True)
        if not matches:
            raise FileNotFoundError(f"Source glob matched no files: {value}")
        return value

    raise FileNotFoundError(
        f"Source not found: {path.resolve()}\nCurrent working directory: {Path.cwd()}"
    )


def resolve_model(value: str) -> str:
    path = Path(value).expanduser()
    if path.exists():
        if not path.is_file() and not path.is_dir():
            raise FileNotFoundError(
                f"Model path is not a file or model directory: {path}"
            )
        return str(path.resolve())

    if (
        path.is_absolute()
        or path.parent != Path(".")
        or path.name.lower() in {"best.pt", "last.pt"}
    ):
        raise FileNotFoundError(f"Model not found: {path.resolve()}")

    # Keep official model names such as yolo11s.pt or exported model identifiers.
    return value


def parse_device(value: str) -> str | int | None:
    text = value.strip().lower()
    if text == "auto":
        return None
    if "," in text:
        raise ValueError(
            "Prediction uses one device; provide one CUDA ID, cpu, or mps."
        )
    if text.lstrip("-").isdigit():
        return int(text)
    return text


def increment_run_dir(path: Path, exist_ok: bool) -> Path:
    if exist_ok or not path.exists():
        return path
    for index in range(2, 100_000):
        candidate = path.with_name(f"{path.name}_{index}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create a unique run directory near {path}")


def class_name(names: Any, class_id: int) -> str:
    if isinstance(names, Mapping):
        return str(names.get(class_id, class_id))
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        if 0 <= class_id < len(names):
            return str(names[class_id])
    return str(class_id)


def extract_detections(result: Any) -> tuple[list[Detection], int, int]:
    height, width = map(int, result.orig_shape)
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return [], width, height

    xyxy = boxes.xyxy.detach().cpu().tolist()
    confidences = boxes.conf.detach().cpu().tolist()
    class_ids = boxes.cls.detach().cpu().tolist()

    detections: list[Detection] = []
    for coordinates, confidence, raw_class_id in zip(
        xyxy, confidences, class_ids, strict=True
    ):
        x1, y1, x2, y2 = (float(value) for value in coordinates)
        class_id = int(raw_class_id)
        box_width = max(0.0, x2 - x1)
        box_height = max(0.0, y2 - y1)
        detections.append(
            Detection(
                class_id=class_id,
                class_name=class_name(result.names, class_id),
                confidence=float(confidence),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                x_center=(x1 + x2) / 2.0,
                y_center=(y1 + y2) / 2.0,
                box_width=box_width,
                box_height=box_height,
                x1_normalized=x1 / width,
                y1_normalized=y1 / height,
                x2_normalized=x2 / width,
                y2_normalized=y2 / height,
            )
        )
    return detections, width, height


def validate_class_filter(model: Any, classes: Sequence[int] | None) -> None:
    if classes is None:
        return
    names = model.names
    if isinstance(names, Mapping):
        valid = {int(key) for key in names}
    else:
        valid = set(range(len(names)))
    invalid = sorted(set(classes) - valid)
    if invalid:
        raise ValueError(
            f"Invalid --classes IDs {invalid}; valid IDs are {sorted(valid)}."
        )


def make_summary(
    *,
    status: str,
    started_at: str,
    model: str,
    source: str | int,
    run_dir: Path,
    args: argparse.Namespace,
    processed_items: int,
    items_with_detections: int,
    total_detections: int,
    class_counts: Counter[str],
) -> dict[str, Any]:
    return {
        "status": status,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "source": source,
        "run_dir": str(run_dir),
        "settings": {
            "confidence_threshold": args.conf,
            "iou_threshold": args.iou,
            "imgsz": args.imgsz,
            "device": args.device,
            "batch": args.batch,
            "max_detections": args.max_det,
            "video_stride": args.vid_stride,
            "classes": args.classes,
            "agnostic_nms": args.agnostic_nms,
        },
        "processed_items": processed_items,
        "items_with_detections": items_with_detections,
        "total_detections": total_detections,
        "class_counts": dict(sorted(class_counts.items())),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    writer: ReportWriter | None = None
    started_at = datetime.now(timezone.utc).isoformat()
    processed_items = 0
    items_with_detections = 0
    total_detections = 0
    total_class_counts: Counter[str] = Counter()
    status = "completed"

    try:
        if args.save_conf and not args.save_txt:
            raise ValueError("--save-conf requires --save-txt.")
        if not args.name or Path(args.name).name != args.name:
            raise ValueError("--name must be a single non-empty directory name.")

        model_reference = resolve_model(args.model)
        source = resolve_source(args.source)
        device = parse_device(args.device)

        project = args.project.expanduser().resolve()
        run_dir = increment_run_dir(project / args.name, args.exist_ok)
        if args.save or args.reports or args.save_txt:
            run_dir.mkdir(parents=True, exist_ok=True)

        from ultralytics import YOLO

        LOGGER.info("Loading model: %s", model_reference)
        model = YOLO(model_reference)
        validate_class_filter(model, args.classes)

        predict_kwargs: dict[str, Any] = {
            "source": source,
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
            "device": device,
            "batch": args.batch,
            "max_det": args.max_det,
            "vid_stride": args.vid_stride,
            "classes": args.classes,
            "agnostic_nms": args.agnostic_nms,
            "save": args.save,
            "show": args.show,
            "save_txt": args.save_txt,
            "save_conf": args.save_conf,
            "project": str(run_dir.parent),
            "name": run_dir.name,
            "exist_ok": True,
            "stream": True,
            "verbose": args.log_level == "DEBUG",
        }
        if args.line_width is not None:
            predict_kwargs["line_width"] = args.line_width

        LOGGER.info("Running prediction on: %s", source)
        writer = ReportWriter(run_dir, args.reports)
        writer.open()

        results = model.predict(**predict_kwargs)
        for item_index, result in enumerate(results):
            detections, width, height = extract_detections(result)
            processed_items += 1
            total_detections += len(detections)
            if detections:
                items_with_detections += 1

            item_counts = Counter(detection.class_name for detection in detections)
            total_class_counts.update(item_counts)
            source_name = str(getattr(result, "path", source))
            speed = {
                str(key): float(value)
                for key, value in (getattr(result, "speed", {}) or {}).items()
                if isinstance(value, (int, float)) and math.isfinite(float(value))
            }
            item = {
                "item_index": item_index,
                "source": source_name,
                "image_width": width,
                "image_height": height,
                "speed_ms": speed,
                "class_counts": dict(sorted(item_counts.items())),
                "detections": [asdict(detection) for detection in detections],
            }
            writer.write_item(item)

            if detections:
                counts_text = ", ".join(
                    f"{name}={count}" for name, count in sorted(item_counts.items())
                )
                LOGGER.info(
                    "[%d] %s: %d detection(s) (%s)",
                    item_index,
                    source_name,
                    len(detections),
                    counts_text,
                )
            else:
                LOGGER.info("[%d] %s: no detections", item_index, source_name)

            if args.show_details:
                for detection in detections:
                    LOGGER.info(
                        "    %s conf=%.3f box=(%.1f, %.1f, %.1f, %.1f)",
                        detection.class_name,
                        detection.confidence,
                        detection.x1,
                        detection.y1,
                        detection.x2,
                        detection.y2,
                    )

    except KeyboardInterrupt:
        status = "interrupted"
        LOGGER.warning("Prediction interrupted by user; saving the partial report.")
    except (FileNotFoundError, ValueError, RuntimeError, OSError, ImportError) as exc:
        status = "failed"
        LOGGER.error("%s", exc)
        return_code = 2
    except Exception as exc:  # Guard the CLI from third-party backend failures.
        status = "failed"
        if args.log_level == "DEBUG":
            LOGGER.exception("Prediction failed")
        else:
            LOGGER.error("Prediction failed (%s): %s", type(exc).__name__, exc)
        return_code = 2
    else:
        return_code = 0
    finally:
        run_dir_value = locals().get("run_dir", Path(args.project) / args.name)
        source_value = locals().get("source", args.source)
        model_value = locals().get("model_reference", args.model)
        summary = make_summary(
            status=status,
            started_at=started_at,
            model=str(model_value),
            source=source_value,
            run_dir=Path(run_dir_value),
            args=args,
            processed_items=processed_items,
            items_with_detections=items_with_detections,
            total_detections=total_detections,
            class_counts=total_class_counts,
        )
        if writer is not None:
            try:
                writer.write_summary(summary)
            finally:
                writer.close()

    LOGGER.info("Processed items:  %d", processed_items)
    LOGGER.info("Total detections: %d", total_detections)
    if args.save or args.reports or args.save_txt:
        LOGGER.info("Output directory: %s", run_dir)
    return 130 if status == "interrupted" else return_code


if __name__ == "__main__":
    raise SystemExit(main())