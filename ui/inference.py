"""In-memory YOLO inference normalized for the FireWatch browser contract."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any

from PIL import Image


class ModelUnavailableError(RuntimeError):
    """The configured model cannot be loaded safely."""


class InferenceFailure(RuntimeError):
    """Inference failed or returned malformed output."""


def _class_names(names: Any) -> dict[int, str]:
    if isinstance(names, Mapping):
        normalized = {}
        for key, value in names.items():
            if isinstance(key, bool) or not isinstance(key, int):
                raise InferenceFailure("The model class map is invalid.")
            if not isinstance(value, str):
                raise InferenceFailure("The model class map is invalid.")
            normalized[key] = value
    elif isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        if not all(isinstance(value, str) for value in names):
            raise InferenceFailure("The model class map is invalid.")
        normalized = {index: value for index, value in enumerate(names)}
    else:
        raise InferenceFailure("The model class map is invalid.")

    if set(normalized.values()) != {"smoke", "fire"}:
        raise InferenceFailure("The model must contain only smoke and fire classes.")
    return normalized


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise InferenceFailure(f"{label} must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise InferenceFailure(f"{label} must be a finite number.") from exc
    if not math.isfinite(number):
        raise InferenceFailure(f"{label} must be a finite number.")
    return number


def normalize_prediction(
    names: Any,
    xyxy: Sequence[Sequence[Any]],
    confidences: Sequence[Any],
    class_ids: Sequence[Any],
    width: Any,
    height: Any,
) -> dict[str, Any]:
    """Convert detector arrays into the strict result shape used by the UI."""

    class_names = _class_names(names)
    image_width = _finite_number(width, "Image width")
    image_height = _finite_number(height, "Image height")
    if image_width <= 0 or image_height <= 0:
        raise InferenceFailure("Image dimensions must be positive.")
    if not (len(xyxy) == len(confidences) == len(class_ids)):
        raise InferenceFailure("Model detection arrays have different lengths.")
    if not xyxy:
        return {
            "category": "clear",
            "confidence": None,
            "detectionCount": 0,
            "box": None,
        }

    detections = []
    for coordinates, raw_confidence, raw_class_id in zip(
        xyxy, confidences, class_ids
    ):
        if len(coordinates) != 4:
            raise InferenceFailure("A model bounding box must have four coordinates.")
        x1, y1, x2, y2 = (
            _finite_number(value, "Bounding-box coordinate") for value in coordinates
        )
        confidence = _finite_number(raw_confidence, "Confidence")
        class_id_number = _finite_number(raw_class_id, "Class ID")
        if not 0 <= confidence <= 1:
            raise InferenceFailure("Confidence must be between zero and one.")
        if not class_id_number.is_integer():
            raise InferenceFailure("Class ID must be an integer.")
        class_id = int(class_id_number)
        if class_id not in class_names:
            raise InferenceFailure("The model returned an unknown class ID.")

        x1 = min(max(x1, 0.0), image_width)
        y1 = min(max(y1, 0.0), image_height)
        x2 = min(max(x2, 0.0), image_width)
        y2 = min(max(y2, 0.0), image_height)
        if x2 <= x1 or y2 <= y1:
            raise InferenceFailure("The model returned an empty bounding box.")

        detections.append(
            {
                "category": class_names[class_id],
                "confidence": confidence,
                "box": {
                    "x": x1 / image_width,
                    "y": y1 / image_height,
                    "width": (x2 - x1) / image_width,
                    "height": (y2 - y1) / image_height,
                },
            }
        )

    best = max(detections, key=lambda detection: detection["confidence"])
    return {
        "category": best["category"],
        "confidence": best["confidence"],
        "detectionCount": len(detections),
        "box": best["box"],
    }


class InferenceService:
    """Lazily load one YOLO model and run artifact-free image prediction."""

    def __init__(self, model_path: Path, model_loader: Any = None) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self._model_loader = (
            model_loader
            if model_loader is not None
            else self._default_model_loader
        )
        self._model = None
        self._load_lock = Lock()
        self._predict_lock = Lock()

    @staticmethod
    def _default_model_loader(model_path: Path) -> Any:
        from ultralytics import YOLO

        return YOLO(str(model_path))

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            if not self.model_path.is_file():
                raise ModelUnavailableError(
                    "The trained FireWatch model is not available."
                )
            try:
                candidate = self._model_loader(self.model_path)
                _class_names(candidate.names)
            except InferenceFailure as exc:
                raise ModelUnavailableError(
                    "The trained FireWatch model is incompatible."
                ) from exc
            except Exception as exc:
                raise ModelUnavailableError(
                    "The trained FireWatch model could not be loaded."
                ) from exc
            self._model = candidate
            return self._model

    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        """Run one in-memory prediction and return the browser result fields."""

        model = self._get_model()
        try:
            with Image.open(BytesIO(image_bytes)) as opened:
                image = opened.convert("RGB")
            with self._predict_lock:
                results = list(
                    model.predict(
                        source=image,
                        conf=0.25,
                        iou=0.45,
                        imgsz=640,
                        device=None,
                        max_det=300,
                        save=False,
                        save_txt=False,
                        save_crop=False,
                        verbose=False,
                    )
                )
            if len(results) != 1:
                raise InferenceFailure(
                    "The model returned an unexpected number of results."
                )
            result = results[0]
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                xyxy, confidences, class_ids = [], [], []
            else:
                xyxy = boxes.xyxy.detach().cpu().tolist()
                confidences = boxes.conf.detach().cpu().tolist()
                class_ids = boxes.cls.detach().cpu().tolist()
            height, width = result.orig_shape
            return normalize_prediction(
                getattr(result, "names", model.names),
                xyxy,
                confidences,
                class_ids,
                width,
                height,
            )
        except InferenceFailure:
            raise
        except Exception as exc:
            raise InferenceFailure(
                "The image could not be analyzed by the trained model."
            ) from exc
