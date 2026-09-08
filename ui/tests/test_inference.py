import math
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

UI_DIR = Path(__file__).resolve().parents[1]
if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))

from inference import (
    InferenceFailure,
    InferenceService,
    ModelUnavailableError,
    normalize_prediction,
)


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.value


class FakeBoxes:
    def __init__(self):
        self.xyxy = FakeTensor([[10, 20, 80, 90]])
        self.conf = FakeTensor([0.88])
        self.cls = FakeTensor([1])

    def __len__(self):
        return 1


class FakeResult:
    names = {0: "smoke", 1: "fire"}
    orig_shape = (100, 100)
    boxes = FakeBoxes()


class FakeModel:
    def __init__(self, names=None, failure=None, results=None):
        self.names = names if names is not None else {0: "smoke", 1: "fire"}
        self.failure = failure
        self.results = results if results is not None else [FakeResult()]
        self.predict_calls = []

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        if self.failure:
            raise self.failure
        return self.results


class OverlapDetectingModel(FakeModel):
    def __init__(self):
        super().__init__()
        self._active = 0
        self._active_lock = threading.Lock()
        self.overlapped = False

    def predict(self, **kwargs):
        with self._active_lock:
            self._active += 1
            if self._active > 1:
                self.overlapped = True
        try:
            time.sleep(0.05)
            return super().predict(**kwargs)
        finally:
            with self._active_lock:
                self._active -= 1


class FalseyModelLoader:
    def __init__(self, model):
        self.model = model
        self.calls = []

    def __bool__(self):
        return False

    def __call__(self, path):
        self.calls.append(path)
        return self.model


def valid_jpeg_bytes():
    buffer = BytesIO()
    Image.new("RGB", (100, 100), (180, 30, 20)).save(buffer, format="JPEG")
    return buffer.getvalue()


class NormalizePredictionTests(unittest.TestCase):
    def test_no_detections_returns_clear(self):
        self.assertEqual(
            normalize_prediction({0: "smoke", 1: "fire"}, [], [], [], 640, 480),
            {
                "category": "clear",
                "confidence": None,
                "detectionCount": 0,
                "box": None,
            },
        )

    def test_highest_confidence_sets_display_result_and_all_detections_are_counted(self):
        prediction = normalize_prediction(
            {0: "smoke", 1: "fire"},
            [[-10, 20, 330, 500], [100, 50, 200, 150]],
            [0.70, 0.92],
            [0, 1],
            640,
            480,
        )

        self.assertEqual(prediction["category"], "fire")
        self.assertEqual(prediction["confidence"], 0.92)
        self.assertEqual(prediction["detectionCount"], 2)
        self.assertEqual(
            prediction["box"],
            {
                "x": 0.15625,
                "y": 50 / 480,
                "width": 0.15625,
                "height": 100 / 480,
            },
        )

    def test_sequence_class_names_are_supported(self):
        prediction = normalize_prediction(
            ["smoke", "fire"],
            [[64, 48, 320, 240]],
            [0.75],
            [0],
            640,
            480,
        )
        self.assertEqual(prediction["category"], "smoke")

    def test_malformed_model_outputs_are_rejected(self):
        cases = (
            ({"names": {0: "person", 1: "fire"}},),
            ({"names": {0: "SMOKE", 1: "FIRE"}},),
            ({"names": {0: 123, 1: "fire"}},),
            ({"names": {0.2: "smoke", 1.2: "fire"}},),
            ({"names": {0.2: "smoke", 0.8: "fire"}},),
            ({"xyxy": [[0, 0, 10, 10]], "confidences": [], "class_ids": [0]},),
            ({"width": 0},),
            ({"confidences": [math.inf]},),
            ({"xyxy": [[10, 10, 10, 20]]},),
            ({"class_ids": [9]},),
        )
        defaults = {
            "names": {0: "smoke", 1: "fire"},
            "xyxy": [[0, 0, 10, 10]],
            "confidences": [0.8],
            "class_ids": [0],
            "width": 100,
            "height": 100,
        }

        for (overrides,) in cases:
            with self.subTest(overrides=overrides):
                values = {**defaults, **overrides}
                with self.assertRaises(InferenceFailure):
                    normalize_prediction(**values)


class InferenceServiceTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.model_path = Path(self.temporary_directory.name) / "model.pt"
        self.model_path.write_bytes(b"checkpoint placeholder")

    def test_model_is_loaded_once_and_prediction_disables_artifact_output(self):
        model = FakeModel()
        load_calls = []

        def loader(path):
            load_calls.append(path)
            return model

        service = InferenceService(self.model_path, model_loader=loader)
        first = service.predict(valid_jpeg_bytes())
        second = service.predict(valid_jpeg_bytes())

        self.assertEqual(len(load_calls), 1)
        self.assertEqual(first, second)
        self.assertEqual(first["category"], "fire")
        self.assertEqual(first["detectionCount"], 1)
        settings = model.predict_calls[0]
        self.assertEqual(settings["conf"], 0.25)
        self.assertEqual(settings["iou"], 0.45)
        self.assertEqual(settings["imgsz"], 640)
        self.assertEqual(settings["max_det"], 300)
        self.assertIsNone(settings["device"])
        for key in ("save", "save_txt", "save_crop", "verbose"):
            self.assertFalse(settings[key])

    def test_concurrent_first_predictions_load_the_model_once(self):
        model = FakeModel()
        start = threading.Barrier(2)
        load_calls = []
        count_lock = threading.Lock()

        def loader(path):
            with count_lock:
                load_calls.append(path)
            time.sleep(0.05)
            return model

        service = InferenceService(self.model_path, model_loader=loader)

        def predict_together():
            start.wait()
            return service.predict(valid_jpeg_bytes())

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _index: predict_together(), range(2)))

        self.assertEqual(len(load_calls), 1)
        self.assertEqual(results[0], results[1])

    def test_predictions_on_the_shared_model_do_not_overlap(self):
        model = OverlapDetectingModel()
        service = InferenceService(
            self.model_path, model_loader=lambda _path: model
        )
        start = threading.Barrier(2)

        def predict_together():
            start.wait()
            return service.predict(valid_jpeg_bytes())

        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(lambda _index: predict_together(), range(2)))

        self.assertFalse(model.overlapped)

    def test_clear_and_malformed_service_results_are_handled_safely(self):
        clear_result = SimpleNamespace(
            names={0: "smoke", 1: "fire"},
            orig_shape=(100, 100),
            boxes=None,
        )
        service = InferenceService(
            self.model_path,
            model_loader=lambda _path: FakeModel(results=[clear_result]),
        )
        self.assertEqual(
            service.predict(valid_jpeg_bytes()),
            {
                "category": "clear",
                "confidence": None,
                "detectionCount": 0,
                "box": None,
            },
        )

        malformed_results = (
            [],
            [FakeResult(), FakeResult()],
            [
                SimpleNamespace(
                    names={0: "smoke", 1: "fire"},
                    orig_shape=(0, 100),
                    boxes=FakeBoxes(),
                )
            ],
            [
                SimpleNamespace(
                    names={0: "smoke", 1: "fire"},
                    orig_shape=(100, 100),
                    boxes=SimpleNamespace(),
                )
            ],
        )
        for results in malformed_results:
            with self.subTest(result_count=len(results)):
                service = InferenceService(
                    self.model_path,
                    model_loader=lambda _path, results=results: FakeModel(
                        results=results
                    ),
                )
                with self.assertRaises(InferenceFailure):
                    service.predict(valid_jpeg_bytes())

    def test_missing_checkpoint_is_reported_as_model_unavailable(self):
        service = InferenceService(
            Path(self.temporary_directory.name) / "missing.pt",
            model_loader=lambda _path: FakeModel(),
        )
        with self.assertRaisesRegex(ModelUnavailableError, "not available"):
            service.predict(valid_jpeg_bytes())

    def test_falsey_injected_model_loader_is_still_used(self):
        loader = FalseyModelLoader(FakeModel())
        service = InferenceService(self.model_path, model_loader=loader)
        result = service.predict(valid_jpeg_bytes())
        self.assertEqual(result["category"], "fire")
        self.assertEqual(loader.calls, [self.model_path.resolve()])

    def test_incompatible_model_classes_are_reported_as_unavailable(self):
        service = InferenceService(
            self.model_path,
            model_loader=lambda _path: FakeModel(
                names={0: "person", 1: "fire"}
            ),
        )
        with self.assertRaisesRegex(ModelUnavailableError, "incompatible"):
            service.predict(valid_jpeg_bytes())

    def test_loader_and_prediction_failures_hide_third_party_details(self):
        service = InferenceService(
            self.model_path,
            model_loader=lambda _path: (_ for _ in ()).throw(
                RuntimeError("secret loader detail")
            ),
        )
        with self.assertRaises(ModelUnavailableError) as context:
            service.predict(valid_jpeg_bytes())
        self.assertNotIn("secret loader detail", str(context.exception))

        service = InferenceService(
            self.model_path,
            model_loader=lambda _path: FakeModel(
                failure=RuntimeError("secret prediction detail")
            ),
        )
        with self.assertRaises(InferenceFailure) as context:
            service.predict(valid_jpeg_bytes())
        self.assertNotIn("secret prediction detail", str(context.exception))


if __name__ == "__main__":
    unittest.main()
