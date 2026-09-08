import tempfile
import unittest
import sys
import os
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml

import train_yolo_v2
from train_yolo_v2 import ConfigError, DatasetError, load_config, validate_dataset, validate_label


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_default_config_has_exact_values_and_resolved_paths(self):
        config_path = PROJECT_ROOT / "train_config_v2.yaml"

        config = load_config(config_path)

        self.assertEqual(config["version"], 2)
        self.assertEqual(config["model"], (PROJECT_ROOT / "yolo11s.pt").resolve())
        self.assertEqual(config["data"], (PROJECT_ROOT / "data_v2.yaml").resolve())
        self.assertEqual(config["project"], (PROJECT_ROOT / "runs" / "detect").resolve())
        self.assertEqual(config["name"], "fire_smoke_v2")
        self.assertEqual(
            config["train"],
            {
                "epochs": 80, "imgsz": 640, "batch": 8, "device": 0, "workers": 2,
                "patience": 15, "optimizer": "AdamW", "lr0": 0.001, "lrf": 0.01,
                "cos_lr": True, "pretrained": True, "amp": True, "cache": False,
                "seed": 42, "deterministic": True, "save_period": 5, "val": True,
                "plots": True, "degrees": 5.0, "translate": 0.1, "scale": 0.4,
                "shear": 2.0, "perspective": 0.0, "flipud": 0.0, "fliplr": 0.5,
                "hsv_h": 0.015, "hsv_s": 0.6, "hsv_v": 0.4, "mosaic": 0.8,
                "mixup": 0.05, "close_mosaic": 10,
            },
        )
        self.assertEqual(
            config["evaluation"],
            {"split": "test", "imgsz": 640, "batch": 8, "device": 0, "half": True, "plots": True},
        )

    def test_load_config_rejects_missing_unknown_and_invalid_values(self):
        base = {
            "version": 2, "model": "model.pt", "data": "data.yaml", "project": "runs",
            "name": "run", "train": {"epochs": 1, "imgsz": 1, "batch": 1, "device": 0,
            "workers": 0, "patience": 0, "optimizer": "AdamW", "lr0": 0.001, "lrf": 0.01,
            "cos_lr": True, "pretrained": True, "amp": True, "cache": False, "seed": 0,
            "deterministic": True, "save_period": 0, "val": True, "plots": True, "degrees": 0.0,
            "translate": 0.0, "scale": 0.0, "shear": 0.0, "perspective": 0.0, "flipud": 0.0,
            "fliplr": 0.0, "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0, "mosaic": 0.0,
            "mixup": 0.0, "close_mosaic": 0},
            "evaluation": {"split": "test", "imgsz": 1, "batch": 1, "device": 0, "half": True, "plots": True},
        }
        cases = []
        missing = dict(base)
        del missing["name"]
        cases.append(("missing", missing))
        unknown = dict(base)
        unknown["unknown"] = "value"
        cases.append(("unknown top-level", unknown))
        invalid_version = dict(base)
        invalid_version["version"] = 3
        cases.append(("version", invalid_version))
        invalid_train = dict(base)
        invalid_train["train"] = dict(base["train"], epochs=0)
        cases.append(("positive", invalid_train))
        invalid_probability = dict(base)
        invalid_probability["train"] = dict(base["train"], mixup=1.1)
        cases.append(("probability", invalid_probability))
        invalid_boolean = dict(base)
        invalid_boolean["train"] = dict(base["train"], amp="yes")
        cases.append(("boolean", invalid_boolean))
        invalid_nested_key = dict(base)
        invalid_nested_key["evaluation"] = dict(base["evaluation"], extra=True)
        cases.append(("unknown nested", invalid_nested_key))
        invalid_split = dict(base)
        invalid_split["evaluation"] = dict(base["evaluation"], split="validation")
        cases.append(("evaluation split", invalid_split))
        invalid_optimizer = dict(base)
        invalid_optimizer["train"] = dict(base["train"], optimizer="NotAnOptimizer")
        cases.append(("optimizer", invalid_optimizer))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            for label, candidate in cases:
                with self.subTest(label=label):
                    path.write_text(yaml.safe_dump(candidate), encoding="utf-8")
                    with self.assertRaises(ConfigError):
                        load_config(path)

    def test_load_config_normalizes_optimizer_case_insensitively(self):
        config = yaml.safe_load((PROJECT_ROOT / "train_config_v2.yaml").read_text(encoding="utf-8"))
        config["train"]["optimizer"] = "nadam"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")

            loaded = load_config(path)

        self.assertEqual(loaded["train"]["optimizer"], "NAdam")


class LabelTests(unittest.TestCase):
    def test_validate_label_counts_positive_boxes_and_accepts_empty_file(self):
        with tempfile.TemporaryDirectory() as directory:
            label = Path(directory) / "example.txt"
            label.write_text("0 0.5 0.5 0.2 0.3\n1 1 0 1 1\n", encoding="utf-8")
            self.assertEqual(validate_label(label, {0, 1}), (2, {0: 1, 1: 1}))
            label.write_text("", encoding="utf-8")
            self.assertEqual(validate_label(label, {0, 1}), (0, {}))

    def test_validate_label_rejects_invalid_rows_with_file_and_line(self):
        cases = {
            "too few fields": "0 0.5 0.5 0.2\n",
            "noninteger class": "zero 0.5 0.5 0.2 0.2\n",
            "unknown class": "2 0.5 0.5 0.2 0.2\n",
            "nonnumeric coordinate": "0 x 0.5 0.2 0.2\n",
            "nonfinite coordinate": "0 nan 0.5 0.2 0.2\n",
            "out of range": "0 1.1 0.5 0.2 0.2\n",
            "zero width": "0 0.5 0.5 0 0.2\n",
            "zero height": "0 0.5 0.5 0.2 0\n",
        }
        with tempfile.TemporaryDirectory() as directory:
            label = Path(directory) / "bad.txt"
            for case, contents in cases.items():
                with self.subTest(case=case):
                    label.write_text(contents, encoding="utf-8")
                    with self.assertRaisesRegex(DatasetError, r"bad\.txt.*line 1"):
                        validate_label(label, {0, 1})


class DatasetTests(unittest.TestCase):
    def _make_dataset(self, directory, names):
        root = Path(directory) / "dataset"
        for split in ("train", "val", "test"):
            images = root / split / "images"
            labels = root / split / "labels"
            images.mkdir(parents=True)
            labels.mkdir()
            (images / (split + ".JpG")).write_bytes(b"")
            (labels / (split + ".txt")).write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        data_path = Path(directory) / "data.yaml"
        data_path.write_text(yaml.safe_dump({"path": "dataset", "names": names,
                                             "train": "train/images", "val": "val/images",
                                             "test": "test/images"}), encoding="utf-8")
        return data_path, root

    def test_validate_dataset_supports_mapping_names_and_summarizes_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path, root = self._make_dataset(directory, {0: "fire", 1: "smoke"})
            (root / "test" / "images" / "negative.png").write_bytes(b"")
            (root / "test" / "labels" / "negative.txt").write_text("", encoding="utf-8")

            summary = validate_dataset(data_path)

            self.assertEqual(summary["train"], {"images": 1, "boxes": 1, "negative_images": 0, "class_counts": {0: 1}})
            self.assertEqual(summary["test"], {"images": 2, "boxes": 1, "negative_images": 1, "class_counts": {0: 1}})

    def test_validate_dataset_supports_list_names_and_requires_every_label(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path, root = self._make_dataset(directory, ["fire", "smoke"])
            (root / "val" / "labels" / "val.txt").unlink()

            with self.assertRaisesRegex(DatasetError, r"val\.txt"):
                validate_dataset(data_path)

    def test_validate_dataset_rejects_split_outside_declared_root(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path, _ = self._make_dataset(directory, ["fire"])
            outside_images = Path(directory) / "outside" / "images"
            outside_labels = Path(directory) / "outside" / "labels"
            outside_images.mkdir(parents=True)
            outside_labels.mkdir()
            (outside_images / "image.jpg").write_bytes(b"")
            (outside_labels / "image.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
            data["train"] = "../outside/images"
            data_path.write_text(yaml.safe_dump(data), encoding="utf-8")

            with self.assertRaisesRegex(DatasetError, r"outside.*dataset root"):
                validate_dataset(data_path)

    def test_v2_descriptor_without_path_is_independent_of_process_cwd(self):
        with tempfile.TemporaryDirectory() as project_directory, \
             tempfile.TemporaryDirectory() as unrelated_directory:
            project = Path(project_directory)
            for split in ("train", "val", "test"):
                images = project / "data" / split / "images"
                labels = project / "data" / split / "labels"
                images.mkdir(parents=True)
                labels.mkdir()
                (images / (split + ".jpg")).write_bytes(b"")
                (labels / (split + ".txt")).write_text(
                    "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
                )
            (project / "data_v2.yaml").write_text(
                "train: data/train/images\n"
                "val: data/val/images\n"
                "test: data/test/images\n\n"
                "names:\n  0: smoke\n  1: fire\n",
                encoding="utf-8",
            )
            config = yaml.safe_load(
                (PROJECT_ROOT / "train_config_v2.yaml").read_text(encoding="utf-8")
            )
            (project / "train_config_v2.yaml").write_text(
                yaml.safe_dump(config), encoding="utf-8"
            )

            original_cwd = Path.cwd()
            try:
                os.chdir(unrelated_directory)
                loaded = load_config(project / "train_config_v2.yaml")
                summary = validate_dataset(loaded["data"])
            finally:
                os.chdir(str(original_cwd))

            self.assertEqual(loaded["data"], (project / "data_v2.yaml").resolve())
            self.assertEqual(
                {split: summary[split]["images"] for split in ("train", "val", "test")},
                {"train": 1, "val": 1, "test": 1},
            )

    def test_validate_dataset_rejects_explicit_null_path(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path, _ = self._make_dataset(directory, ["fire"])
            data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
            data["path"] = None
            data_path.write_text(yaml.safe_dump(data), encoding="utf-8")

            with self.assertRaisesRegex(DatasetError, r"path.*non-empty string"):
                validate_dataset(data_path)

    def test_validate_dataset_rejects_noncontiguous_mapping_class_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path, _ = self._make_dataset(directory, {0: "smoke", 2: "fire"})

            with self.assertRaisesRegex(DatasetError, r"contiguous.*0"):
                validate_dataset(data_path)

    def test_validate_dataset_rejects_orphan_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path, root = self._make_dataset(directory, ["fire"])
            (root / "train" / "labels" / "orphan.txt").write_text("", encoding="utf-8")

            with self.assertRaisesRegex(DatasetError, r"[Oo]rphan.*orphan\.txt"):
                validate_dataset(data_path)


class TrainingOrchestrationTests(unittest.TestCase):
    def _config(self, directory):
        root = Path(directory)
        return {
            "model": root / "model.pt",
            "data": root / "data.yaml",
            "project": root / "runs",
            "name": "fire_smoke_v2",
            "train": {"epochs": 3, "imgsz": 640, "batch": 8, "device": 0, "workers": 2},
            "evaluation": {"split": "test", "imgsz": 640, "batch": 8,
                           "device": 0, "half": True, "plots": True},
        }

    def test_validate_cuda_rejects_unavailable_and_normalizes_valid_device(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(train_yolo_v2.RuntimeEnvironmentError, r"device 0"):
                train_yolo_v2.validate_cuda("0")

        properties = SimpleNamespace(total_memory=8 * 1024 ** 3)
        with mock.patch("torch.cuda.is_available", return_value=True), \
             mock.patch("torch.cuda.device_count", return_value=2), \
             mock.patch("torch.cuda.get_device_name", return_value="Test GPU"), \
             mock.patch("torch.cuda.get_device_properties", return_value=properties):
            details = train_yolo_v2.validate_cuda("1")

        self.assertEqual(details, {"index": 1, "name": "Test GPU", "vram_gb": 8.0})

    def test_validate_cuda_rejects_invalid_and_out_of_range_devices(self):
        with self.assertRaisesRegex(train_yolo_v2.RuntimeEnvironmentError, r"cpu"):
            train_yolo_v2.validate_cuda("cpu")
        with self.assertRaisesRegex(train_yolo_v2.RuntimeEnvironmentError, r"-1"):
            train_yolo_v2.validate_cuda(-1)
        with mock.patch("torch.cuda.is_available", return_value=True), \
             mock.patch("torch.cuda.device_count", return_value=1):
            with self.assertRaisesRegex(train_yolo_v2.RuntimeEnvironmentError, r"device 1"):
                train_yolo_v2.validate_cuda(1)

    def test_dependency_loaders_translate_missing_imports_to_runtime_errors(self):
        with mock.patch.dict(sys.modules, {"torch": None}):
            with self.assertRaisesRegex(train_yolo_v2.RuntimeEnvironmentError, r"PyTorch"):
                train_yolo_v2.validate_cuda(0)
        with mock.patch.dict(sys.modules, {"ultralytics": None}):
            with self.assertRaisesRegex(train_yolo_v2.RuntimeEnvironmentError, r"Ultralytics"):
                train_yolo_v2._load_yolo_class()

    def test_build_train_args_copies_all_training_values_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory)
            original_train = dict(config["train"])

            arguments = train_yolo_v2.build_train_args(config)

            self.assertEqual(arguments, dict(original_train, data=str(config["data"]),
                                             project=str(config["project"]), name=config["name"],
                                             exist_ok=False))
            self.assertEqual(config["train"], original_train)

    def test_cli_propagates_batch_and_device_overrides_and_forwards_resume_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory)
            captured = {}
            invocation = {}

            def capture_run(value, resume=None, resume_batch=None):
                captured.update(value)
                invocation.update({"resume": resume, "resume_batch": resume_batch})
                return Path("best.pt")

            with mock.patch.object(train_yolo_v2, "load_config", return_value=config), \
                 mock.patch.object(train_yolo_v2, "validate_dataset", return_value={"train": {"images": 1, "boxes": 1, "negative_images": 0, "class_counts": {0: 1}}, "val": {"images": 1, "boxes": 1, "negative_images": 0, "class_counts": {0: 1}}, "test": {"images": 1, "boxes": 1, "negative_images": 0, "class_counts": {0: 1}}}), \
                 mock.patch.object(train_yolo_v2, "validate_cuda", return_value={"index": 1, "name": "GPU", "vram_gb": 8.0}), \
                 mock.patch.object(train_yolo_v2, "run_training", side_effect=capture_run):
                with redirect_stdout(StringIO()):
                    self.assertEqual(train_yolo_v2.main(["--config", "ignored.yaml", "--epochs", "5", "--batch", "3", "--device", "1", "--name", "overridden", "--resume", "checkpoint.pt"]), 0)

            self.assertEqual(captured["train"]["epochs"], 5)
            self.assertEqual(captured["train"]["batch"], 3)
            self.assertEqual(captured["evaluation"]["batch"], 3)
            self.assertEqual(captured["train"]["device"], 1)
            self.assertEqual(captured["evaluation"]["device"], 1)
            self.assertEqual(captured["name"], "overridden")
            self.assertEqual(invocation["resume"], Path("checkpoint.pt"))
            self.assertEqual(invocation["resume_batch"], 3)
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
                train_yolo_v2.main(["--epochs", "0"])
            self.assertEqual(error.exception.code, 2)

    def test_check_only_uses_real_dataset_and_never_loads_ultralytics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            for split in ("train", "val", "test"):
                images = dataset / split / "images"
                labels = dataset / split / "labels"
                images.mkdir(parents=True)
                labels.mkdir()
                (images / (split + ".jpg")).write_bytes(b"")
                (labels / (split + ".txt")).write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            data_path = root / "data.yaml"
            data_path.write_text(yaml.safe_dump({"path": "dataset", "names": ["fire"],
                                                  "train": "train/images", "val": "val/images",
                                                  "test": "test/images"}), encoding="utf-8")
            config_path = root / "config.yaml"
            config = yaml.safe_load((PROJECT_ROOT / "train_config_v2.yaml").read_text(encoding="utf-8"))
            config["model"] = "model.pt"
            config["data"] = "data.yaml"
            config["project"] = "runs"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            output = StringIO()
            with mock.patch.object(train_yolo_v2, "validate_cuda", return_value={"index": 0, "name": "Test GPU", "vram_gb": 8.0}), \
                 mock.patch.object(train_yolo_v2, "_load_yolo_class", side_effect=AssertionError("must remain lazy")):
                with redirect_stdout(output):
                    self.assertEqual(train_yolo_v2.main(["--config", str(config_path), "--check-only"]), 0)

            self.assertIn("Preflight checks passed", output.getvalue())
            self.assertFalse((root / "runs").exists())

    def test_fresh_training_uses_best_checkpoint_for_test_evaluation_and_prints_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory)
            config["model"].write_bytes(b"model")
            best = Path(directory) / "best.pt"
            best.write_bytes(b"best")

            class FakeYOLO:
                instances = []

                def __init__(self, path):
                    self.path = path
                    self.trainer = SimpleNamespace(best=best)
                    self.train_arguments = None
                    self.val_arguments = None
                    self.instances.append(self)

                def train(self, **arguments):
                    self.train_arguments = arguments

                def val(self, **arguments):
                    self.val_arguments = arguments
                    return SimpleNamespace(box=SimpleNamespace(mp=0.1, mr=0.2, map50=0.3, map=0.4))

            output = StringIO()
            with mock.patch.object(train_yolo_v2, "_load_yolo_class", return_value=FakeYOLO):
                with redirect_stdout(output):
                    returned = train_yolo_v2.run_training(config)

            self.assertEqual(returned, best)
            self.assertEqual(FakeYOLO.instances[0].path, str(config["model"]))
            self.assertEqual(FakeYOLO.instances[0].train_arguments, train_yolo_v2.build_train_args(config))
            self.assertEqual(FakeYOLO.instances[1].path, str(best))
            self.assertEqual(FakeYOLO.instances[1].val_arguments,
                             dict(config["evaluation"], data=str(config["data"]), project=str(config["project"]),
                                  name="fire_smoke_v2_test", exist_ok=False))
            self.assertIn("Precision: 0.1", output.getvalue())
            self.assertIn("Recall: 0.2", output.getvalue())
            self.assertIn("mAP50: 0.3", output.getvalue())
            self.assertIn("mAP50-95: 0.4", output.getvalue())
            self.assertIn("Best checkpoint: {}".format(best), output.getvalue())

    def test_resume_is_explicit_and_missing_model_or_checkpoint_fails_actionably(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory)
            with self.assertRaisesRegex(ConfigError, r"model"):
                train_yolo_v2.run_training(config)
            checkpoint = Path(directory) / "resume.pt"
            with self.assertRaisesRegex(ConfigError, r"resume"):
                train_yolo_v2.run_training(config, checkpoint)

            checkpoint.write_bytes(b"checkpoint")
            best = Path(directory) / "best.pt"
            best.write_bytes(b"best")
            (config["project"] / config["name"]).mkdir(parents=True)
            (config["project"] / (config["name"] + "_test")).mkdir()

            class FakeYOLO:
                instances = []

                def __init__(self, path):
                    self.path = path
                    self.trainer = SimpleNamespace(best=best)
                    self.arguments = None
                    self.instances.append(self)

                def train(self, **arguments):
                    self.arguments = arguments

            with mock.patch.object(train_yolo_v2, "_load_yolo_class", return_value=FakeYOLO):
                self.assertEqual(train_yolo_v2.run_training(config, checkpoint), best)
                self.assertEqual(
                    train_yolo_v2.run_training(config, checkpoint, resume_batch=3), best
                )
            self.assertEqual(FakeYOLO.instances[0].path, str(checkpoint))
            self.assertEqual(FakeYOLO.instances[0].arguments, {"resume": True, "device": 0})
            self.assertEqual(FakeYOLO.instances[1].path, str(checkpoint))
            self.assertEqual(
                FakeYOLO.instances[1].arguments,
                {"resume": True, "device": 0, "batch": 3},
            )

    def test_fresh_training_rejects_existing_train_or_test_output_directory(self):
        for output_name in ("fire_smoke_v2", "fire_smoke_v2_test"):
            with self.subTest(output_name=output_name), tempfile.TemporaryDirectory() as directory:
                config = self._config(directory)
                config["model"].write_bytes(b"model")
                (config["project"] / output_name).mkdir(parents=True)

                with mock.patch.object(
                    train_yolo_v2, "_load_yolo_class", side_effect=AssertionError("must fail before loading")
                ), self.assertRaisesRegex(
                    train_yolo_v2.RuntimeEnvironmentError, r"already exists.*--name.*resume"
                ):
                    train_yolo_v2.run_training(config)

    def test_cli_returns_expected_oom_and_interrupt_exit_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory)
            common = (mock.patch.object(train_yolo_v2, "load_config", return_value=config),
                      mock.patch.object(train_yolo_v2, "validate_dataset", return_value={"train": {"images": 1, "boxes": 1, "negative_images": 0, "class_counts": {0: 1}}, "val": {"images": 1, "boxes": 1, "negative_images": 0, "class_counts": {0: 1}}, "test": {"images": 1, "boxes": 1, "negative_images": 0, "class_counts": {0: 1}}}),
                      mock.patch.object(train_yolo_v2, "validate_cuda", return_value={"index": 0, "name": "GPU", "vram_gb": 8.0}))
            for patch in common:
                patch.start()
            self.addCleanup(lambda: [patch.stop() for patch in reversed(common)])
            fresh_oom_errors = StringIO()
            with mock.patch.object(train_yolo_v2, "run_training", side_effect=RuntimeError("CUDA out of memory")), redirect_stdout(StringIO()), redirect_stderr(fresh_oom_errors):
                self.assertEqual(train_yolo_v2.main(["--batch", "3"]), 3)
            fresh_message = fresh_oom_errors.getvalue()
            planned_checkpoint = config["project"] / config["name"] / "weights" / "last.pt"
            self.assertIn(str(planned_checkpoint), fresh_message)
            self.assertIn("--resume", fresh_message)
            self.assertIn("--batch 1", fresh_message)
            self.assertRegex(fresh_message, r"distinct.*--name|--name.*distinct")

            resume_run = Path(directory) / "previous_run"
            checkpoint = resume_run / "weights" / "last.pt"
            explicit_resume_errors = StringIO()
            with mock.patch.object(train_yolo_v2, "run_training", side_effect=RuntimeError("CUDA out of memory")), redirect_stdout(StringIO()), redirect_stderr(explicit_resume_errors):
                self.assertEqual(
                    train_yolo_v2.main(["--resume", str(checkpoint), "--batch", "6"]), 3
                )
            explicit_message = explicit_resume_errors.getvalue()
            self.assertIn(str(checkpoint.resolve()), explicit_message)
            self.assertIn("explicitly requested batch 6", explicit_message)
            self.assertIn("--batch 3", explicit_message)

            implicit_resume_errors = StringIO()
            with mock.patch.object(train_yolo_v2, "run_training", side_effect=RuntimeError("CUDA out of memory")), redirect_stdout(StringIO()), redirect_stderr(implicit_resume_errors):
                self.assertEqual(train_yolo_v2.main(["--resume", str(checkpoint)]), 3)
            implicit_message = implicit_resume_errors.getvalue()
            self.assertIn(str(checkpoint.resolve()), implicit_message)
            self.assertIn("checkpoint batch is unknown", implicit_message)
            self.assertIn("conservative", implicit_message)
            self.assertIn("--batch 4", implicit_message)
            self.assertNotIn("batch from 8", implicit_message)

            fresh_errors = StringIO()
            with mock.patch.object(train_yolo_v2, "run_training", side_effect=KeyboardInterrupt), redirect_stdout(StringIO()), redirect_stderr(fresh_errors):
                self.assertEqual(train_yolo_v2.main([]), 130)
            self.assertIn(str(config["project"] / config["name"]), fresh_errors.getvalue())

            resume_errors = StringIO()
            with mock.patch.object(train_yolo_v2, "run_training", side_effect=KeyboardInterrupt), redirect_stdout(StringIO()), redirect_stderr(resume_errors):
                self.assertEqual(train_yolo_v2.main(["--resume", str(checkpoint)]), 130)
            self.assertIn(str(resume_run), resume_errors.getvalue())

    def test_cli_returns_error_code_two_when_ultralytics_is_not_installed(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory)
            config["model"].write_bytes(b"model")
            summaries = {split: {"images": 1, "boxes": 1, "negative_images": 0, "class_counts": {0: 1}}
                         for split in ("train", "val", "test")}
            errors = StringIO()
            with mock.patch.object(train_yolo_v2, "load_config", return_value=config), \
                 mock.patch.object(train_yolo_v2, "validate_dataset", return_value=summaries), \
                 mock.patch.object(train_yolo_v2, "validate_cuda", return_value={"index": 0, "name": "GPU", "vram_gb": 8.0}), \
                 mock.patch.dict(sys.modules, {"ultralytics": None}), \
                 redirect_stdout(StringIO()), redirect_stderr(errors):
                self.assertEqual(train_yolo_v2.main([]), 2)

            self.assertIn("ERROR:", errors.getvalue())
            self.assertIn("Ultralytics", errors.getvalue())
