# Fire and Smoke Training V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new, validated YOLO11s training entry point tuned for the local GTX 1660 and automatically evaluate its best checkpoint on the test split.

**Architecture:** A versioned YAML file owns all repeatable training parameters. A single import-safe Python entry point validates configuration, dataset labels, and CUDA before lazily loading Ultralytics, then trains and evaluates; pure validation and argument-building functions keep expensive GPU work outside unit tests.

**Tech Stack:** Python 3.8, PyYAML, PyTorch, Ultralytics YOLO11, standard-library `unittest`

## Global Constraints

- Keep `train_yolo.py`, `detect.py`, existing weights, and existing run directories unchanged.
- Start fresh from `yolo11s.pt` unless `--resume CHECKPOINT` is explicitly supplied.
- Default to 640-pixel images, 80 epochs, batch size 8, CUDA device 0, AMP, and early stopping patience 15.
- Write the production run to `runs/detect/fire_smoke_v2` without overwriting it.
- Validate all train, validation, and test labels before starting training.
- Evaluate the best checkpoint on the test split after a successful fresh run.
- Remain compatible with the project's Python 3.8 environment.
- Git checkpoint steps cannot be performed because this directory is not a Git repository.

---

### Task 1: Versioned configuration and validation

**Files:**
- Create: `train_config_v2.yaml`
- Create: `data_v2.yaml`
- Create: `train_yolo_v2.py`
- Create: `tests/test_train_yolo_v2.py`

**Interfaces:**
- Produces: `load_config(path: Path) -> Dict[str, Any]`
- Produces: `validate_dataset(data_path: Path) -> Dict[str, Dict[str, Any]]`
- Produces: `ConfigError` and `DatasetError`

- [ ] **Step 1: Write failing configuration and label-validation tests**

Create temporary YAML datasets in `tests/test_train_yolo_v2.py`. Assert that
`load_config` resolves `model`, `data`, and `project` relative to the config
directory; assert `validate_dataset` accepts a valid row and an empty negative
label; and use subtests to reject malformed rows, class ID 2, nonnumeric values,
coordinates outside 0 through 1, and zero-width boxes.

```python
class DatasetValidationTests(unittest.TestCase):
    def test_accepts_positive_and_negative_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_file = make_dataset(root, ["0 0.5 0.5 0.2 0.3", ""])
            summary = training.validate_dataset(data_file)
            self.assertEqual(summary["train"]["images"], 2)
            self.assertEqual(summary["train"]["boxes"], 1)
            self.assertEqual(summary["train"]["negative_images"], 1)

    def test_rejects_invalid_rows(self):
        invalid_rows = (
            "2 0.5 0.5 0.2 0.3",
            "0 nope 0.5 0.2 0.3",
            "0 1.1 0.5 0.2 0.3",
            "0 0.5 0.5 0 0.3",
            "0 0.5 0.5 0.2",
        )
        for row in invalid_rows:
            with self.subTest(row=row):
                with tempfile.TemporaryDirectory() as directory:
                    data_file = make_dataset(Path(directory), [row])
                    with self.assertRaises(training.DatasetError):
                        training.validate_dataset(data_file)
```

- [ ] **Step 2: Run tests and confirm they fail before implementation**

Run: `.\yolo_env\Scripts\python.exe -m unittest discover -s tests -v`

Expected: import or attribute failures because `train_yolo_v2.py` and its
functions do not exist.

- [ ] **Step 3: Add the V2 YAML configuration**

Create `data_v2.yaml` beside the training script without a `path` key, using
`data/train/images`, `data/val/images`, and `data/test/images` plus the
contiguous `smoke`, `fire` class mapping. Create `train_config_v2.yaml` with
exact defaults:

```yaml
version: 2
model: yolo11s.pt
data: data_v2.yaml
project: runs/detect
name: fire_smoke_v2
train:
  epochs: 80
  imgsz: 640
  batch: 8
  device: 0
  workers: 2
  patience: 15
  optimizer: AdamW
  lr0: 0.001
  lrf: 0.01
  cos_lr: true
  pretrained: true
  amp: true
  cache: false
  seed: 42
  deterministic: true
  save_period: 5
  val: true
  plots: true
  degrees: 5.0
  translate: 0.1
  scale: 0.4
  shear: 2.0
  perspective: 0.0
  flipud: 0.0
  fliplr: 0.5
  hsv_h: 0.015
  hsv_s: 0.6
  hsv_v: 0.4
  mosaic: 0.8
  mixup: 0.05
  close_mosaic: 10
evaluation:
  split: test
  imgsz: 640
  batch: 8
  device: 0
  half: true
  plots: true
```

- [ ] **Step 4: Implement strict configuration and dataset validation**

In `train_yolo_v2.py`, use `yaml.safe_load`, explicit top-level and nested key
allowlists, required-key checks, numeric range checks, and Python 3.8-compatible
typing. Resolve paths against the configuration directory. When the dataset YAML
omits `path`, use its absolute parent as the root; otherwise resolve `path`
relative to the YAML parent. Map each `images` directory to its sibling `labels`
directory, and validate every `.txt` row as five YOLO fields. Return per-split
image, box, negative-image, and per-class counts. Reject noncontiguous mapping
class IDs and orphan labels.

```python
def validate_label(path, class_ids):
    boxes = 0
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise DatasetError("{}:{} must contain 5 fields".format(path, line_number))
        try:
            class_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError:
            raise DatasetError("{}:{} contains a nonnumeric value".format(path, line_number))
        if class_id not in class_ids:
            raise DatasetError("{}:{} uses unknown class {}".format(path, line_number, class_id))
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in coordinates):
            raise DatasetError("{}:{} coordinates must be finite and normalized".format(path, line_number))
        if coordinates[2] <= 0.0 or coordinates[3] <= 0.0:
            raise DatasetError("{}:{} width and height must be positive".format(path, line_number))
        boxes += 1
    return boxes
```

- [ ] **Step 5: Run validation tests**

Run: `.\yolo_env\Scripts\python.exe -m unittest discover -s tests -v`

Expected: configuration and dataset-validation tests pass.

---

### Task 2: CUDA preflight, training orchestration, and CLI

**Files:**
- Modify: `train_yolo_v2.py`
- Modify: `tests/test_train_yolo_v2.py`

**Interfaces:**
- Consumes: `load_config` and `validate_dataset` from Task 1
- Produces: `validate_cuda(device: Any) -> Dict[str, Any]`
- Produces: `build_train_args(config: Dict[str, Any]) -> Dict[str, Any]`
- Produces: `run_training(config: Dict[str, Any], resume: Optional[Path], resume_batch: Optional[int] = None) -> Path`
- Produces: `main(argv: Optional[List[str]]) -> int`

- [ ] **Step 1: Write failing orchestration tests**

Mock `torch.cuda` and the lazy YOLO factory. Assert unavailable CUDA fails,
configuration plus CLI overrides form the expected Ultralytics arguments,
check-only never creates a YOLO instance, fresh training loads `yolo11s.pt`,
and evaluation loads the trainer's `best.pt` with `split="test"`.

```python
def test_build_train_args_preserves_v2_defaults(self):
    config = training.load_config(self.config_path)
    args = training.build_train_args(config)
    self.assertEqual(args["epochs"], 80)
    self.assertEqual(args["batch"], 8)
    self.assertEqual(args["imgsz"], 640)
    self.assertEqual(args["device"], 0)
    self.assertTrue(args["amp"])
    self.assertEqual(args["name"], "fire_smoke_v2")
```

- [ ] **Step 2: Run targeted tests and confirm failure**

Run: `.\yolo_env\Scripts\python.exe -m unittest tests.test_train_yolo_v2.TrainingOrchestrationTests -v`

Expected: failures for the not-yet-defined CUDA and orchestration functions.

- [ ] **Step 3: Implement CUDA preflight and training argument construction**

Normalize numeric device strings to indexes, reject CPU/non-CUDA targets, check
`torch.cuda.is_available()` and device bounds, then return name and VRAM. Build a
new dictionary containing `data`, `project`, `name`, `exist_ok=False`, and all
validated `train` values without mutating the loaded configuration.

```python
def validate_cuda(device):
    try:
        index = int(device)
    except (TypeError, ValueError):
        raise RuntimeEnvironmentError("V2 training requires a numeric CUDA device")
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        raise RuntimeEnvironmentError("CUDA device {} is unavailable".format(index))
    properties = torch.cuda.get_device_properties(index)
    return {
        "index": index,
        "name": torch.cuda.get_device_name(index),
        "vram_gb": properties.total_memory / float(1024 ** 3),
    }
```

- [ ] **Step 4: Implement CLI, fresh/resume training, and best-model evaluation**

Parse `--config`, `--check-only`, `--epochs`, `--batch`, `--device`, `--name`, and
`--resume`. Apply an explicit batch consistently to training and final evaluation.
Run configuration, dataset, and CUDA preflight before importing `YOLO`. For a
fresh run call `YOLO(model).train(**train_args)`. For resume, require the
checkpoint and call `YOLO(checkpoint).train(resume=True, device=device)`, adding
`batch` only when the CLI supplied it so checkpoint batch semantics otherwise
remain intact.
After a fresh run, load `Path(model.trainer.best)`, ensure it exists, and call
`YOLO(best).val(data=data, split="test", ...)`. Print precision, recall, mAP50,
mAP50-95, and the checkpoint path.

Require `evaluation.split` to be exactly `test`, normalize supported optimizer
names case-insensitively, and reject any existing exact fresh-run training or
test-evaluation directory before loading YOLO while allowing explicit resume.
Catch custom validation errors at the CLI boundary with exit code 2. Fresh-run
CUDA OOM guidance must use the planned `weights/last.pt` resume command with a
smaller explicit batch and mention a distinct `--name` fallback if no checkpoint
exists. Resume OOM guidance must use the exact checkpoint and explicit batch;
when no explicit batch is known, label the proposed batch as conservative rather
than claiming it came from the checkpoint. Return exit code 3. Keyboard
interrupt uses exit code 130 plus the planned fresh-run or checkpoint-owning
resume directory.

- [ ] **Step 5: Run all unit tests**

Run: `.\yolo_env\Scripts\python.exe -m unittest discover -s tests -v`

Expected: all tests pass and no training run directory is created.

---

### Task 3: Real-data preflight and launch verification

**Files:**
- Modify only if verification identifies a defect: `train_yolo_v2.py`
- Modify only if behavior changes: `tests/test_train_yolo_v2.py`

**Interfaces:**
- Consumes: complete V2 CLI from Task 2
- Produces: checked, launch-ready `train_yolo_v2.py` and `train_config_v2.yaml`

- [ ] **Step 1: Compile the entry point**

Run: `.\yolo_env\Scripts\python.exe -m py_compile train_yolo_v2.py`

Expected: exit code 0 with no output.

- [ ] **Step 2: Run the complete current-dataset preflight**

Run: `.\yolo_env\Scripts\python.exe train_yolo_v2.py --check-only`

Expected: reports GTX 1660 CUDA details and summaries for all 14,122 train,
3,099 validation, and 4,306 test images, then exits without loading YOLO or
creating a run.

- [ ] **Step 3: Re-run the complete unit suite after real-data validation**

Run: `.\yolo_env\Scripts\python.exe -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 4: Confirm protected files and runs remain unchanged**

Run: `Get-Item train_yolo.py,detect.py,runs\detect\fire_smoke_yolo11s\weights\best.pt | Select-Object FullName,Length,LastWriteTime`

Expected: the legacy files exist and no V2 check-only run directory was created.

- [ ] **Step 5: Hand off the production launch command**

Run when ready for the long training job:

```powershell
.\yolo_env\Scripts\python.exe train_yolo_v2.py
```

If CUDA reports out of memory, follow the printed collision-safe resume command
with its explicit smaller `--batch`; when no checkpoint exists, select a distinct
`--name` as instructed.
