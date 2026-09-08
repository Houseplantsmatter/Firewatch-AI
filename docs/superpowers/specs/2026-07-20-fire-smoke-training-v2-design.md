# Fire and Smoke Training V2 Design

## Goal

Create a new, independent training pipeline for the existing two-class fire and
smoke dataset. The pipeline must favor detection accuracy while remaining safe
to train on the available NVIDIA GeForce GTX 1660 with 6 GB of VRAM. Existing
training and detection files remain unchanged.

The trained model is probabilistic; the pipeline will measure accuracy but will
not claim that the resulting model is 100% correct.

## Deliverables

- `train_yolo_v2.py`: command-line training, dataset validation, and evaluation.
- `data_v2.yaml`: portable V2 dataset descriptor whose relative split paths are
  anchored to the descriptor's directory without changing legacy `data.yaml`.
- `train_config_v2.yaml`: versioned model, training, augmentation, and output
  settings.
- Automated tests for configuration loading and dataset-label validation.

Training artifacts will be written to `runs/detect/fire_smoke_v2`. The existing
`runs/detect/fire_smoke_yolo11s` run will not be resumed or overwritten.

## Model and Training Strategy

V2 starts a fresh transfer-learning run from the pretrained `yolo11s.pt`
checkpoint. YOLO11s is the chosen balance: it provides more capacity than the
nano model while the current machine has already demonstrated that it can train
the small model at 640-pixel resolution with a batch size of eight.

The default settings are:

- 640 by 640 input images.
- 80 maximum epochs with early stopping after 15 epochs without improvement.
- Batch size 8, mixed-precision training, and CUDA device 0.
- AdamW optimizer with cosine learning-rate decay.
- Fixed random seed and deterministic training where supported.
- Two Windows-safe data-loader workers and no in-memory image cache.
- Validation every epoch, periodic checkpoints, metric plots, and best/last
  checkpoints.

Augmentation will be controlled rather than extreme: small rotations and
translations, horizontal flips, moderate scale and HSV changes, mosaic during
the early epochs, and a small amount of mixup. Mosaic will be disabled for the
last ten epochs so final optimization sees natural images.

All defaults live in `train_config_v2.yaml`. Command-line overrides will support
the practical recovery and smoke-test cases without requiring source edits:
configuration path, device, epoch count, batch size, run name, resume checkpoint,
and a check-only mode. An explicit batch override applies consistently to both
training and final evaluation.

## Components and Data Flow

1. The CLI loads `train_config_v2.yaml` and rejects missing, unknown, or
   incorrectly typed required settings. It requires evaluation on exactly the
   `test` split and accepts only installed Ultralytics optimizer names,
   normalizing valid names to their canonical spelling case-insensitively.
2. Paths are resolved relative to the project/configuration location, avoiding
   dependence on the caller's current directory.
3. A preflight validator reads `data_v2.yaml`. Because the descriptor deliberately
   omits `path`, its absolute parent is the dataset root regardless of process
   working directory; descriptors that provide `path` still resolve it relative
   to their own parent. Root-containment checks remain enforced.
4. Valid label rows must contain a class ID plus four normalized coordinates;
   mapping-form class IDs must be contiguous from zero, coordinates must be
   finite and in range, and box width and height must be positive. Empty files
   remain valid negative examples. Missing labels, orphan labels, and unreadable
   configuration fail before expensive training begins.
5. The script confirms CUDA availability and reports the detected GPU and VRAM.
   It fails with an actionable message if the requested CUDA device cannot be
   used instead of silently beginning a very slow CPU run.
6. Ultralytics trains a new YOLO11s run with the validated configuration.
7. After training, the best checkpoint is evaluated once against the untouched
   test split. Precision, recall, mAP50, and mAP50-95 are printed along with the
   checkpoint location.

## Failure and Recovery Behavior

Configuration, dataset, label, checkpoint, CUDA, and out-of-memory errors must
produce concise messages and a nonzero exit code. Before a fresh run, either
existing exact output directory (`project/name` or `project/name_test`) fails
with guidance to choose `--name` or explicitly resume. A fresh-run
out-of-memory message points to the planned `weights/last.pt` resume command with
a smaller explicit batch, plus a distinct-name fallback if no checkpoint exists.
A resume OOM message always includes the exact checkpoint; it halves an
explicitly supplied batch, but labels the fallback batch as a conservative
suggestion when the checkpoint's current batch is unknown. Keyboard interrupts
report the planned fresh-run directory or the resume checkpoint's owning run
directory.

A supplied resume checkpoint starts a resume run explicitly. Without `--batch`,
Ultralytics preserves the checkpoint's batch semantics; only an explicit CLI
batch is forwarded to resumed training. With no resume argument, V2 always
begins from `yolo11s.pt`, so an old partial run cannot be continued accidentally.

## Verification

Fast automated tests will cover:

- Valid configuration parsing and relative path resolution.
- Rejection of missing required sections and invalid values.
- Acceptance of valid labels and empty negative-example labels.
- Rejection of unknown class IDs, malformed rows, nonnumeric values,
  out-of-range coordinates, nonpositive boxes, and missing label files.
- Construction of the expected Ultralytics training arguments without launching
  a long training job.

The implementation will also expose `--check-only`, which must complete a full
dataset preflight without initializing training. A one-epoch smoke run may be
used to verify GPU integration, but the full 80-epoch training run is a separate,
long-running user operation.

## Acceptance Criteria

- Existing scripts, weights, and run directories are untouched.
- `--check-only` succeeds on the current dataset and reports split/image/box
  counts plus the GTX 1660 device details.
- Unit tests pass without training a model.
- A one-epoch override can initialize YOLO11s on CUDA, train, validate, and save
  into a distinct test run without overwriting the production V2 run.
- A default invocation starts a fresh 80-epoch V2 run and evaluates `best.pt` on
  the test split after training.
