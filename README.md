# FireWatch AI

Fire and smoke detection with Ultralytics YOLO and a Flask web interface.
Upload an image to see a detection category, confidence, bounding box, and
detection count. The repository also includes dataset preparation, training,
evaluation, and command-line inference tools.

The current application runs on a computer, with an optional phone launcher
for devices on the same Wi-Fi. Uploading this repository does not deploy a
working GitHub Pages application.

## Setup

Use Python 3.10 or newer; Python 3.11 is recommended. Run the following in
PowerShell:

```powershell
git clone https://github.com/Houseplantsmatter/Firewatch-AI.git
cd Firewatch-AI
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python check_environment.py
```

On macOS/Linux, create the environment with `python3 -m venv .venv` and
activate it with `source .venv/bin/activate`, then use the same pip commands.
For CUDA, install matching GPU-enabled `torch` and `torchvision` builds for
your system before installing the requirements. The V2 training script requires
a CUDA GPU; web inference can also run on CPU. Node.js is only needed to run
the JavaScript tests; there is no frontend build step.

## Supply the trained model

**Model weights and datasets are not included in Git.** Obtain a compatible
trained fire/smoke checkpoint separately, or train one using the workflow
below. Place the trained checkpoint at:

```text
runs/detect/fire_smoke_v2/weights/best.pt
```

The model must use class `0` for `smoke` and class `1` for `fire`.
A stock `yolo11s.pt` is a training starting point, not the trained fire/smoke
model. The page can open without `best.pt`, but analysis returns a model
unavailable error until the checkpoint is supplied.

## Run the web interface

From the repository root:

```powershell
python ui/app.py
```

Open http://127.0.0.1:5000 and upload a JPG, PNG, or WebP image, up to 10 MB
and 40 million pixels. The first analysis loads the model and can take longer.
Uploads are analyzed in memory; recent results and thumbnails are stored in
the browser's local storage.

To test on a phone connected to the same trusted Wi-Fi:

```powershell
python ui/phone_test.py
```

Open the printed address on the phone and keep the computer running. This
launcher uses Flask's development server for local testing. See
[the UI guide](ui/README.md) for details.

## Prepare data and train

Provide your own images and matching YOLO label files under `dataset/images/`
and `dataset/labels/`. Each label row contains
`class_id x_center y_center width height`, with normalized coordinates.
Use the same smoke/fire class order as above.

```powershell
python prepare_dataset.py --images dataset/images --labels dataset/labels --output data --data-yaml data.yaml
```

This creates train, validation, and test splits under `data/` with a default
70/20/10 split. The included `data_v2.yaml` points to those same split folders.
Put the pretrained `yolo11s.pt` starting checkpoint in the repository root;
the V2 trainer requires that file to exist before training.

```powershell
python train_yolo_v2.py --check-only
python train_yolo_v2.py
```

Review `train_config_v2.yaml` for the training settings. The default run uses
CUDA device 0, batch size 8, and 80 epochs, and evaluates on the test split
after training. Outputs go to `runs/detect/fire_smoke_v2/`.

To resume an interrupted run:

```powershell
python train_yolo_v2.py --resume runs/detect/fire_smoke_v2/weights/last.pt
```

The older `train_yolo.py` supports CPU/MPS and additional overrides. Its
existing `train_config.yaml` contains a malformed `translate` value; supply
`--set translate=0.10` when using that configuration.

## Command-line detection

Supply your own input file and explicitly select the trained checkpoint:

```powershell
python detect.py path/to/image.jpg --model runs/detect/fire_smoke_v2/weights/best.pt
```

Run `python detect.py --help` for source, threshold, output, and reporting
options. The script's default source is a local test image that is excluded
from this repository.

## Tests

Run from the repository root after installing the requirements:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
python -m unittest discover -s ui/tests -p "test_*.py" -v
node --test ui/tests/core.test.js
node --check ui/static/js/core.js
node --check ui/static/js/app.js
```

The unit tests use temporary fixtures and injected model services; they do
not require the training dataset or trained weights.

## Repository contents

| Included | Purpose |
| --- | --- |
| `ui/` | Flask app, inference service, browser UI, assets, and tests |
| `prepare_dataset.py` | Validate labels and prepare dataset splits |
| `train_yolo.py`, `train_yolo_v2.py` | Training and evaluation tools |
| `detect.py`, `check_environment.py` | Detection CLI and environment diagnostics |
| `*.yaml`, `requirements.txt`, `tests/` | Configuration, dependencies, and training tests |

## Limitations

FireWatch is still a prototype. It can miss fire or smoke and flag images
that show neither, so don't rely on it to tell you whether a situation is safe.
Keep using smoke alarms, follow official alerts, and contact emergency services
if you suspect a fire.
