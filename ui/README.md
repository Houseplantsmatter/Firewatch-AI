# FireWatch AI

FireWatch is a responsive local web interface that performs real YOLO inference
on uploaded images. It uses the completed V2 fire-and-smoke checkpoint at
`runs/detect/fire_smoke_v2/weights/best.pt`.

## Setup

From the project root, activate the Python environment that has access to your
GPU-enabled PyTorch installation, then install the UI dependencies:

```powershell
cd ui
python -m pip install -r requirements.txt
```

## Run

```powershell
python app.py
```

Open `http://127.0.0.1:5000`.

The model is loaded lazily on the first analysis request, so that request takes
longer than later analyses. Ultralytics automatically uses an available CUDA
GPU and otherwise falls back to CPU; CPU inference can be noticeably slower.

## Quick phone test on the same Wi-Fi

Keep the computer and terminal running, connect the phone to the same trusted
Wi-Fi network, and run this command from `ui`:

```powershell
python phone_test.py
```

Open the printed `http://<IPv4-address>:5000` URL on the phone. If Windows
Firewall prompts you, allow Python on **private networks only**. If the page
does not load, confirm both devices are on the same non-guest Wi-Fi; guest
networks often prevent devices from reaching each other.

Press `Ctrl+C` in the terminal when the test is finished. This launcher uses
Flask's development server and is not intended for public internet access.

## Analyze an image

Choose a JPG, PNG, or WebP file, then select **Analyze image**. Files may be at
most 10 MB and 40,000,000 pixels. The Flask server validates and analyzes the
image in memory. It does not save the upload or create YOLO prediction runs.

The interface displays the highest-confidence fire or smoke box at the
configured 0.25 threshold and reports the total number of detections. Results
with no detection at or above that threshold are shown as **No concern found**.
Recent results and their small browser-generated thumbnails are stored only in
this browser's local storage.

## Test

Run these commands from `ui`:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
node --test tests/core.test.js
node --check static/js/core.js
node --check static/js/app.js
python -m compileall -q .
```

The complete Python suite uses an injected inference service and does not load
the checkpoint. To run a one-image smoke test against the real checkpoint from
`ui`, use:

```powershell
python -c "from pathlib import Path; from inference import InferenceService; root=Path('..').resolve(); print(InferenceService(root / 'runs/detect/fire_smoke_v2/weights/best.pt').predict((root / 'AoF00001.jpg').read_bytes()))"
```

This prints a normalized real-model result and must not create a new prediction
directory under `runs/detect`.

## Limitations and safety

Model predictions are probabilistic. A detection may be incorrect, and **No
concern found** does not establish that a scene is safe. FireWatch does not
replace working smoke alarms, official emergency alerts, trained firefighters,
or calling 911 during an emergency.
