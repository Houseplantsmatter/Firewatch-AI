# FireWatch Trained Model Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace browser-simulated FireWatch results with real inference from the completed V2 YOLO checkpoint in the existing image-upload interface.

**Architecture:** A lazy, process-scoped Python inference service will normalize Ultralytics results without depending on Flask. Flask will validate multipart images and expose a small JSON endpoint. The browser will call that endpoint while retaining the existing rendering, history, accessibility, and responsive behavior.

**Tech Stack:** Python 3.11, Flask 3.0, Pillow, Ultralytics YOLO, browser JavaScript, Node's built-in test runner, Python `unittest`.

## Global Constraints

- Use `runs/detect/fire_smoke_v2/weights/best.pt`.
- Accept one JPG, PNG, or WebP image no larger than 10 MiB or 40,000,000 decoded pixels.
- Use confidence `0.25`, IoU `0.45`, input size `640`, automatic device selection, and at most `300` detections.
- Do not save uploaded images or Ultralytics prediction artifacts.
- Display the highest-confidence detection and count every accepted fire or smoke detection.
- A no-detection result uses `category: "clear"`, `confidence: null`, `detectionCount: 0`, and `box: null`.
- Preserve the existing UI result renderer, local history, focus behavior, responsive layout, and safety notice.
- Follow red-green-refactor for every production behavior.

---

### Task 1: Normalize Real YOLO Predictions Behind a Lazy Service

**Files:**
- Create: `ui/inference.py`
- Create: `ui/tests/test_inference.py`

**Interfaces:**
- Produces: `ModelUnavailableError`, `InferenceFailure`, `normalize_prediction(names, xyxy, confidences, class_ids, width, height) -> dict`, and `InferenceService(model_path, model_loader=None).predict(image_bytes) -> dict`.
- Consumes: a model loader compatible with `ultralytics.YOLO(path)` and a model whose `predict` returns one Ultralytics-style result.

- [ ] **Step 1: Write failing normalization tests**

Add fake tensor/result helpers and tests that call `normalize_prediction` with:

```python
self.assertEqual(
    normalize_prediction({0: "smoke", 1: "fire"}, [], [], [], 640, 480),
    {"category": "clear", "confidence": None, "detectionCount": 0, "box": None},
)

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
    {"x": 0.15625, "y": 50 / 480, "width": 0.15625, "height": 100 / 480},
)
```

Also assert rejection of unknown class names, mismatched tensor lengths, nonpositive image dimensions, nonfinite values, and zero-area boxes.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m unittest ui/tests/test_inference.py -v
```

Expected: import failure because `ui/inference.py` does not exist.

- [ ] **Step 3: Implement minimal normalization**

Implement a pure function that maps class IDs through either mapping- or
sequence-form names, validates every numeric value with `math.isfinite`, clamps
`xyxy` coordinates to the source dimensions, discards no accepted model
detections silently, chooses the greatest confidence, and returns its normalized
top-left/width/height box plus the total detection count.

The accepted label set must be exactly `{"smoke", "fire"}`. Any malformed model
output raises `InferenceFailure`.

- [ ] **Step 4: Run normalization tests and verify GREEN**

Run:

```powershell
python -m unittest ui/tests/test_inference.py -v
```

Expected: all normalization tests pass.

- [ ] **Step 5: Write failing lazy-load and prediction tests**

Add a counting loader and Ultralytics-style fake model:

```python
service = InferenceService(Path("model.pt"), model_loader=counting_loader)
first = service.predict(valid_jpeg_bytes())
second = service.predict(valid_jpeg_bytes())
self.assertEqual(load_calls, 1)
self.assertEqual(first, second)
self.assertFalse(fake_model.predict_calls[0]["save"])
self.assertEqual(fake_model.predict_calls[0]["conf"], 0.25)
self.assertEqual(fake_model.predict_calls[0]["iou"], 0.45)
self.assertEqual(fake_model.predict_calls[0]["imgsz"], 640)
self.assertEqual(fake_model.predict_calls[0]["max_det"], 300)
```

Assert that a missing checkpoint raises `ModelUnavailableError`, incompatible
class names raise `ModelUnavailableError`, and third-party load/predict errors
are wrapped without exposing their original text.

- [ ] **Step 6: Run service tests and verify RED**

Run the same unittest command.

Expected: failures because `InferenceService` behavior is not implemented.

- [ ] **Step 7: Implement the lazy inference service**

Use a `threading.Lock` and double-checked `_model` initialization. Resolve the
checkpoint once, check `is_file()`, import `ultralytics.YOLO` only inside the
default loader, validate `model.names`, decode request bytes to an RGB Pillow
image, and call:

```python
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
```

Require exactly one result, detach its box tensors to CPU lists, and pass them
to `normalize_prediction`.

- [ ] **Step 8: Run service tests and verify GREEN**

Run:

```powershell
python -m unittest ui/tests/test_inference.py -v
```

Expected: all inference tests pass with no real model initialization.

### Task 2: Expose a Validated Flask Inference Endpoint

**Files:**
- Modify: `ui/app.py`
- Modify: `ui/tests/test_app.py`

**Interfaces:**
- Consumes: `InferenceService.predict(image_bytes) -> dict`.
- Produces: `create_app(test_config=None, inference_service=None)` and `POST /api/analyze`.

- [ ] **Step 1: Write failing success-contract tests**

Insert `UI_DIR` into `sys.path`, create a fake service that records request
bytes, and construct the test app with:

```python
self.service = FakeInferenceService(
    {"category": "fire", "confidence": 0.91, "detectionCount": 2,
     "box": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4}}
)
self.app = APP_MODULE.create_app(
    {"TESTING": True}, inference_service=self.service
)
```

POST a generated Pillow JPEG as field `image`; assert HTTP 200, exact prediction
fields, original filename, an ISO UTC millisecond timestamp, an ID beginning
with that timestamp, and that the fake received the original bytes.

- [ ] **Step 2: Run endpoint success test and verify RED**

Run:

```powershell
python -m unittest ui.tests.test_app.AppTests.test_analyze_returns_normalized_model_result -v
```

Expected: failure because the factory does not accept the injected service and
the route does not exist.

- [ ] **Step 3: Implement the endpoint success path**

Update the factory signature, configure the default checkpoint relative to the
repository root, and create a default `InferenceService` only when no injected
service is supplied.

Use `werkzeug.utils.secure_filename`, a UTC millisecond timestamp, and
`secrets.token_hex(8)`. Return:

```python
return jsonify(result={
    "id": f"{timestamp}-{token_hex(8)}",
    "filename": safe_filename,
    "timestamp": timestamp,
    **prediction,
})
```

- [ ] **Step 4: Run endpoint success test and verify GREEN**

Run the single test again and expect PASS.

- [ ] **Step 5: Write failing upload and failure-response tests**

Add tests for:

- missing field and blank filename: HTTP 400;
- zero-byte body: HTTP 400;
- payload above 10 MiB: HTTP 413;
- unsupported declared MIME type: HTTP 400;
- malformed bytes: HTTP 400;
- valid JPEG declared as PNG: HTTP 400;
- an image whose width times height exceeds 40,000,000: HTTP 400 using a patched
  Pillow image object rather than allocating a giant bitmap;
- `ModelUnavailableError`: HTTP 503;
- `InferenceFailure` and an unexpected exception: HTTP 500 with the same generic
  retry message and without the exception text.

- [ ] **Step 6: Run failure tests and verify RED**

Run:

```powershell
python -m unittest ui/tests/test_app.py -v
```

Expected: the new validation/failure tests fail while prior UI-shell tests still
pass.

- [ ] **Step 7: Implement bounded in-memory image validation and safe errors**

Read at most `10 * 1024 * 1024 + 1` bytes. Check MIME type against:

```python
{"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}
```

Open the bytes with Pillow, check the detected format, positive dimensions, and
the 40,000,000-pixel cap, then call `verify()`. Map known validation errors to
safe 400/413 messages, `ModelUnavailableError` to 503, and inference/unexpected
errors to a logged generic 500 response.

- [ ] **Step 8: Run the complete Python UI suite and verify GREEN**

Run:

```powershell
python -m unittest discover -s ui/tests -p "test_*.py" -v
```

Expected: all inference, endpoint, and existing UI contract tests pass.

### Task 3: Replace Browser Simulation with Multipart Fetch

**Files:**
- Modify: `ui/static/js/core.js`
- Modify: `ui/static/js/app.js`
- Modify: `ui/tests/core.test.js`
- Modify: `ui/tests/test_app.py`

**Interfaces:**
- Produces: `validateServerResult(value) -> boolean` in `FireWatchCore`.
- Consumes: `POST /api/analyze` returning `{result: ServerResult}`.

- [ ] **Step 1: Replace simulation tests with failing server-contract tests**

Remove tests for `simulateResult`. Add:

```javascript
assert.equal(core.validateServerResult({
  id: "2026-07-23T20:00:00.000Z-a1",
  filename: "scene.jpg",
  timestamp: "2026-07-23T20:00:00.000Z",
  category: "fire",
  confidence: 0.91,
  detectionCount: 2,
  box: {x: 0.1, y: 0.2, width: 0.3, height: 0.4},
}), true);

assert.equal(core.validateServerResult({
  id: "2026-07-23T20:00:00.000Z-a2",
  filename: "scene.jpg",
  timestamp: "2026-07-23T20:00:00.000Z",
  category: "clear",
  confidence: null,
  detectionCount: 0,
  box: null,
}), true);
```

Assert false for a thumbnail supplied by the server, missing/extra keys,
category-dependent mismatches, invalid timestamps, unbounded strings,
nonfinite confidence/counts, and out-of-bounds boxes.

- [ ] **Step 2: Run Node tests and verify RED**

Run:

```powershell
node --test ui/tests/core.test.js
```

Expected: failures because `validateServerResult` is missing.

- [ ] **Step 3: Implement strict server-result validation**

Refactor the existing history validation so shared fields and
category-dependent rules are reused. `validateServerResult` must accept exactly
`id`, `filename`, `timestamp`, `category`, `confidence`, `detectionCount`, and
`box`; `isValidEntry` must require the same fields plus `thumbnail`.

Remove `hashMetadata` and `simulateResult` from the module and exports.

- [ ] **Step 4: Run Node tests and verify GREEN**

Run the Node test command and expect PASS.

- [ ] **Step 5: Add failing controller contract tests**

Replace the README prototype expectation and add source-level controller
contracts asserting `app.js`:

- creates `new FormData()` and appends `"image"`;
- calls `fetch("/api/analyze", {method: "POST", body: formData})`;
- calls `core.validateServerResult`;
- appends `thumbnail: state.thumbnail` only after validation;
- contains no `simulateResult` or fixed 900 ms delay;
- catches failures, restores `uploadPanel` and `previewPanel`, hides
  `progressPanel`, re-enables `analyzeButton`, and announces the safe error;
- changes the clear-result copy from “simulated” to “at or above the model's
  detection threshold.”

- [ ] **Step 6: Run controller tests and verify RED**

Run:

```powershell
python -m unittest ui/tests/test_app.py -v
```

Expected: the new browser integration contracts fail.

- [ ] **Step 7: Implement real request and retryable error flow**

Add a response helper that attempts JSON parsing, prefers a non-empty server
`error`, falls back to:

```text
The image could not be analyzed. Please try again.
```

In `analyzeCurrentFile`, clear stale errors, submit the selected file, reject
non-OK or invalid response shapes, merge `{...payload.result, thumbnail:
state.thumbnail}`, save and render it, and retain the existing completion
announcement/focus behavior. In `catch`, restore the preview without clearing
the selected file or object URL and focus the Analyze button.

- [ ] **Step 8: Run browser and UI tests and verify GREEN**

Run:

```powershell
node --test ui/tests/core.test.js
python -m unittest ui/tests/test_app.py -v
node --check ui/static/js/core.js
node --check ui/static/js/app.js
```

Expected: every command passes.

### Task 4: Document Runtime Dependencies and Verify the Real Checkpoint

**Files:**
- Modify: `ui/requirements.txt`
- Modify: `ui/README.md`
- Modify: `ui/tests/test_app.py`
- No production artifact created by the smoke test.

**Interfaces:**
- Consumes: the completed checkpoint and `InferenceService.predict`.
- Produces: reproducible local setup/run/test instructions.

- [ ] **Step 1: Write failing documentation/dependency contract tests**

Change the README test to require the checkpoint path, real YOLO inference,
first-request lazy loading, automatic GPU/CPU selection, supported file limits,
and the Python/Node test commands. Require `Pillow` and `ultralytics` in
`ui/requirements.txt`, and reject “UI prototype”, “does not run YOLO”, and
`simulateResult`.

- [ ] **Step 2: Run the documentation test and verify RED**

Run:

```powershell
python -m unittest ui.tests.test_app.AppTests.test_readme_documents_real_model_setup_and_limits -v
```

Expected: failure against the prototype README.

- [ ] **Step 3: Update dependencies and README**

Declare:

```text
Flask>=3.0,<3.1
Pillow>=10.0
ultralytics>=8.3
```

Document `python -m pip install -r requirements.txt`, `python app.py`,
`http://127.0.0.1:5000`, the exact checkpoint, supported file/pixel limits,
lazy first-request loading, device behavior, the safety limitation, and all
verification commands.

- [ ] **Step 4: Run fast suites and syntax checks**

Run:

```powershell
python -m unittest discover -s ui/tests -p "test_*.py" -v
node --test ui/tests/core.test.js
node --check ui/static/js/core.js
node --check ui/static/js/app.js
python -m compileall -q ui
```

Expected: all tests and checks pass without warnings or tracebacks.

- [ ] **Step 5: Run a real-checkpoint inference smoke test**

Using `yolo311_env\Scripts\python.exe`, load
`runs/detect/fire_smoke_v2/weights/best.pt` through `InferenceService` and
analyze `AoF00001.jpg` from memory. Assert only stable contract properties:

```python
assert result["category"] in {"fire", "smoke", "clear"}
assert isinstance(result["detectionCount"], int)
assert result["detectionCount"] >= 0
assert (result["category"] == "clear") == (result["box"] is None)
```

Expected: process exits 0, uses the real checkpoint, and creates no new
`runs/detect` directory.

- [ ] **Step 6: Confirm no inference artifacts were written**

Capture the `runs/detect` directory names before and after the smoke test and
assert they are identical.

Expected: no new run directory or uploaded-image file exists.

- [ ] **Step 7: Perform final spec coverage review**

Check the implementation against every acceptance criterion in
`docs/superpowers/specs/2026-07-23-firewatch-model-integration-design.md`.
Re-run any targeted test affected by final cleanup, then repeat the complete
verification commands from Step 4.
