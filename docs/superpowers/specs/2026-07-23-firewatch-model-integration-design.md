# FireWatch Trained Model Integration Design

## Goal

Replace the FireWatch UI prototype's deterministic browser simulation with
real, server-side inference from the completed V2 checkpoint at
`runs/detect/fire_smoke_v2/weights/best.pt`. The existing single-image upload,
result, history, accessibility, responsive-layout, and safety-notice behavior
must remain intact.

Live camera and video inference are outside this change.

## Architecture

Flask will expose `POST /api/analyze`. The endpoint will validate the uploaded
image and pass its bytes to a focused inference service. The service will
lazily load the Ultralytics YOLO checkpoint once per application process, run
prediction in memory, and normalize its output to the result contract already
used by the browser.

The browser will submit the selected image as multipart form data and render the
JSON response with the existing result and history code. Uploaded images will
not be written to disk, and prediction will disable Ultralytics run-artifact
output.

The application factory will support injecting a fake inference service for
fast endpoint tests without loading PyTorch or the checkpoint.

## Model and Prediction Settings

- Checkpoint: `runs/detect/fire_smoke_v2/weights/best.pt`.
- Classes: `smoke` and `fire`, matching `data_v2.yaml`.
- Confidence threshold: `0.25`.
- IoU threshold: `0.45`.
- Input size: `640`.
- Device selection: Ultralytics automatic selection, allowing CUDA when
  available and CPU fallback when it is not.
- Maximum detections: `300`.
- Saved images, labels, reports, and run directories: disabled.

The service will validate that the checkpoint exists and that model class names
map to only `smoke` and `fire`. Missing or incompatible model artifacts will
produce an actionable service-unavailable response rather than a false result.

## Result Normalization

The inference service will return a model-level prediction containing all
accepted detections. Each detection includes its class name, confidence, and a
normalized bounding box:

```json
{
  "category": "fire",
  "confidence": 0.91,
  "detectionCount": 2,
  "box": {
    "x": 0.12,
    "y": 0.18,
    "width": 0.41,
    "height": 0.53
  }
}
```

The displayed category, confidence, and box come from the single
highest-confidence accepted detection. `detectionCount` is the total number of
accepted fire and smoke detections in the image. If there are no accepted
detections, the normalized output is:

```json
{
  "category": "clear",
  "confidence": null,
  "detectionCount": 0,
  "box": null
}
```

`clear` means the model found no fire or smoke detection at or above the
configured threshold; it is not a safety guarantee.

The Flask endpoint will add the original sanitized filename, a UTC timestamp,
and a bounded unique ID. The browser will add its locally generated JPEG
thumbnail before saving the result to local history. The server will never
accept or return the thumbnail.

## Request and Response Contract

`POST /api/analyze` accepts one multipart field named `image`.

Server-side validation will enforce:

- A file is present and has a non-empty filename.
- The payload is non-empty and no larger than 10 MiB.
- The declared media type is JPEG, PNG, or WebP.
- Pillow can decode and verify the payload as an image of the declared type.
- The decoded dimensions are positive and contain no more than 40,000,000
  total pixels to avoid decompression-bomb resource exhaustion.

Successful responses use HTTP 200 and contain:

```json
{
  "result": {
    "id": "2026-07-23T12:34:56.789Z-<random>",
    "filename": "scene.jpg",
    "timestamp": "2026-07-23T12:34:56.789Z",
    "category": "fire",
    "confidence": 0.91,
    "detectionCount": 2,
    "box": {
      "x": 0.12,
      "y": 0.18,
      "width": 0.41,
      "height": 0.53
    }
  }
}
```

Expected client errors use HTTP 400 or 413 and return
`{"error": "<safe actionable message>"}`. A missing or unloadable model uses
HTTP 503. Unexpected inference failures use HTTP 500 with a generic retry
message; implementation details are logged server-side and not exposed to the
browser.

## Browser Flow

The existing client-side file validation and preview remain the first line of
feedback. When the user selects **Analyze image**, the browser will:

1. Disable the Analyze button, hide the upload panel, show the current progress
   panel, and announce that analysis began.
2. Submit the original selected file to `/api/analyze`.
3. Validate the response shape before trusting it.
4. Attach the current thumbnail, persist the result through the existing
   transactional local-history path, and render it with the existing result
   panel.
5. Keep the current focus and live-region behavior for a completed result.

If the request fails, the browser will hide the progress panel, restore the
preview and Analyze button, show an inline accessible error, and announce that
analysis failed. The selected image remains available for a retry. Starting a
new selection or a new analysis clears stale inference errors.

The deterministic simulation functions will be removed from the production
path. Hashing may remain only if another existing behavior still needs it;
otherwise both simulation and its test coverage will be removed.

## Components

### `ui/inference.py`

Owns model loading, prediction settings, model-output extraction, class
validation, box clamping, and normalized prediction output. It has no Flask or
browser responsibilities.

### `ui/app.py`

Owns HTTP validation, safe JSON responses, application configuration, and
dependency injection. It does not interpret tensor results directly.

### `ui/static/js/core.js`

Owns pure result-contract validation in addition to the existing file and
history helpers. It contains no networking or model code.

### `ui/static/js/app.js`

Owns multipart submission, request/response error handling, UI state
transitions, and assembly of the server result with the local thumbnail.

## Error Handling and Resource Safety

Model loading will be lazy so importing the Flask application and running unit
tests do not immediately allocate model resources. Access to first-load
initialization will be guarded so concurrent initial requests do not load the
checkpoint more than once.

Uploaded bytes remain request-scoped. Flask's request-size limit and explicit
payload checks enforce the existing 10 MiB UI contract. Pillow verification
rejects malformed or mislabeled content before Ultralytics receives it.
Normalized boxes will be clamped to the image boundary, and malformed
third-party outputs will fail the request instead of being passed to history.

Client errors will be recoverable without reselecting the file. Repeated clicks
will be prevented while a request is in flight.

## Testing

Automated coverage will include:

- Inference normalization for no detections, one detection, multiple classes,
  highest-confidence selection, total detection count, and bounded boxes.
- Lazy model initialization and actionable missing-checkpoint behavior.
- Endpoint success with an injected fake inference service.
- Endpoint rejection of missing, empty, oversized, unsupported, mislabeled,
  malformed, and excessive-pixel images.
- Endpoint handling for model-unavailable and unexpected inference failures.
- Browser result-contract validation for valid fire, smoke, and clear results
  plus malformed server data.
- Browser controller checks confirming that real multipart `fetch` replaces
  simulation and that failure restores a retryable preview state.
- The existing Python and Node test suites.
- A real-checkpoint smoke test using a repository image, asserting a valid
  normalized response without asserting a particular probabilistic category.

## Documentation and Runtime

`ui/requirements.txt` will add the image-validation and inference dependencies
needed by the Flask process. The UI README will describe the checkpoint,
installation, startup, first-request model loading, CPU/GPU behavior, supported
formats, and test commands. Prototype/simulation claims will be replaced with
accurate real-inference limitations.

The application remains a local Flask application started from `ui/app.py`.

## Acceptance Criteria

- Selecting a supported image and pressing **Analyze image** runs the V2
  checkpoint and renders its real result.
- The server loads the checkpoint no more than once per application process.
- No uploaded image or prediction artifact is written to disk.
- Fire, smoke, and no-detection results conform to the existing UI result and
  history behavior.
- Failed inference leaves the selected image visible and retryable with an
  accessible error.
- Missing or invalid model files, uploads, and model failures return controlled
  responses without exposing stack traces.
- Existing responsive, accessibility, local-history, and safety-notice
  contracts continue to pass.
- Automated unit suites and a real-checkpoint smoke test pass.
