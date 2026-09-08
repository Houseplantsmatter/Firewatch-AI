# FireWatch Public Phone Deployment Design

## Goal

Make FireWatch usable from any modern phone through a permanent public HTTPS
URL without a paid server. Preserve the existing image upload, result display,
bounding box, local history, accessibility, responsive layout, and safety
messaging while moving inference from Flask/PyTorch to the user's browser.

The target is a static Progressive Web App deployed through GitHub Pages. It
must work over ordinary Wi-Fi or cellular data without requiring the developer's
computer to be running.

Live camera streams, video inference, user accounts, shared history, alerts,
and emergency-service integration are outside this change.

## Chosen Approach

Export the completed checkpoint at
`runs/detect/fire_smoke_v2/weights/best.pt` to ONNX and run it with ONNX
Runtime Web in a browser worker. The site will contain only static HTML, CSS,
JavaScript, WebAssembly runtime files, icons, and the exported model. GitHub
Pages will serve those assets through HTTPS.

This approach avoids the memory and CPU limits of free Python hosting and does
not have a sleeping server. Each phone downloads the model once, performs
analysis locally, and keeps selected images and history on that device.

Because browsers must download the ONNX model, the model weights and client
inference code are public assets. Training data, Python environments, training
runs other than the selected checkpoint, and local test images will not be
published.

## Architecture

The existing UI remains the presentation layer. Its production analysis path
will call a focused browser inference module instead of `POST /api/analyze`.
The inference module will communicate with a Web Worker so model loading and
prediction do not freeze the interface.

The worker will:

1. Load the vendored ONNX Runtime Web WebAssembly backend and the exported
   model.
2. Decode the selected image in the browser.
3. Letterbox it to 640 by 640 using the same geometry and fill value as the
   Ultralytics reference path.
4. Convert pixels to normalized RGB `NCHW` float input.
5. Run the ONNX session.
6. Decode the raw YOLO output, apply a 0.25 confidence threshold, class-aware
   non-maximum suppression at 0.45 IoU, and a 300-detection limit.
7. Map the highest-confidence accepted detection and total accepted count to
   the existing FireWatch result contract.

The UI thread will add the sanitized filename, UTC timestamp, a
`crypto.randomUUID()`-based unique ID, and
locally generated thumbnail before using the existing rendering and history
flow.

The class mapping remains:

- `0`: `smoke`
- `1`: `fire`

No image bytes, inference results, or history entries will be transmitted to a
FireWatch server.

## Static Site and Deployment Layout

The production entry page will use relative asset paths so it works both at a
GitHub Pages project URL such as `https://name.github.io/firewatch/` and from a
local static server. Flask-only template expressions and the production
dependency on `/api/analyze` will be removed from the public path.

A GitHub Actions workflow will build a minimal Pages artifact containing only:

- the static FireWatch interface;
- ONNX Runtime Web JavaScript and WebAssembly assets;
- the exported ONNX model;
- a web app manifest, service worker, and valid app icons.

The workflow will not publish Python virtual environments, datasets, training
images, labels, caches, or unrelated prediction runs. Repository ignore rules
will protect those large local artifacts from accidental commits.

The public URL will be usable without authentication. The deployable source
repository and the browser-downloaded model will be public. GitHub Pages deployment
will be the supported production path; a local static HTTP server will be the
supported development path. The Flask implementation will remain as a local
reference and parity oracle, but it will not be included in the Pages artifact
and the deployed app will not depend on it.

## Model Export and Versioning

A repeatable Python export script will create a fixed-shape 640 by 640 ONNX
artifact from the selected checkpoint with one-image batches and raw detector
output. The script will fail clearly if the checkpoint is missing, the class
mapping is not exactly smoke/fire, the export is malformed, or required output
metadata cannot be verified.

The exported model will have an explicit version identifier. That identifier
will also version the service-worker cache so a new model cannot be mixed with
stale inference code. Exported artifacts will be checked for GitHub's file-size
limits before deployment.

Implementation will inspect the actual exported input and output names and
shapes rather than assuming a particular Ultralytics tensor layout. Those
verified values will be recorded in a small model manifest consumed by the
worker.

## Browser Result Contract

The local inference result will preserve the current shape:

```json
{
  "id": "2026-08-24T12:34:56.789Z-<random>",
  "filename": "scene.jpg",
  "timestamp": "2026-08-24T12:34:56.789Z",
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

The displayed category, confidence, and normalized box come from the
highest-confidence accepted detection. `detectionCount` is the total number of
accepted fire and smoke detections after non-maximum suppression. With no
accepted detection, the result is `clear`, confidence and box are `null`, and
the count is zero.

`clear` continues to mean that no detection met the configured threshold; it
does not establish that a real scene is safe.

## Loading and Interaction Flow

The app will begin loading the inference runtime and model after the interface
becomes interactive. A visible status will distinguish model download,
ready-to-analyze, active analysis, and unavailable states. Selecting an image
may occur during model loading, but analysis will wait for readiness.

On analysis:

1. The app validates file type, file size, successful decode, positive
   dimensions, and the existing 40,000,000-pixel limit.
2. The Analyze button becomes unavailable and the accessible progress panel is
   shown.
3. The worker performs preprocessing, prediction, and postprocessing.
4. The UI validates the worker result before rendering or storing it.
5. The image and result remain available for another analysis or local history.

Only one analysis will run at a time. Replacing an image invalidates stale work
so a late worker response cannot overwrite the new selection.

## Offline and PWA Behavior

The app will include non-empty 192-pixel and 512-pixel icons and a manifest
whose paths and scope work beneath the GitHub Pages project path.

The service worker will cache the versioned application shell, runtime assets,
and model after successful retrieval. It will never cache selected user images.
An interrupted first model download will produce a retryable error. Once all
required assets have been cached successfully, later visits may work offline on
the same device, subject to browser storage eviction.

The initial visit still requires an internet connection and may take longer
because the phone must download and initialize the model.

## Compatibility and Failure Handling

The supported baseline is one of the two most recent stable major versions of
mobile Safari on iOS or Chrome on Android at implementation time,
with WebAssembly, Web Workers, typed arrays, canvas image decoding, and enough
memory for the model and a 640-pixel tensor. WebAssembly will be the only
production backend in this version so every supported browser follows the same
validated inference path.

Failures will be categorized without exposing internal stack traces:

- unsupported browser: explain that a current Safari or Chrome version is
  required;
- model or runtime download failure: keep the selected image and offer retry;
- invalid or undecodable image: show the current actionable upload error;
- insufficient browser memory or inference failure: restore a retryable state
  and suggest closing other tabs or trying another current device;
- stale or malformed worker response: reject it and show the generic analysis
  failure message.

The existing emergency safety notice remains visible. No result may claim that
a scene is safe or replace alarms, official alerts, firefighters, or 911.

## Testing and Validation

Automated tests will cover:

- preprocessing geometry, pixel normalization, and letterbox reversal;
- confidence filtering, class mapping, class-aware non-maximum suppression,
  highest-confidence selection, detection count, and normalized boxes;
- clear results and malformed tensor rejection;
- worker message validation, stale-response rejection, retry behavior, and
  single-analysis locking;
- file validation and existing local-history contracts;
- relative asset paths, manifest validity, required icon files, service-worker
  cache contents, and absence of private images from caches;
- a production build check that rejects missing or oversized assets.

A parity suite will run at least twelve local images through both the current
PyTorch path and the exported ONNX path: at least four known-fire examples,
four known-smoke examples, and four examples expected to produce a clear
result. The two paths must return the same category and accepted detection
count for every image. Non-clear confidence values and each normalized box
coordinate may differ by no more than 0.02 in absolute value. Any larger or
systematic mismatch must be resolved before the browser path becomes the
production path.

Manual acceptance testing will cover at least one current iPhone browser and
one current Android browser over cellular or a network different from the
developer computer. It will verify first load, repeat load, image analysis,
history, add-to-home-screen behavior, and controlled failure while offline
before the first model download.

## Acceptance Criteria

- The deployed GitHub Pages URL opens from an unrelated phone over cellular or
  Wi-Fi without the developer computer running.
- A supported JPG, PNG, or WebP image can be analyzed entirely in the browser.
- Browser output matches the existing PyTorch reference within the approved
  parity tolerances.
- Fire, smoke, and clear results retain the existing result, bounding-box,
  history, accessibility, responsive-layout, and safety behavior.
- Network inspection confirms that selected image data is not uploaded.
- The app gives actionable, retryable feedback for model-loading, unsupported
  browser, invalid-image, memory, and inference failures.
- The public deployment contains no training dataset, virtual environment,
  local test image, or unrelated run artifact.
- Automated suites pass, and manual iPhone and Android checks pass before the
  competition link is submitted.
