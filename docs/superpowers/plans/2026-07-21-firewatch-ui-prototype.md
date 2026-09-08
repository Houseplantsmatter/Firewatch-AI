# FireWatch AI Responsive UI Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a responsive Flask-served FireWatch AI prototype that simulates image analysis, renders understandable results, and preserves a 12-item browser-local history.

**Architecture:** Flask serves one semantic HTML page and local static assets; it never receives the selected image. A framework-free JavaScript core validates metadata, derives deterministic results, and normalizes history, while a separate DOM controller manages image decoding, thumbnails, view transitions, rendering, and local storage. This keeps a clean boundary for replacing the simulator with a future inference request.

**Tech Stack:** Python 3.8+, Flask 3.0.x, HTML5, CSS3, vanilla JavaScript, Python `unittest`, and Node's built-in test runner.

## Global Constraints

- All new runtime code, tests, dependencies, and setup instructions must live under `ui`.
- Do not modify existing detector, training, configuration, data, weight, or root test files.
- Do not import Ultralytics, YOLO, OpenCV, Torch, or any model checkpoint.
- Accept JPG, PNG, and WebP files up to exactly 10 MB.
- The result UI must not contain the phrase “Demo result.”
- Keep the persistent safety notice visible in Analyze and History.
- Store at most the 12 newest thumbnail-based history entries in browser local storage.
- Support a 320-pixel viewport without horizontal scrolling.
- Use text and icons as well as color for fire, smoke, and no-concern states.
- The workspace currently has an empty `.git` directory and is not recognized as a Git repository; run commit steps only if valid Git metadata is restored before execution.

---

## Planned File Structure

- `ui/app.py` — Flask application factory, root route, and development entry point.
- `ui/requirements.txt` — UI-only Flask dependency.
- `ui/templates/index.html` — semantic Analyze, Result, and History application shell.
- `ui/static/css/styles.css` — Calm & Civic tokens, components, responsive rules, focus, and reduced motion.
- `ui/static/js/core.js` — pure validation, hashing, simulation, and history normalization functions.
- `ui/static/js/app.js` — DOM state, upload decoding, thumbnail creation, rendering, navigation, and persistence.
- `ui/tests/test_app.py` — Flask delivery, semantic-shell, safety-copy, and containment tests.
- `ui/tests/core.test.js` — pure JavaScript behavior tests using `node:test`.
- `ui/README.md` — setup, run, test, limitations, and future integration seam.

### Task 1: Flask Server and Semantic Application Shell

**Files:**
- Create: `ui/app.py`
- Create: `ui/requirements.txt`
- Create: `ui/templates/index.html`
- Create: `ui/tests/test_app.py`

**Interfaces:**
- Consumes: no project runtime code.
- Produces: `create_app(test_config: dict | None = None) -> Flask`, route `GET /`, and stable element IDs used by later JavaScript.

- [ ] **Step 1: Write the failing Flask delivery tests**

Create `ui/tests/test_app.py` with tests that import only `ui/app.py`, create a Flask test client, and assert the approved shell and safety constraints:

```python
import importlib.util
import sys
import unittest
from pathlib import Path

UI_DIR = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("firewatch_ui_app", UI_DIR / "app.py")
APP_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = APP_MODULE
SPEC.loader.exec_module(APP_MODULE)


class AppTests(unittest.TestCase):
    def setUp(self):
        self.app = APP_MODULE.create_app({"TESTING": True})
        self.client = self.app.test_client()

    def test_index_serves_firewatch_shell(self):
        response = self.client.get("/")
        body = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("FireWatch AI", body)
        for element_id in (
            "analyze-view", "history-view", "file-input", "drop-zone",
            "analyze-button", "result-panel", "history-list", "live-region",
        ):
            self.assertIn('id="{}"'.format(element_id), body)

    def test_index_contains_safety_notice_and_no_demo_result_label(self):
        body = self.client.get("/").get_data(as_text=True)
        self.assertIn("does not replace", body)
        self.assertIn("calling 911", body)
        self.assertNotIn("Demo result", body)

    def test_static_assets_are_local_and_declared(self):
        body = self.client.get("/").get_data(as_text=True)
        self.assertIn('/static/css/styles.css', body)
        self.assertIn('/static/js/core.js', body)
        self.assertIn('/static/js/app.js', body)
        self.assertNotIn("https://", body)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run from `ui` after installing its dependency: `python -m unittest discover -s tests -p "test_*.py" -v`

Expected: FAIL because `app.py` does not exist.

- [ ] **Step 3: Implement the server and semantic shell**

Create `ui/requirements.txt`:

```text
Flask>=3.0,<3.1
```

Create `ui/app.py`:

```python
from flask import Flask, render_template


def create_app(test_config=None):
    app = Flask(__name__)
    if test_config:
        app.config.update(test_config)

    @app.get("/")
    def index():
        return render_template("index.html")

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=True)
```

Create `ui/templates/index.html` as a complete semantic document. It must contain:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="FireWatch AI image-analysis interface prototype">
  <title>FireWatch AI</title>
  <link rel="stylesheet" href="{{ url_for('static', filename='css/styles.css') }}">
</head>
<body>
  <a class="skip-link" href="#main-content">Skip to main content</a>
  <header class="site-header">
    <a class="brand" href="#analyze" aria-label="FireWatch AI home"><span aria-hidden="true">◆</span> FireWatch <em>AI</em></a>
    <nav aria-label="Primary navigation">
      <button class="nav-button is-active" type="button" data-view="analyze" aria-current="page">Analyze</button>
      <button class="nav-button" type="button" data-view="history">History <span id="history-count" class="count-badge">0</span></button>
    </nav>
    <p class="ready-status"><span aria-hidden="true">●</span> Ready</p>
  </header>

  <main id="main-content">
    <section id="analyze-view" class="view" aria-labelledby="analyze-title">
      <div id="upload-panel" class="upload-layout">
        <div class="intro-copy"><p class="eyebrow">Image analysis</p><h1 id="analyze-title">See the warning signs sooner.</h1><p>Check a photo for possible fire or smoke.</p></div>
        <div class="upload-card">
          <input id="file-input" class="visually-hidden" type="file" accept="image/jpeg,image/png,image/webp">
          <label id="drop-zone" class="drop-zone" for="file-input">
            <span class="upload-icon" aria-hidden="true">↑</span><strong>Upload an image</strong>
            <span>Drag and drop, or choose a JPG, PNG, or WebP file</span><small>Maximum size: 10 MB</small>
          </label>
          <p id="upload-error" class="inline-message error-message" role="alert" hidden></p>
          <div id="preview-panel" class="preview-panel" hidden>
            <img id="preview-image" alt="Selected image preview">
            <div><strong id="file-name"></strong><span id="file-size"></span></div>
            <button id="replace-button" class="button secondary" type="button">Replace</button>
            <button id="analyze-button" class="button primary" type="button">Analyze image</button>
          </div>
        </div>
      </div>

      <section id="progress-panel" class="progress-panel" aria-labelledby="progress-title" hidden>
        <p class="eyebrow">Analyzing image</p><h2 id="progress-title">Looking for possible fire or smoke…</h2>
        <div class="progress-track" aria-hidden="true"><span></span></div>
      </section>

      <section id="result-panel" class="result-layout" aria-labelledby="result-title" hidden>
        <div class="result-visual"><div class="image-stage"><img id="result-image" alt="Analyzed scene"><div id="detection-box" class="detection-box" hidden><span id="box-label"></span></div></div></div>
        <aside class="result-summary"><p id="result-kicker" class="eyebrow"></p><h2 id="result-title"></h2><p id="result-description"></p><dl><div><dt>Confidence</dt><dd id="result-confidence"></dd></div><div><dt>Detections</dt><dd id="result-count"></dd></div><div><dt>Analyzed</dt><dd id="result-time"></dd></div></dl><div class="guidance"><h3>What to do next</h3><p id="result-guidance"></p></div><button id="analyze-another-button" class="button primary" type="button">Analyze another image</button><button class="button secondary" data-view="history" type="button">View history</button></aside>
      </section>
    </section>

    <section id="history-view" class="view" aria-labelledby="history-title" hidden>
      <div class="section-heading"><div><p class="eyebrow">Saved on this device</p><h1 id="history-title">Detection history</h1></div><button id="clear-history-button" class="button text-button" type="button">Clear history</button></div>
      <div id="history-filters" class="filter-group" aria-label="Filter history"><button class="filter-button is-active" data-filter="all" type="button">All</button><button class="filter-button" data-filter="fire" type="button">Fire</button><button class="filter-button" data-filter="smoke" type="button">Smoke</button><button class="filter-button" data-filter="clear" type="button">No concern</button></div>
      <p id="history-message" class="inline-message" hidden></p><div id="history-list" class="history-list"></div>
    </section>
  </main>

  <footer><strong>Safety notice</strong><p>FireWatch AI is an additional safety tool. It does not replace working smoke alarms, official emergency alerts, trained firefighters, or calling 911 during an emergency.</p></footer>
  <div id="live-region" class="visually-hidden" aria-live="polite" aria-atomic="true"></div>
  <script src="{{ url_for('static', filename='js/core.js') }}"></script>
  <script src="{{ url_for('static', filename='js/app.js') }}" defer></script>
</body>
</html>
```

- [ ] **Step 4: Run the Flask tests to verify they pass**

Run: `python -m unittest discover -s tests -p "test_*.py" -v`

Expected: 3 tests PASS.

- [ ] **Step 5: Commit if Git is available**

```powershell
git add ui/app.py ui/requirements.txt ui/templates/index.html ui/tests/test_app.py
git commit -m "feat: add FireWatch UI server shell"
```

### Task 2: Deterministic Simulation and History Core

**Files:**
- Create: `ui/static/js/core.js`
- Create: `ui/tests/core.test.js`

**Interfaces:**
- Consumes: file metadata shaped as `{name: string, size: number, type: string, lastModified?: number}`.
- Produces: global `FireWatchCore` with `validateFile`, `hashMetadata`, `simulateResult`, `normalizeHistory`, `MAX_BYTES`, `MAX_HISTORY`, and `STORAGE_KEY`.

- [ ] **Step 1: Write failing pure-JavaScript tests**

Create `ui/tests/core.test.js` using `node:test`. Cover exact limits, MIME/extension agreement, deterministic results, normalized boxes, category coverage, newest-first truncation, and malformed history removal:

```javascript
const test = require("node:test");
const assert = require("node:assert/strict");
const core = require("../static/js/core.js");

test("accepts supported images at the 10 MB boundary", () => {
  assert.equal(core.validateFile({name: "scene.JPG", type: "image/jpeg", size: core.MAX_BYTES}), "");
});

test("rejects unsupported, mismatched, empty, and oversized files", () => {
  assert.match(core.validateFile({name: "scene.gif", type: "image/gif", size: 20}), /JPG, PNG, or WebP/);
  assert.match(core.validateFile({name: "scene.png", type: "image/jpeg", size: 20}), /does not match/);
  assert.match(core.validateFile({name: "scene.jpg", type: "image/jpeg", size: 0}), /empty/);
  assert.match(core.validateFile({name: "scene.jpg", type: "image/jpeg", size: core.MAX_BYTES + 1}), /10 MB/);
});

test("simulates the same normalized result for the same metadata", () => {
  const metadata = {name: "ridge.jpg", type: "image/jpeg", size: 48291, lastModified: 1234};
  const first = core.simulateResult(metadata, "2026-07-21T20:00:00.000Z", "thumb");
  const second = core.simulateResult(metadata, "2026-07-21T20:00:00.000Z", "thumb");
  assert.deepEqual(first, second);
  assert.ok(["fire", "smoke", "clear"].includes(first.category));
  assert.equal(first.confidence === null || (first.confidence >= 0.74 && first.confidence <= 0.96), true);
  if (first.box) Object.values(first.box).forEach(value => assert.ok(value >= 0 && value <= 1));
});

test("metadata samples exercise all result categories", () => {
  const categories = new Set();
  for (let index = 0; index < 200; index += 1) {
    categories.add(core.simulateResult({name: `image-${index}.jpg`, type: "image/jpeg", size: index + 1, lastModified: index}, "2026-07-21T20:00:00.000Z", "").category);
  }
  assert.deepEqual([...categories].sort(), ["clear", "fire", "smoke"]);
});

test("history is newest first, valid only, and capped at 12", () => {
  const entries = Array.from({length: 15}, (_, index) => ({id: String(index), filename: `${index}.jpg`, timestamp: new Date(2026, 0, index + 1).toISOString(), category: index % 2 ? "fire" : "clear", confidence: index % 2 ? 0.8 : null, detectionCount: index % 2 ? 1 : 0, box: index % 2 ? {x: .2, y: .2, width: .3, height: .3} : null, thumbnail: "data:image/jpeg;base64,x"}));
  entries.push({bad: true});
  const normalized = core.normalizeHistory(entries);
  assert.equal(normalized.length, core.MAX_HISTORY);
  assert.equal(normalized[0].id, "14");
  assert.equal(normalized[normalized.length - 1].id, "3");
});
```

- [ ] **Step 2: Run the Node tests to verify they fail**

Run: `node --test tests/core.test.js`

Expected: FAIL because `static/js/core.js` does not exist.

- [ ] **Step 3: Implement the pure core module**

Create `ui/static/js/core.js` as a strict-mode UMD module. Use these constants and signatures:

```javascript
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.FireWatchCore = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";
  const MAX_BYTES = 10 * 1024 * 1024;
  const MAX_HISTORY = 12;
  const STORAGE_KEY = "firewatch-ai-history-v1";
  const TYPES = {"image/jpeg": ["jpg", "jpeg"], "image/png": ["png"], "image/webp": ["webp"]};

  function validateFile(file) {
    if (!file || typeof file.name !== "string") return "Choose an image to continue.";
    if (!Number.isFinite(file.size) || file.size <= 0) return "This image is empty. Choose a different file.";
    if (file.size > MAX_BYTES) return "This image is larger than 10 MB. Choose a smaller file.";
    const extension = file.name.includes(".") ? file.name.split(".").pop().toLowerCase() : "";
    if (!Object.prototype.hasOwnProperty.call(TYPES, file.type)) return "Choose a JPG, PNG, or WebP image.";
    if (!TYPES[file.type].includes(extension)) return "The file extension does not match the image type.";
    return "";
  }

  function hashMetadata(file) {
    const text = [file.name.toLowerCase(), file.size, file.type, file.lastModified || 0].join("|");
    let hash = 2166136261;
    for (let index = 0; index < text.length; index += 1) {
      hash ^= text.charCodeAt(index);
      hash = Math.imul(hash, 16777619);
    }
    return hash >>> 0;
  }

  function simulateResult(file, timestamp, thumbnail) {
    const seed = hashMetadata(file);
    const category = ["smoke", "fire", "clear"][seed % 3];
    const detected = category !== "clear";
    const box = detected ? {
      x: 0.08 + ((seed >>> 3) % 36) / 100,
      y: 0.08 + ((seed >>> 9) % 30) / 100,
      width: 0.24 + ((seed >>> 14) % 18) / 100,
      height: 0.25 + ((seed >>> 19) % 24) / 100,
    } : null;
    return {
      id: `${timestamp}-${seed}`,
      filename: file.name,
      timestamp,
      category,
      confidence: detected ? (74 + (seed % 23)) / 100 : null,
      detectionCount: detected ? 1 : 0,
      box,
      thumbnail,
    };
  }

  function isValidEntry(entry) {
    return entry && typeof entry.id === "string" && typeof entry.filename === "string" &&
      !Number.isNaN(Date.parse(entry.timestamp)) && ["fire", "smoke", "clear"].includes(entry.category) &&
      typeof entry.thumbnail === "string";
  }

  function normalizeHistory(value) {
    if (!Array.isArray(value)) return [];
    return value.filter(isValidEntry).sort((a, b) => Date.parse(b.timestamp) - Date.parse(a.timestamp)).slice(0, MAX_HISTORY);
  }

  return {MAX_BYTES, MAX_HISTORY, STORAGE_KEY, validateFile, hashMetadata, simulateResult, normalizeHistory};
});
```

- [ ] **Step 4: Run the core tests and syntax check**

Run: `node --test tests/core.test.js`

Expected: 5 tests PASS.

Run: `node --check static/js/core.js`

Expected: exit code 0 with no output.

- [ ] **Step 5: Commit if Git is available**

```powershell
git add ui/static/js/core.js ui/tests/core.test.js
git commit -m "feat: add deterministic FireWatch simulation core"
```

### Task 3: Browser Controller and Complete Interaction Flow

**Files:**
- Create: `ui/static/js/app.js`
- Modify: `ui/tests/test_app.py`

**Interfaces:**
- Consumes: `window.FireWatchCore` and the stable IDs from Task 1.
- Produces: working upload, decode, preview, analyze, result, navigation, filter, reopen, delete, clear, and persistence behaviors.

- [ ] **Step 1: Extend delivery tests for controller wiring and no model dependency**

Add to `ui/tests/test_app.py`:

```python
    def test_browser_controller_uses_core_without_model_dependencies(self):
        app_js = (UI_DIR / "static" / "js" / "app.js").read_text(encoding="utf-8")
        self.assertIn("FireWatchCore", app_js)
        for forbidden in ("ultralytics", "torch", "cv2", "best.pt", "YOLO("):
            self.assertNotIn(forbidden, app_js)

    def test_every_ui_runtime_file_stays_under_ui(self):
        expected = {
            UI_DIR / "app.py", UI_DIR / "requirements.txt", UI_DIR / "templates" / "index.html",
            UI_DIR / "static" / "js" / "core.js", UI_DIR / "static" / "js" / "app.js",
            UI_DIR / "static" / "css" / "styles.css",
        }
        for path in expected:
            path.resolve().relative_to(UI_DIR.resolve())
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `python -m unittest tests.test_app.AppTests.test_browser_controller_uses_core_without_model_dependencies tests.test_app.AppTests.test_every_ui_runtime_file_stays_under_ui -v`

Expected: FAIL because `static/js/app.js` and `static/css/styles.css` do not exist.

- [ ] **Step 3: Implement controller state and image helpers**

Create `ui/static/js/app.js` inside a `DOMContentLoaded` handler. Define one state object:

```javascript
const state = {file: null, imageUrl: "", thumbnail: "", result: null, history: [], filter: "all", storageAvailable: true};
```

Implement `setView(name)`, `setUploadError(message)`, `announce(message)`, `formatBytes(bytes)`, `decodeImage(file)`, `makeThumbnail(image)`, `selectFile(file)`, and `resetAnalyze()`. `decodeImage` must reject on `image.onerror`; `selectFile` must call `FireWatchCore.validateFile`, revoke the old object URL, confirm browser decoding, create a maximum-320-by-200 JPEG thumbnail, and only then show the preview. Drag handlers must prevent defaults and toggle `.is-dragging`.

- [ ] **Step 4: Implement result rendering and analysis state**

Implement `renderResult(result, fullImageUrl)`. For fire and smoke, set the normalized box through percentage `left`, `top`, `width`, and `height`, show an uppercase category plus rounded percentage in `box-label`, and display review-needed guidance. For clear, hide the box, show `No concern found`, display `—` for confidence, and explicitly say the result cannot confirm that a real scene is safe.

Implement `analyzeCurrentFile()` as an async function that disables Analyze, reveals `progress-panel`, updates the live region, waits 900 ms, creates a result using `FireWatchCore.simulateResult(state.file, new Date().toISOString(), state.thumbnail)`, saves it, renders it, hides progress, reveals result, and focuses `result-title`. Guard against re-entry while the button is disabled.

- [ ] **Step 5: Implement resilient local history**

Implement `loadHistory()`, `saveHistory()`, `renderHistory()`, `openHistoryItem(id)`, `deleteHistoryItem(id)`, and `clearHistory()`. Parse storage inside `try/catch`, normalize with `FireWatchCore.normalizeHistory`, keep the in-memory result if saving fails, and show `History is unavailable in this browser, but your current result is still visible.` Filter using exact categories. Build history rows with DOM methods and `textContent`, not `innerHTML`. Clear must call `window.confirm("Clear all saved detection history from this device?")`.

- [ ] **Step 6: Wire controls and make static assets resolvable**

Attach change, click, drag, navigation, filter, delete, and clear listeners once. On initial load, call `loadHistory()`, `renderHistory()`, and `setView(location.hash === "#history" ? "history" : "analyze")`. The stylesheet is created in Task 4; this task verifies controller behavior independently.

Use this complete controller implementation for Steps 3–6:

```javascript
document.addEventListener("DOMContentLoaded", () => {
  "use strict";
  const core = window.FireWatchCore;
  const byId = id => document.getElementById(id);
  const state = {file: null, imageUrl: "", thumbnail: "", result: null, history: [], filter: "all", storageAvailable: true};
  const nodes = {
    analyzeView: byId("analyze-view"), historyView: byId("history-view"), uploadPanel: byId("upload-panel"),
    fileInput: byId("file-input"), dropZone: byId("drop-zone"), uploadError: byId("upload-error"),
    previewPanel: byId("preview-panel"), previewImage: byId("preview-image"), fileName: byId("file-name"),
    fileSize: byId("file-size"), replaceButton: byId("replace-button"), analyzeButton: byId("analyze-button"),
    progressPanel: byId("progress-panel"), resultPanel: byId("result-panel"), resultImage: byId("result-image"),
    detectionBox: byId("detection-box"), boxLabel: byId("box-label"), resultKicker: byId("result-kicker"),
    resultTitle: byId("result-title"), resultDescription: byId("result-description"),
    resultConfidence: byId("result-confidence"), resultCount: byId("result-count"), resultTime: byId("result-time"),
    resultGuidance: byId("result-guidance"), analyzeAnother: byId("analyze-another-button"),
    historyCount: byId("history-count"), historyList: byId("history-list"), historyMessage: byId("history-message"),
    clearHistory: byId("clear-history-button"), liveRegion: byId("live-region"), filters: byId("history-filters"),
  };

  function announce(message) { nodes.liveRegion.textContent = ""; window.setTimeout(() => { nodes.liveRegion.textContent = message; }, 20); }
  function setUploadError(message) { nodes.uploadError.textContent = message; nodes.uploadError.hidden = !message; }
  function formatBytes(bytes) { return bytes >= 1024 * 1024 ? `${(bytes / 1024 / 1024).toFixed(1)} MB` : `${Math.ceil(bytes / 1024)} KB`; }
  function setView(name) {
    const history = name === "history";
    nodes.analyzeView.hidden = history; nodes.historyView.hidden = !history;
    document.querySelectorAll("[data-view]").forEach(button => {
      const active = button.dataset.view === name && button.classList.contains("nav-button");
      button.classList.toggle("is-active", active);
      if (button.classList.contains("nav-button")) active ? button.setAttribute("aria-current", "page") : button.removeAttribute("aria-current");
    });
    window.history.replaceState(null, "", history ? "#history" : "#analyze");
    if (history) renderHistory();
  }
  function decodeImage(file) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file); const image = new Image();
      image.onload = () => resolve({image, url});
      image.onerror = () => { URL.revokeObjectURL(url); reject(new Error("decode")); };
      image.src = url;
    });
  }
  function makeThumbnail(image) {
    const scale = Math.min(1, 320 / image.naturalWidth, 200 / image.naturalHeight);
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(image.naturalWidth * scale)); canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
    canvas.getContext("2d").drawImage(image, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.72);
  }
  async function selectFile(file) {
    setUploadError("");
    const error = core.validateFile(file); if (error) { setUploadError(error); return; }
    try {
      const decoded = await decodeImage(file);
      if (state.imageUrl) URL.revokeObjectURL(state.imageUrl);
      state.file = file; state.imageUrl = decoded.url; state.thumbnail = makeThumbnail(decoded.image);
      nodes.previewImage.src = state.imageUrl; nodes.fileName.textContent = file.name; nodes.fileSize.textContent = formatBytes(file.size);
      nodes.dropZone.hidden = true; nodes.previewPanel.hidden = false; nodes.analyzeButton.disabled = false;
      announce(`${file.name} is ready to analyze.`);
    } catch (_) { setUploadError("This image could not be read. Choose a different file."); }
  }
  function resetAnalyze() {
    state.file = null; state.result = null; state.thumbnail = "";
    if (state.imageUrl) URL.revokeObjectURL(state.imageUrl); state.imageUrl = "";
    nodes.fileInput.value = ""; nodes.previewImage.removeAttribute("src"); nodes.resultImage.removeAttribute("src");
    nodes.previewPanel.hidden = true; nodes.dropZone.hidden = false; nodes.progressPanel.hidden = true; nodes.resultPanel.hidden = true; nodes.uploadPanel.hidden = false;
    nodes.analyzeButton.disabled = false; setUploadError(""); nodes.fileInput.focus();
  }
  function copyFor(category) {
    if (category === "fire") return {title: "Possible fire detected.", kicker: "Review needed", description: "A possible flame pattern was identified in this image.", guidance: "Check the original scene. If there is immediate danger, move to safety and call emergency services."};
    if (category === "smoke") return {title: "Possible smoke detected.", kicker: "Review needed", description: "A possible smoke pattern was identified in this image.", guidance: "Check the original scene. If there is immediate danger, move to safety and call emergency services."};
    return {title: "No concern found.", kicker: "Analysis complete", description: "No simulated fire or smoke region was identified.", guidance: "This result cannot confirm that a real scene is safe. Continue to use working smoke alarms and official emergency information."};
  }
  function renderResult(result, imageUrl) {
    const copy = copyFor(result.category); state.result = result; nodes.resultPanel.dataset.category = result.category;
    nodes.resultImage.src = imageUrl; nodes.resultKicker.textContent = copy.kicker; nodes.resultTitle.textContent = copy.title;
    nodes.resultDescription.textContent = copy.description; nodes.resultGuidance.textContent = copy.guidance;
    nodes.resultConfidence.textContent = result.confidence === null ? "—" : `${Math.round(result.confidence * 100)}%`;
    nodes.resultCount.textContent = String(result.detectionCount); nodes.resultTime.textContent = new Date(result.timestamp).toLocaleString();
    nodes.detectionBox.hidden = !result.box;
    if (result.box) {
      Object.assign(nodes.detectionBox.style, {left: `${result.box.x * 100}%`, top: `${result.box.y * 100}%`, width: `${result.box.width * 100}%`, height: `${result.box.height * 100}%`});
      nodes.boxLabel.textContent = `${result.category.toUpperCase()} · ${Math.round(result.confidence * 100)}%`;
    }
    nodes.uploadPanel.hidden = true; nodes.progressPanel.hidden = true; nodes.resultPanel.hidden = false;
  }
  function loadHistory() {
    try { state.history = core.normalizeHistory(JSON.parse(localStorage.getItem(core.STORAGE_KEY) || "[]")); }
    catch (_) { state.history = []; state.storageAvailable = false; }
  }
  function saveHistory(result) {
    state.history = core.normalizeHistory([result, ...state.history.filter(item => item.id !== result.id)]);
    try { localStorage.setItem(core.STORAGE_KEY, JSON.stringify(state.history)); }
    catch (_) { state.storageAvailable = false; nodes.historyMessage.textContent = "History is unavailable in this browser, but your current result is still visible."; nodes.historyMessage.hidden = false; }
    renderHistory();
  }
  function historyCard(result) {
    const article = document.createElement("article"); article.className = `history-card ${result.category}`;
    const image = document.createElement("img"); image.src = result.thumbnail; image.alt = "";
    const copy = copyFor(result.category); const details = document.createElement("div"); const title = document.createElement("h2");
    title.textContent = copy.title; const meta = document.createElement("p"); meta.textContent = `${new Date(result.timestamp).toLocaleString()} · ${result.filename}`;
    details.append(title, meta); const score = document.createElement("strong"); score.textContent = result.confidence === null ? "Clear" : `${Math.round(result.confidence * 100)}%`;
    const actions = document.createElement("div"); actions.className = "history-actions";
    const open = document.createElement("button"); open.type = "button"; open.className = "button secondary"; open.dataset.openId = result.id; open.textContent = "Open";
    const remove = document.createElement("button"); remove.type = "button"; remove.className = "button text-button"; remove.dataset.deleteId = result.id; remove.textContent = "Delete";
    actions.append(open, remove); article.append(image, details, score, actions); return article;
  }
  function renderHistory() {
    nodes.historyList.replaceChildren(); nodes.historyCount.textContent = String(state.history.length); nodes.clearHistory.disabled = state.history.length === 0;
    document.querySelectorAll("[data-filter]").forEach(button => button.classList.toggle("is-active", button.dataset.filter === state.filter));
    const filtered = state.filter === "all" ? state.history : state.history.filter(item => item.category === state.filter);
    if (!state.storageAvailable) { nodes.historyMessage.textContent = "History is unavailable in this browser, but your current result is still visible."; nodes.historyMessage.hidden = false; }
    else if (!filtered.length) { nodes.historyMessage.textContent = state.history.length ? "No saved results match this filter." : "No analyses saved yet. Your recent results will appear here."; nodes.historyMessage.hidden = false; }
    else nodes.historyMessage.hidden = true;
    filtered.forEach(item => nodes.historyList.append(historyCard(item)));
  }
  function openHistoryItem(id) { const result = state.history.find(item => item.id === id); if (!result) return; setView("analyze"); renderResult(result, result.thumbnail); nodes.resultTitle.focus(); }
  function deleteHistoryItem(id) { state.history = state.history.filter(item => item.id !== id); try { localStorage.setItem(core.STORAGE_KEY, JSON.stringify(state.history)); } catch (_) { state.storageAvailable = false; } renderHistory(); announce("History item deleted."); }
  function clearHistory() { if (!window.confirm("Clear all saved detection history from this device?")) return; state.history = []; try { localStorage.removeItem(core.STORAGE_KEY); } catch (_) { state.storageAvailable = false; } renderHistory(); announce("Detection history cleared."); }
  async function analyzeCurrentFile() {
    if (!state.file || nodes.analyzeButton.disabled) return; nodes.analyzeButton.disabled = true; nodes.uploadPanel.hidden = true; nodes.progressPanel.hidden = false; announce("Analyzing image.");
    await new Promise(resolve => window.setTimeout(resolve, 900));
    const result = core.simulateResult(state.file, new Date().toISOString(), state.thumbnail); saveHistory(result); renderResult(result, state.imageUrl); announce(`${copyFor(result.category).title} Analysis complete.`); nodes.resultTitle.tabIndex = -1; nodes.resultTitle.focus();
  }

  nodes.fileInput.addEventListener("change", event => { if (event.target.files[0]) selectFile(event.target.files[0]); });
  ["dragenter", "dragover"].forEach(type => nodes.dropZone.addEventListener(type, event => { event.preventDefault(); nodes.dropZone.classList.add("is-dragging"); }));
  ["dragleave", "drop"].forEach(type => nodes.dropZone.addEventListener(type, event => { event.preventDefault(); nodes.dropZone.classList.remove("is-dragging"); if (type === "drop" && event.dataTransfer.files[0]) selectFile(event.dataTransfer.files[0]); }));
  nodes.replaceButton.addEventListener("click", () => nodes.fileInput.click()); nodes.analyzeButton.addEventListener("click", analyzeCurrentFile); nodes.analyzeAnother.addEventListener("click", resetAnalyze);
  document.querySelectorAll("[data-view]").forEach(button => button.addEventListener("click", () => setView(button.dataset.view)));
  nodes.filters.addEventListener("click", event => { const button = event.target.closest("[data-filter]"); if (button) { state.filter = button.dataset.filter; renderHistory(); } });
  nodes.historyList.addEventListener("click", event => { const open = event.target.closest("[data-open-id]"); const remove = event.target.closest("[data-delete-id]"); if (open) openHistoryItem(open.dataset.openId); if (remove) deleteHistoryItem(remove.dataset.deleteId); });
  nodes.clearHistory.addEventListener("click", clearHistory); loadHistory(); renderHistory(); setView(location.hash === "#history" ? "history" : "analyze");
});
```

- [ ] **Step 7: Run controller and syntax checks**

Run: `python -m unittest discover -s tests -p "test_*.py" -v`

Expected: 5 tests PASS.

Run: `node --check static/js/app.js`

Expected: exit code 0 with no output.

- [ ] **Step 8: Commit if Git is available**

```powershell
git add ui/static/js/app.js ui/static/css/styles.css ui/tests/test_app.py
git commit -m "feat: add FireWatch prototype interactions"
```

### Task 4: Calm & Civic Responsive Design and Accessibility

**Files:**
- Modify: `ui/static/css/styles.css`
- Modify: `ui/tests/test_app.py`

**Interfaces:**
- Consumes: semantic classes and state attributes from Tasks 1 and 3.
- Produces: approved Calm & Civic presentation at 320 px and wider, visible interaction states, and reduced-motion support.

- [ ] **Step 1: Add failing design-contract tests**

Add this test to `ui/tests/test_app.py`:

```python
    def test_styles_include_responsive_focus_and_reduced_motion_contracts(self):
        css = (UI_DIR / "static" / "css" / "styles.css").read_text(encoding="utf-8")
        for required in (
            "--forest: #18312b", "--ember: #d95028", "@media (max-width: 760px)",
            "@media (prefers-reduced-motion: reduce)", ":focus-visible", ".result-layout",
            ".history-list", ".detection-box", "overflow-wrap: anywhere",
        ):
            self.assertIn(required, css)
```

- [ ] **Step 2: Run the targeted test to verify it fails**

Run: `python -m unittest tests.test_app.AppTests.test_styles_include_responsive_focus_and_reduced_motion_contracts -v`

Expected: FAIL because the style contract is absent.

- [ ] **Step 3: Implement the complete Calm & Civic stylesheet**

Replace `ui/static/css/styles.css` with a mobile-first stylesheet using these exact tokens:

```css
:root {
  --forest: #18312b; --forest-soft: #285c46; --ember: #d95028;
  --ember-soft: #fbe6db; --canvas: #f4f2eb; --surface: #fffef9;
  --line: #d8d8ce; --muted: #65766f; --safe: #2f7658;
  --shadow: 0 18px 50px rgba(24,49,43,.10); --radius: 18px;
}
```

The complete stylesheet must also include: border-box sizing; zero body margin; warm canvas; Georgia heading stack; system sans-serif body stack; dark-green sticky header; skip-link; two-state navigation; visible `:focus-visible` outline; centered 1180 px main width; `.eyebrow`; `.upload-layout`; white `.upload-card`; dashed `.drop-zone` with hover, focus-within, and `.is-dragging`; preview image; primary, secondary, and text buttons; progress track animation; desktop `.result-layout` columns; bounded `.image-stage`; absolute ember `.detection-box`; result definition-list cards; guidance; filter pills; history cards with category stripe, thumbnail, actions, and empty state; full-width safety footer; `.visually-hidden`; `[hidden] { display: none !important; }`; `overflow-wrap: anywhere` for filenames; and hover/focus states that never depend on color alone.

At `@media (max-width: 760px)`, stack upload and result layouts, hide the wordmark's `<em>`, allow navigation to wrap, make action buttons full width, and reduce page padding while preserving 44 px touch targets. At `@media (min-width: 761px)`, use the approved two-column result view. Under `@media (prefers-reduced-motion: reduce)`, set scroll behavior to auto and animation/transition duration to `0.01ms` with one iteration.

Use this complete stylesheet:

```css
:root {
  --forest: #18312b; --forest-soft: #285c46; --ember: #d95028;
  --ember-soft: #fbe6db; --canvas: #f4f2eb; --surface: #fffef9;
  --line: #d8d8ce; --muted: #65766f; --safe: #2f7658;
  --shadow: 0 18px 50px rgba(24,49,43,.10); --radius: 18px;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { margin: 0; min-width: 320px; color: var(--forest); background: var(--canvas); font: 16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }
button,input { font: inherit; }
button { min-height: 44px; }
img { display: block; max-width: 100%; }
h1,h2,h3,p { margin-top: 0; }
h1,h2 { font-family: Georgia,"Times New Roman",serif; line-height: 1.08; letter-spacing: -.025em; }
h1 { max-width: 680px; font-size: clamp(2.35rem,7vw,5.3rem); margin-bottom: 1rem; }
h2 { font-size: clamp(1.8rem,4vw,3rem); }
.skip-link { position: fixed; z-index: 100; top: 8px; left: 8px; padding: 10px 14px; color: #fff; background: var(--ember); transform: translateY(-150%); }
.skip-link:focus { transform: translateY(0); }
.site-header { position: sticky; z-index: 20; top: 0; display: flex; align-items: center; gap: 24px; min-height: 72px; padding: 12px clamp(18px,4vw,56px); color: #fff; background: var(--forest); border-bottom: 1px solid rgba(255,255,255,.14); }
.brand { color: #fff; font: 700 1.2rem Georgia,serif; text-decoration: none; white-space: nowrap; }
.brand span { color: #f17c57; }.brand em { color: #b7c9c1; font: inherit; font-weight: 400; }
nav { display: flex; gap: 4px; margin-left: auto; }
.nav-button { padding: 8px 14px; color: #cbd8d2; background: transparent; border: 0; border-radius: 9px; cursor: pointer; }
.nav-button:hover,.nav-button.is-active { color: #fff; background: rgba(255,255,255,.12); }
.count-badge { display: inline-grid; place-items: center; min-width: 22px; height: 22px; margin-left: 4px; padding: 0 6px; color: var(--forest); background: #dbe8e2; border-radius: 999px; font-size: .75rem; }
.ready-status { margin: 0; color: #bce2cf; font-size: .82rem; white-space: nowrap; }.ready-status span { color: #75d6a9; }
main { width: min(1180px,calc(100% - 36px)); min-height: calc(100vh - 250px); margin: 0 auto; padding: clamp(52px,8vw,108px) 0; }
.view { animation: enter .35s ease both; }.eyebrow { margin-bottom: 10px; color: var(--ember); font-size: .75rem; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }
.intro-copy>p:last-child { max-width: 560px; color: var(--muted); font-size: 1.1rem; }
.upload-layout { display: grid; gap: clamp(28px,6vw,72px); align-items: center; }
.upload-card,.result-summary,.progress-panel { padding: clamp(20px,4vw,38px); background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); box-shadow: var(--shadow); }
.drop-zone { display: grid; place-items: center; min-height: 340px; padding: 36px 22px; text-align: center; background: #fff; border: 2px dashed #9aac9f; border-radius: 14px; cursor: pointer; transition: border-color .2s,background .2s,transform .2s; }
.drop-zone:hover,.drop-zone:focus-within,.drop-zone.is-dragging { background: #f5f8f4; border-color: var(--forest-soft); transform: translateY(-2px); }
.drop-zone strong { margin: 14px 0 4px; font: 700 1.2rem Georgia,serif; }.drop-zone span,.drop-zone small { color: var(--muted); }
.upload-icon { display: grid; place-items: center; width: 52px; height: 52px; color: var(--ember)!important; background: var(--ember-soft); border-radius: 14px; font-size: 1.7rem; }
.preview-panel { display: grid; grid-template-columns: 112px minmax(0,1fr); gap: 16px; align-items: center; }
.preview-panel img { width: 112px; height: 88px; object-fit: cover; border-radius: 10px; }.preview-panel div { min-width: 0; }.preview-panel strong,.preview-panel span { display: block; overflow-wrap: anywhere; }.preview-panel span { color: var(--muted); font-size: .85rem; }
.button { display: inline-flex; justify-content: center; align-items: center; gap: 8px; min-height: 46px; padding: 10px 17px; border-radius: 9px; font-weight: 750; cursor: pointer; transition: transform .2s,box-shadow .2s,background .2s; }
.button:hover:not(:disabled) { transform: translateY(-1px); }.button:disabled { opacity: .48; cursor: not-allowed; }
.primary { color: #fff; background: var(--forest); border: 1px solid var(--forest); }.primary:hover { background: var(--forest-soft); box-shadow: 0 8px 22px rgba(24,49,43,.16); }
.secondary { color: var(--forest); background: transparent; border: 1px solid #aeb9b3; }.secondary:hover { background: #eef1ec; }
.text-button { color: #a63d21; background: transparent; border: 0; }.text-button:hover { background: var(--ember-soft); }
:focus-visible { outline: 3px solid #f28a68; outline-offset: 3px; }
.inline-message { margin: 14px 0 0; padding: 11px 13px; color: var(--forest); background: #e5ece7; border-radius: 8px; }.error-message { color: #8e2d18; background: var(--ember-soft); }
.progress-panel { max-width: 720px; margin: 8vh auto; text-align: center; }.progress-track { height: 8px; overflow: hidden; background: #e4e5df; border-radius: 99px; }.progress-track span { display: block; width: 35%; height: 100%; background: var(--ember); border-radius: inherit; animation: scan 1s ease-in-out infinite alternate; }
.result-layout { display: grid; gap: 24px; align-items: start; }.image-stage { position: relative; overflow: hidden; min-height: 280px; background: #dfe3df; border-radius: var(--radius); box-shadow: var(--shadow); }.image-stage>img { width: 100%; height: 100%; max-height: 660px; object-fit: contain; }
.detection-box { position: absolute; border: 3px solid var(--ember); background: rgba(217,80,40,.10); box-shadow: 0 0 0 1px rgba(255,255,255,.8) inset; }.detection-box span { position: absolute; top: -30px; left: -3px; padding: 5px 8px; color: #fff; background: var(--ember); font-size: .72rem; font-weight: 850; letter-spacing: .04em; white-space: nowrap; }
.result-summary { border-top: 5px solid var(--ember); }.result-layout[data-category="clear"] .result-summary { border-top-color: var(--safe); }.result-summary dl { display: grid; grid-template-columns: repeat(3,1fr); gap: 8px; margin: 22px 0; }.result-summary dl div { padding: 12px; background: #f0f1ec; border-radius: 9px; }.result-summary dt { color: var(--muted); font-size: .7rem; text-transform: uppercase; }.result-summary dd { margin: 2px 0 0; font-weight: 800; overflow-wrap: anywhere; }.guidance { margin: 20px 0; padding: 16px; background: var(--ember-soft); border-left: 4px solid var(--ember); border-radius: 8px; }.guidance h3 { margin-bottom: 5px; font-size: 1rem; }.guidance p { margin: 0; color: #5b645f; }
.section-heading { display: flex; justify-content: space-between; gap: 20px; align-items: end; }.section-heading h1 { font-size: clamp(2.4rem,6vw,4.5rem); margin-bottom: 0; }
.filter-group { display: flex; gap: 7px; margin: 28px 0; overflow-x: auto; padding: 3px; }.filter-button { min-height: 40px; padding: 7px 14px; color: var(--forest); background: transparent; border: 1px solid #aeb9b3; border-radius: 999px; cursor: pointer; white-space: nowrap; }.filter-button.is-active { color: #fff; background: var(--forest); border-color: var(--forest); }
.history-list { display: grid; gap: 12px; }.history-card { display: grid; grid-template-columns: 96px minmax(0,1fr) auto auto; gap: 16px; align-items: center; padding: 12px; background: var(--surface); border: 1px solid var(--line); border-left: 5px solid var(--safe); border-radius: 12px; box-shadow: 0 7px 22px rgba(24,49,43,.06); }.history-card.fire,.history-card.smoke { border-left-color: var(--ember); }.history-card img { width: 96px; height: 70px; object-fit: cover; border-radius: 8px; }.history-card h2 { margin: 0 0 4px; font: 700 1rem system-ui,sans-serif; }.history-card p { margin: 0; color: var(--muted); font-size: .82rem; overflow-wrap: anywhere; }.history-actions { display: flex; gap: 5px; }
footer { padding: 28px max(18px,calc((100% - 1180px)/2)); color: #d5e1dc; background: var(--forest); border-top: 4px solid var(--ember); }footer strong { color: #fff; }footer p { max-width: 850px; margin: 5px 0 0; }
.visually-hidden { position: absolute!important; width: 1px!important; height: 1px!important; padding: 0!important; margin: -1px!important; overflow: hidden!important; clip: rect(0,0,0,0)!important; white-space: nowrap!important; border: 0!important; }[hidden] { display: none!important; }
@keyframes scan { from { transform: translateX(-10%); } to { transform: translateX(195%); } }@keyframes enter { from { opacity: 0; transform: translateY(8px); } }
@media (min-width: 761px) { .upload-layout { grid-template-columns: minmax(0,1fr) minmax(380px,.75fr); }.result-layout { grid-template-columns: minmax(0,1.45fr) minmax(300px,.55fr); } }
@media (max-width: 760px) {
  .site-header { flex-wrap: wrap; gap: 8px 12px; padding: 10px 18px; }.brand em,.ready-status { display: none; }nav { order: 3; width: 100%; }.nav-button { flex: 1; }
  main { width: min(100% - 28px,1180px); padding: 46px 0; }h1 { font-size: clamp(2.3rem,13vw,3.6rem); }.drop-zone { min-height: 260px; }
  .preview-panel { grid-template-columns: 80px minmax(0,1fr); }.preview-panel img { width: 80px; height: 70px; }.preview-panel .button { grid-column: 1/-1; width: 100%; }
  .result-summary dl { grid-template-columns: 1fr; }.result-summary>.button { width: 100%; margin-top: 7px; }.section-heading { align-items: start; flex-direction: column; }
  .history-card { grid-template-columns: 72px minmax(0,1fr) auto; }.history-card img { width: 72px; height: 58px; }.history-actions { grid-column: 1/-1; }.history-actions .button { flex: 1; }
}
@media (prefers-reduced-motion: reduce) { html { scroll-behavior: auto; }*,*::before,*::after { animation-duration: .01ms!important; animation-iteration-count: 1!important; transition-duration: .01ms!important; } }
```

- [ ] **Step 4: Run automated design contracts**

Run: `python -m unittest discover -s tests -p "test_*.py" -v`

Expected: 6 tests PASS.

- [ ] **Step 5: Perform browser verification at three widths**

Start: `python app.py`

Open `http://127.0.0.1:5000` and verify at 320×700, 768×1024, and 1440×900:

- No horizontal scrolling.
- File selection and drag state are visible.
- Analyze cannot run before a valid decoded image exists.
- Progress is announced and cannot be started twice.
- Detection box remains aligned to the displayed image.
- Clear results do not show a bounding box or confidence number.
- History filters, reopen, delete, and confirmed clear work after reload.
- Tab order follows navigation, upload, actions, filters, and history.
- Focus is always visible and reduced-motion mode removes nonessential motion.
- The safety notice is present in Analyze and History.
- “Demo result” does not appear anywhere.

- [ ] **Step 6: Commit if Git is available**

```powershell
git add ui/static/css/styles.css ui/tests/test_app.py
git commit -m "feat: style responsive FireWatch interface"
```

### Task 5: UI Documentation and Final Verification

**Files:**
- Create: `ui/README.md`
- Modify: `ui/tests/test_app.py`

**Interfaces:**
- Consumes: completed UI runtime and verification commands.
- Produces: reproducible local setup and a documented future inference seam.

- [ ] **Step 1: Add a failing documentation test**

Add to `ui/tests/test_app.py`:

```python
    def test_readme_documents_setup_limits_and_future_integration(self):
        readme = (UI_DIR / "README.md").read_text(encoding="utf-8")
        for text in ("python -m pip install -r requirements.txt", "python app.py", "UI prototype", "does not run YOLO", "simulateResult"):
            self.assertIn(text, readme)

    def test_declared_static_assets_resolve(self):
        for asset in ("/static/css/styles.css", "/static/js/core.js", "/static/js/app.js"):
            with self.subTest(asset=asset):
                self.assertEqual(self.client.get(asset).status_code, 200)
```

- [ ] **Step 2: Run the documentation test to verify it fails**

Run: `python -m unittest tests.test_app.AppTests.test_readme_documents_setup_limits_and_future_integration -v`

Expected: FAIL because `README.md` does not exist.

- [ ] **Step 3: Write the UI README**

Create `ui/README.md` with these sections and exact commands:

````markdown
# FireWatch AI UI Prototype

This directory contains a responsive UI prototype. It simulates results in the browser and does not run YOLO or upload selected images to Flask.

## Setup

```powershell
cd ui
python -m pip install -r requirements.txt
```

## Run

```powershell
python app.py
```

Open `http://127.0.0.1:5000`.

## Test

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
node --test tests/core.test.js
node --check static/js/app.js
```

## Prototype limitations

The outcome and confidence are deterministic simulated values for interface evaluation. This tool does not replace smoke alarms, official alerts, trained firefighters, or calling 911.

## Future model integration

Replace the `simulateResult` call in `static/js/app.js` with a request to a future inference endpoint that returns the same normalized result shape. Keep validation, result rendering, history, and accessibility behavior unchanged.
````

- [ ] **Step 4: Run the complete automated suite**

Run from `ui`:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
node --test tests/core.test.js
node --check static/js/core.js
node --check static/js/app.js
```

Expected: 8 Python tests PASS, 5 Node tests PASS, and both syntax checks exit 0.

- [ ] **Step 5: Confirm scope containment**

Run from the project root:

```powershell
Get-ChildItem ui -Recurse -File | Select-Object -ExpandProperty FullName
```

Expected: only the nine planned files plus generated `__pycache__` files, if any. Compare root files by timestamp/status when Git remains unavailable and confirm no pre-existing source file was modified.

- [ ] **Step 6: Repeat the acceptance smoke test**

Run `python app.py`, upload one supported image, complete analysis, open History, reload, reopen the saved result, delete it, then verify the empty state. Repeat once with an unsupported file and once at a 320-pixel viewport.

Expected: all flows behave as specified, no console errors occur, the safety notice remains visible, and no real-model language or “Demo result” label appears.

- [ ] **Step 7: Commit if Git is available**

```powershell
git add ui/README.md ui/tests/test_app.py
git commit -m "docs: document FireWatch UI prototype"
```
