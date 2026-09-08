# FireWatch AI Responsive UI Prototype Design

## Goal

Create a polished, responsive web-interface prototype for the FireWatch AI fire
and smoke detection project. The application must be easy for homeowners,
schools, businesses, park workers, security staff, and other non-technical users
to understand. It will demonstrate the intended image-analysis workflow while
the YOLO model is still under development.

The prototype will not load a model or perform real inference. It will generate
simulated but deterministic results in the browser. The result itself will not
carry a “Demo result” label, per the product owner's direction. A persistent
safety notice will still state that FireWatch AI is an additional safety tool
and does not replace working smoke alarms, official emergency alerts, trained
firefighters, or calling 911 during an emergency.

## Scope and Constraints

All new runtime code, templates, static assets, tests, and setup instructions
will live inside the existing `ui` directory. No existing training, detection,
configuration, data, weight, or test file will be modified.

The prototype includes:

- A single-image drag-and-drop and file-picker workflow.
- Image preview and an explicit analysis action.
- Simulated fire, smoke, and no-concern outcomes.
- Bounding boxes, confidence scores, timestamps, and next-step guidance.
- A browser-local, filterable history of the 12 newest results.
- Responsive phone, tablet, and desktop layouts.
- Accessible navigation, controls, status messages, and error states.

Live cameras, real YOLO inference, user accounts, cloud persistence,
notifications, emergency-service integration, and multi-image batch analysis are
outside this prototype's scope.

## Architecture

The prototype will use a small Flask server with vanilla HTML, CSS, and
JavaScript. Flask will serve the application shell and static assets. It will not
accept uploaded image data or expose a mock-inference API; the selected image,
simulation, and history stay in the browser.

The runtime units are:

- `app.py`: creates the Flask application, serves the main page, and provides a
  development entry point.
- `templates/index.html`: defines semantic page structure and loads local
  assets.
- `static/css/styles.css`: contains the Calm & Civic design system, responsive
  layout, component states, and accessibility preferences.
- `static/js/app.js`: owns upload validation, previews, deterministic simulated
  analysis, result rendering, navigation, history storage, filters, deletion,
  and announcements.
- `tests/`: verifies server delivery and browser-independent behavior that can
  be tested without loading the unfinished model.
- `README.md` and a UI-local dependency file: explain installation, startup,
  prototype limitations, and verification commands.

Keeping the simulation behind a small JavaScript function gives the future real
inference integration a clear seam: the UI can later replace that function with
an HTTP request while retaining the same normalized result shape.

## Information Architecture and User Flow

The product is a single-page dashboard with two primary views: **Analyze** and
**History**.

1. The header shows the FireWatch AI identity, the two navigation choices, and
   a ready indicator.
2. Analyze opens with a short purpose statement and a large upload surface.
3. The user drops an image or selects one with the file picker.
4. A preview appears with filename, file size, Replace, and Analyze image
   controls.
5. Analyze image starts a brief staged progress treatment, with status text
   announced to assistive technology.
6. The result view presents the annotated image first, followed by the outcome,
   highest confidence, detection count, timestamp, and practical next steps.
7. Analyze another image resets the working state. View history opens the saved
   results.
8. History lists the newest entries first. The user can filter All, Fire, Smoke,
   or No concern; reopen an entry; delete one entry; or clear the full history
   after confirmation.

The persistent footer safety notice remains visible in both primary views.

## Visual System

The approved **Calm & Civic** direction uses warm off-white page surfaces,
forest-green structure and primary actions, and restrained ember orange for
attention and detected concerns. The interface should feel trustworthy and
reassuring before analysis, but unambiguous when a possible hazard appears.

Large editorial headings use a readable serif face, while controls, labels, and
body copy use a highly legible sans-serif system stack. Surfaces use modest
corner rounding, thin neutral borders, and soft shadows. Fire and smoke states
must use text and icons in addition to color. Decorative effects must not
compete with the uploaded image or status message.

On desktop, the result view uses a wide image panel beside a narrower summary
panel. The upload state uses a centered, generous workspace. On tablets and
phones, panels stack into one column, navigation remains compact, and important
actions expand to comfortable touch widths. No control should require
horizontal scrolling at a 320-pixel viewport.

## Prototype Data and Simulation

Accepted uploads are JPG, PNG, and WebP images no larger than 10 MB. Validation
checks both the browser-provided MIME type and the filename extension, then
confirms that the file can be decoded as an image before enabling analysis.
reco
The simulator derives a stable numeric seed from non-sensitive file properties
and uses it to select fire, smoke, or no concern. The same file therefore gives
the same outcome on repeat analysis in the same browser implementation.
Detected outcomes receive one deterministic normalized bounding box and a
confidence value in a credible display range. No-concern outcomes have no box
and do not imply that the real scene is safe.

The UI works with a normalized result object containing an ID, filename,
timestamp, category, confidence, detection count, optional bounding-box
coordinates, and thumbnail. Rendering reads this object rather than simulator
internals, which preserves the future backend integration boundary.

Before storing a result, the browser produces a small, compressed thumbnail so
full-resolution uploads are not placed in local storage. Only the 12 newest
items are retained. If storage is unavailable or full, the current result still
works and the user receives a non-blocking explanation that history could not be
saved.

## States and Error Handling

Analyze supports empty, drag-active, file-selected, analyzing, concern-result,
no-concern-result, and inline-error states. History supports populated,
filtered-empty, globally empty, and storage-unavailable states.

Errors use concise, actionable language and remain near the relevant control:

- Unsupported type: ask for JPG, PNG, or WebP.
- File over 10 MB: state the limit and ask for a smaller image.
- Decode failure: explain that the image could not be read and allow immediate
  replacement.
- Storage failure: keep the current result but explain that it was not added to
  history.

A new valid selection clears the prior upload error. Repeated Analyze clicks are
disabled while progress is active. Deleting all history requires explicit
confirmation. JavaScript failures must not hide the page's purpose or safety
notice.

## Accessibility

The experience will use semantic landmarks, native buttons and inputs, an
associated file-upload label, visible keyboard focus, and a logical source
order. Status changes will use a polite live region, while urgent result text is
presented prominently without relying on animation.

Color pairs must meet WCAG AA contrast for their text sizes. Icons and color
will always be paired with words. The drag-and-drop target must also work via
keyboard file selection. Motion will be brief and functional, and reduced-motion
preferences will remove nonessential transitions.

## Verification

Automated checks will verify:

- The Flask root route returns the UI and required local assets resolve.
- The page exposes its two primary views and safety notice.
- Runtime files stay under `ui` and no model import is introduced.
- JavaScript syntax is valid.
- Deterministic simulation returns a stable normalized result for the same
  input and valid values for all three categories.

Manual browser verification will cover:

- Drag-and-drop and keyboard file selection.
- Valid, invalid, oversized, and unreadable upload paths.
- Upload, analyzing, result, history, filtering, reopen, delete, and clear flows.
- Layouts at 320-pixel phone, common tablet, and wide desktop widths.
- Keyboard-only navigation, visible focus, live announcements, and reduced
  motion.
- Persistence across reload and graceful behavior when storage is unavailable.

## Acceptance Criteria

- The app starts from instructions contained in `ui` and serves through Python.
- It is a functional UI prototype with no YOLO/model dependency or real
  detection claim.
- Users can upload an image, run a simulated analysis, understand the result,
  and revisit recent local history.
- The approved Calm & Civic design remains clear and responsive from 320-pixel
  phones through desktop screens.
- Result states include bounding boxes, category, confidence, time, and next
  steps where applicable.
- The requested “Demo result” label is absent.
- The safety notice remains present throughout the experience.
- New runtime and test files are contained within `ui`, and existing project
  source files remain unchanged.
