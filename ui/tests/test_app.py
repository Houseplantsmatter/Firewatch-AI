import importlib.util
import re
import sys
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from werkzeug.datastructures import MultiDict

UI_DIR = Path(__file__).resolve().parents[1]
if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))
SPEC = importlib.util.spec_from_file_location("firewatch_ui_app", UI_DIR / "app.py")
APP_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = APP_MODULE
SPEC.loader.exec_module(APP_MODULE)

from inference import InferenceFailure, ModelUnavailableError


def image_bytes(image_format="JPEG", size=(32, 24)):
    buffer = BytesIO()
    Image.new("RGB", size, (210, 80, 30)).save(buffer, format=image_format)
    return buffer.getvalue()


class FakeInferenceService:
    def __init__(self, prediction=None, failure=None):
        self.prediction = prediction or {
            "category": "fire",
            "confidence": 0.91,
            "detectionCount": 2,
            "box": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
        }
        self.failure = failure
        self.received = []

    def predict(self, payload):
        self.received.append(payload)
        if self.failure:
            raise self.failure
        return self.prediction


class FalseyInferenceService(FakeInferenceService):
    def __bool__(self):
        return False


class FakeHugeImage:
    format = "JPEG"
    size = (10_000, 4_001)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def verify(self):
        return None


def contrast_ratio(foreground, background):
    def luminance(hex_color):
        channels = [int(hex_color[index:index + 2], 16) / 255 for index in (1, 3, 5)]
        linear = [value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4 for value in channels]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    lighter, darker = sorted((luminance(foreground), luminance(background)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


class AppTests(unittest.TestCase):
    def setUp(self):
        self.app = APP_MODULE.create_app({"TESTING": True})
        self.client = self.app.test_client()

    def test_analyze_returns_normalized_model_result(self):
        service = FakeInferenceService()
        app = APP_MODULE.create_app(
            {"TESTING": True}, inference_service=service
        )
        client = app.test_client()
        payload = image_bytes()

        response = client.post(
            "/api/analyze",
            data={"image": (BytesIO(payload), "../../scene.jpg", "image/jpeg")},
        )

        self.assertEqual(response.status_code, 200)
        result = response.get_json()["result"]
        self.assertEqual(
            {
                key: result[key]
                for key in ("filename", "category", "confidence", "detectionCount", "box")
            },
            {
                "filename": "scene.jpg",
                "category": "fire",
                "confidence": 0.91,
                "detectionCount": 2,
                "box": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
            },
        )
        self.assertRegex(
            result["timestamp"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$"
        )
        self.assertTrue(result["id"].startswith(result["timestamp"] + "-"))
        self.assertRegex(result["id"], re.escape(result["timestamp"]) + r"-[0-9a-f]{16}$")
        self.assertEqual(service.received, [payload])
        response.close()

    def test_analyze_accepts_supported_formats_and_the_exact_byte_limit(self):
        service = FakeInferenceService()
        client = APP_MODULE.create_app(
            {"TESTING": True}, inference_service=service
        ).test_client()
        formats = (
            ("JPEG", "image/jpeg", "scene.jpg"),
            ("PNG", "image/png", "scene.png"),
            ("WEBP", "image/webp", "scene.webp"),
        )

        for image_format, mimetype, filename in formats:
            with self.subTest(image_format=image_format):
                payload = image_bytes(image_format)
                with patch.object(APP_MODULE, "MAX_IMAGE_BYTES", len(payload)):
                    response = client.post(
                        "/api/analyze",
                        data={
                            "image": (
                                BytesIO(payload),
                                filename,
                                mimetype,
                            )
                        },
                    )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.get_json()["result"]["filename"], filename)
                response.close()
        self.assertEqual(len(service.received), 3)

    def test_falsey_injected_service_is_still_used(self):
        service = FalseyInferenceService()
        client = APP_MODULE.create_app(
            {"TESTING": True}, inference_service=service
        ).test_client()
        response = client.post(
            "/api/analyze",
            data={
                "image": (
                    BytesIO(image_bytes()),
                    "scene.jpg",
                    "image/jpeg",
                )
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(service.received), 1)
        response.close()

    def test_analyze_rejects_missing_blank_and_empty_uploads(self):
        service = FakeInferenceService()
        client = APP_MODULE.create_app(
            {"TESTING": True}, inference_service=service
        ).test_client()
        requests = (
            {},
            {"image": (BytesIO(image_bytes()), "", "image/jpeg")},
            {"image": (BytesIO(b""), "empty.jpg", "image/jpeg")},
        )

        for data in requests:
            with self.subTest(data=data):
                response = client.post("/api/analyze", data=data)
                self.assertEqual(response.status_code, 400)
                self.assertIsInstance(response.get_json()["error"], str)
                response.close()
        self.assertEqual(service.received, [])

    def test_analyze_rejects_oversized_unsupported_and_invalid_images(self):
        service = FakeInferenceService()
        client = APP_MODULE.create_app(
            {"TESTING": True}, inference_service=service
        ).test_client()
        with patch.object(APP_MODULE, "MAX_IMAGE_BYTES", 8):
            response = client.post(
                "/api/analyze",
                data={
                    "image": (
                        BytesIO(b"123456789"),
                        "large.jpg",
                        "image/jpeg",
                    )
                },
            )
        self.assertEqual(response.status_code, 413)
        response.close()

        requests = (
            (
                {"image": (BytesIO(image_bytes()), "scene.gif", "image/gif")},
                400,
            ),
            (
                {"image": (BytesIO(b"not an image"), "scene.jpg", "image/jpeg")},
                400,
            ),
            (
                {"image": (BytesIO(image_bytes()), "scene.png", "image/png")},
                400,
            ),
        )

        for data, status in requests:
            with self.subTest(status=status, filename=data["image"][1]):
                response = client.post("/api/analyze", data=data)
                self.assertEqual(response.status_code, status)
                self.assertIsInstance(response.get_json()["error"], str)
                response.close()
        self.assertEqual(service.received, [])

    def test_request_size_limit_returns_json_before_multipart_processing(self):
        service = FakeInferenceService()
        app = APP_MODULE.create_app(
            {"TESTING": True, "MAX_CONTENT_LENGTH": 1024},
            inference_service=service,
        )
        self.assertGreater(
            APP_MODULE.create_app(
                {"TESTING": True}, inference_service=service
            ).config["MAX_CONTENT_LENGTH"],
            APP_MODULE.MAX_IMAGE_BYTES,
        )
        client = app.test_client()
        response = client.post(
            "/api/analyze",
            data={
                "image": (
                    BytesIO(b"x" * 2048),
                    "large.jpg",
                    "image/jpeg",
                )
            },
        )
        self.assertEqual(response.status_code, 413)
        self.assertIn("10 MB", response.get_json()["error"])
        self.assertEqual(service.received, [])
        response.close()

    def test_analyze_rejects_duplicate_or_extra_multipart_fields(self):
        service = FakeInferenceService()
        client = APP_MODULE.create_app(
            {"TESTING": True}, inference_service=service
        ).test_client()
        duplicate_images = MultiDict(
            [
                ("image", (BytesIO(image_bytes()), "one.jpg", "image/jpeg")),
                ("image", (BytesIO(image_bytes()), "two.jpg", "image/jpeg")),
            ]
        )
        requests = (
            duplicate_images,
            {
                "image": (BytesIO(image_bytes()), "one.jpg", "image/jpeg"),
                "extra": (BytesIO(image_bytes()), "two.jpg", "image/jpeg"),
            },
            {
                "image": (BytesIO(image_bytes()), "one.jpg", "image/jpeg"),
                "note": "extra form field",
            },
        )
        for data in requests:
            with self.subTest(keys=list(data.keys())):
                response = client.post("/api/analyze", data=data)
                self.assertEqual(response.status_code, 400)
                self.assertIsInstance(response.get_json()["error"], str)
                response.close()
        self.assertEqual(service.received, [])

    def test_analyze_rejects_images_above_the_decoded_pixel_limit(self):
        service = FakeInferenceService()
        client = APP_MODULE.create_app(
            {"TESTING": True}, inference_service=service
        ).test_client()
        with patch("PIL.Image.open", return_value=FakeHugeImage()):
            response = client.post(
                "/api/analyze",
                data={
                    "image": (
                        BytesIO(image_bytes()),
                        "huge.jpg",
                        "image/jpeg",
                    )
                },
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("large", response.get_json()["error"].lower())
        self.assertEqual(service.received, [])
        response.close()

    def test_analyze_maps_model_and_inference_failures_to_safe_json(self):
        failures = (
            (ModelUnavailableError("private model path"), 503),
            (InferenceFailure("private tensor detail"), 500),
            (RuntimeError("private internal detail"), 500),
        )
        generic_inference_message = None

        for failure, expected_status in failures:
            with self.subTest(failure=type(failure).__name__):
                app = APP_MODULE.create_app(
                    {"TESTING": True},
                    inference_service=FakeInferenceService(failure=failure),
                )
                client = app.test_client()
                with self.assertLogs(app.logger.name, level="ERROR") as captured:
                    response = client.post(
                        "/api/analyze",
                        data={
                            "image": (
                                BytesIO(image_bytes()),
                                "../../scene.jpg",
                                "image/jpeg",
                            )
                        },
                    )
                self.assertEqual(response.status_code, expected_status)
                message = response.get_json()["error"]
                self.assertNotIn("private", message)
                self.assertIn("private", "\n".join(captured.output))
                if expected_status == 500:
                    if generic_inference_message is None:
                        generic_inference_message = message
                    self.assertEqual(message, generic_inference_message)
                response.close()

    def test_index_serves_firewatch_shell(self):
        response = self.client.get("/")
        body = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("FireWatch AI", body)
        self.assertIn('rel="icon" href="data:image/svg+xml,', body)
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

    def test_controller_uses_clean_result_symbols_and_makes_result_title_focusable(self):
        app_js = (UI_DIR / "static" / "js" / "app.js").read_text(encoding="utf-8")
        self.assertIn(r'result.confidence === null ? "\u2014"', app_js)
        self.assertIn(r'result.category.toUpperCase()} \u00b7 ${Math.round', app_js)
        self.assertIn(r'toLocaleString()} \u00b7 ${result.filename}', app_js)
        self.assertNotIn("\u00e2\u20ac\u201d", app_js)
        self.assertNotIn("\u00c2\u00b7", app_js)
        render_result = app_js[app_js.index("function renderResult"):app_js.index("function loadHistory")]
        self.assertIn("nodes.resultTitle.tabIndex = -1;", render_result)

    def test_styles_include_responsive_focus_and_reduced_motion_contracts(self):
        css = (UI_DIR / "static" / "css" / "styles.css").read_text(encoding="utf-8")
        for required in (
            "--forest: #18312b", "--ember: #d95028", "@media (max-width: 760px)",
            "@media (prefers-reduced-motion: reduce)", ":focus-visible", ".result-layout",
            ".history-list", ".detection-box", "overflow-wrap: anywhere",
        ):
            self.assertIn(required, css)

    def test_styles_keep_detection_overlays_aligned_and_controls_accessible(self):
        css = (UI_DIR / "static" / "css" / "styles.css").read_text(encoding="utf-8")
        for required in (
            ".result-visual { text-align: center; }",
            ".image-stage { position: relative; display: inline-block; max-width: 100%;",
            "min-height: 0;",
            ".image-stage>img { width: auto; height: auto; max-width: 100%;",
            ":focus-visible { outline: 3px solid #8e2d18;",
            ".filter-button { min-height: 44px;",
        ):
            self.assertIn(required, css)

    def test_styles_do_not_force_body_wider_than_mobile_client_width(self):
        css = (UI_DIR / "static" / "css" / "styles.css").read_text(encoding="utf-8")
        self.assertIn("body { margin: 0; min-width: 0;", css)
        self.assertNotIn("body { margin: 0; min-width: 320px;", css)

    def test_upload_focus_is_exposed_on_the_adjacent_drop_zone(self):
        css = (UI_DIR / "static" / "css" / "styles.css").read_text(encoding="utf-8")
        app_js = (UI_DIR / "static" / "js" / "app.js").read_text(encoding="utf-8")
        self.assertIn("#file-input:focus-visible + .drop-zone", css)
        reset = app_js[app_js.index("function resetAnalyze"):app_js.index("function copyFor")]
        self.assertIn("nodes.fileInput.focus();", reset)

    def test_exact_text_and_focus_color_pairs_meet_wcag_contrast(self):
        css = (UI_DIR / "static" / "css" / "styles.css").read_text(encoding="utf-8")
        for required in (
            "--ember-text: #a63d21", "--muted: #566760", ".eyebrow { margin-bottom: 10px; color: var(--ember-text);",
            ".detection-box span {", "color: #fff; background: #8e2d18;",
            ".site-header :focus-visible { outline-color: #f7c9b7;",
        ):
            self.assertIn(required, css)
        for foreground, background, minimum in (
            ("#a63d21", "#f4f2eb", 4.5),
            ("#a63d21", "#fffef9", 4.5),
            ("#ffffff", "#8e2d18", 4.5),
            ("#f7c9b7", "#18312b", 3.0),
            ("#566760", "#f4f2eb", 4.5),
            ("#566760", "#fffef9", 4.5),
            ("#566760", "#f0f1ec", 4.5),
        ):
            with self.subTest(foreground=foreground, background=background):
                self.assertGreaterEqual(contrast_ratio(foreground, background), minimum)

    def test_brand_routes_to_analyze_through_controller_navigation(self):
        body = self.client.get("/").get_data(as_text=True)
        app_js = (UI_DIR / "static" / "js" / "app.js").read_text(encoding="utf-8")
        self.assertIn('class="brand" href="#analyze" data-view="analyze"', body)
        navigation = app_js[app_js.index('document.querySelectorAll("[data-view]")'):app_js.index("nodes.filters.addEventListener")]
        self.assertIn("event.preventDefault();", navigation)
        self.assertIn("setView(control.dataset.view)", navigation)

    def test_result_has_non_blocking_storage_status_and_save_failure_announcement(self):
        body = self.client.get("/").get_data(as_text=True)
        app_js = (UI_DIR / "static" / "js" / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="result-storage-message"', body)
        self.assertIn('role="status"', body)
        self.assertIn('resultStorageMessage: byId("result-storage-message")', app_js)
        self.assertIn("History could not be saved", app_js)
        analyze = app_js[app_js.index("async function analyzeCurrentFile"):app_js.index("nodes.fileInput.addEventListener")]
        self.assertIn("storageMessage", analyze)
        self.assertIn("announce(", analyze)

    def test_browser_submits_real_inference_and_restores_preview_after_failure(self):
        app_js = (UI_DIR / "static" / "js" / "app.js").read_text(encoding="utf-8")
        analyze = app_js[
            app_js.index("async function analyzeCurrentFile"):
            app_js.index("nodes.fileInput.addEventListener")
        ]
        for required in (
            "new FormData()",
            'formData.append("image", state.file)',
            'fetch("/api/analyze", {method: "POST", body: formData})',
            "core.validateServerResult",
            "thumbnail: state.thumbnail",
            "catch",
            "nodes.uploadPanel.hidden = false",
            "nodes.previewPanel.hidden = false",
            "nodes.progressPanel.hidden = true",
            "nodes.analyzeButton.disabled = false",
            "setUploadError(",
            "nodes.analyzeButton.focus()",
        ):
            self.assertIn(required, analyze)
        self.assertNotIn("simulateResult", app_js)
        self.assertNotIn("setTimeout(resolve, 900)", app_js)
        self.assertNotIn("simulated fire or smoke", app_js)
        self.assertIn("model's detection threshold", app_js)

    def test_history_candidate_is_committed_only_after_storage_save_succeeds(self):
        app_js = (UI_DIR / "static" / "js" / "app.js").read_text(encoding="utf-8")
        save_history = app_js[app_js.index("function saveHistory"):app_js.index("function historyCard")]
        self.assertIn("const candidateHistory", save_history)
        self.assertIn("localStorage.setItem(core.STORAGE_KEY, JSON.stringify(candidateHistory))", save_history)
        self.assertIn("state.history = candidateHistory", save_history)
        candidate = save_history.index("const candidateHistory")
        persisted = save_history.index("localStorage.setItem(core.STORAGE_KEY, JSON.stringify(candidateHistory))")
        committed = save_history.index("state.history = candidateHistory")
        self.assertLess(candidate, persisted)
        self.assertLess(persisted, committed)

    def test_corrupt_json_is_reset_without_marking_storage_unavailable(self):
        app_js = (UI_DIR / "static" / "js" / "app.js").read_text(encoding="utf-8")
        load_history = app_js[app_js.index("function loadHistory"):app_js.index("function saveHistory")]
        self.assertIn("JSON.parse", load_history)
        self.assertIn("localStorage.removeItem(core.STORAGE_KEY)", load_history)
        self.assertIn("state.storageAvailable = true", load_history)

    def test_filters_wrap_and_keep_aria_pressed_synchronized(self):
        body = self.client.get("/").get_data(as_text=True)
        css = (UI_DIR / "static" / "css" / "styles.css").read_text(encoding="utf-8")
        app_js = (UI_DIR / "static" / "js" / "app.js").read_text(encoding="utf-8")
        self.assertIn('data-filter="all" type="button" aria-pressed="true"', body)
        for category in ("fire", "smoke", "clear"):
            self.assertIn('data-filter="{}" type="button" aria-pressed="false"'.format(category), body)
        filter_rule = css[css.index(".filter-group {"):css.index(".filter-button {")]
        self.assertIn("flex-wrap: wrap", filter_rule)
        self.assertNotIn("overflow-x: auto", filter_rule)
        self.assertIn('button.setAttribute("aria-pressed", String(active))', app_js)

    def test_result_and_history_render_visible_category_icons(self):
        body = self.client.get("/").get_data(as_text=True)
        app_js = (UI_DIR / "static" / "js" / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="result-icon" class="category-icon" aria-hidden="true"', body)
        self.assertIn("function categoryIcon(category)", app_js)
        self.assertIn("nodes.resultIcon.textContent = categoryIcon(result.category);", app_js)
        history_card = app_js[app_js.index("function historyCard"):app_js.index("function renderHistory")]
        self.assertIn('icon.className = "category-icon"', history_card)
        self.assertIn("icon.textContent = categoryIcon(result.category)", history_card)

    def test_delete_moves_focus_to_a_remaining_action_or_empty_message(self):
        app_js = (UI_DIR / "static" / "js" / "app.js").read_text(encoding="utf-8")
        delete_item = app_js[app_js.index("function deleteHistoryItem"):app_js.index("function clearHistory")]
        self.assertIn('nodes.historyList.querySelector("[data-open-id], [data-delete-id]")', delete_item)
        self.assertIn("remainingAction.focus()", delete_item)
        self.assertIn("nodes.historyMessage.focus()", delete_item)

    def test_delete_storage_failure_keeps_candidate_uncommitted_and_announces_failure(self):
        app_js = (UI_DIR / "static" / "js" / "app.js").read_text(encoding="utf-8")
        delete_item = app_js[app_js.index("function deleteHistoryItem"):app_js.index("function clearHistory")]
        self.assertIn("const candidateHistory", delete_item)
        self.assertIn("core.commitHistoryMutation", delete_item)
        self.assertIn("if (!outcome.saved)", delete_item)
        self.assertIn("state.storageAvailable = false", delete_item)
        self.assertIn("History item could not be deleted", delete_item)
        self.assertIn("announce(failureMessage)", delete_item)
        failure_branch = delete_item[delete_item.index("if (!outcome.saved)"):delete_item.index("state.history = outcome.history")]
        self.assertIn("nodes.historyMessage.focus()", failure_branch)
        self.assertLess(failure_branch.index("renderHistory()"), failure_branch.index("nodes.historyMessage.focus()"))
        self.assertLess(failure_branch.index("nodes.historyMessage.focus()"), failure_branch.index("announce(failureMessage)"))
        success_branch = delete_item[delete_item.index("state.history = outcome.history"):]
        self.assertIn("state.storageAvailable = true", success_branch)
        self.assertIn('announce("History item deleted.")', success_branch)

    def test_clear_storage_failure_is_transactional_and_success_focuses_empty_message(self):
        app_js = (UI_DIR / "static" / "js" / "app.js").read_text(encoding="utf-8")
        clear_item = app_js[app_js.index("function clearHistory"):app_js.index("async function analyzeCurrentFile")]
        self.assertIn("const candidateHistory = []", clear_item)
        self.assertIn("core.commitHistoryMutation", clear_item)
        self.assertIn("if (!outcome.saved)", clear_item)
        self.assertIn("History could not be cleared", clear_item)
        self.assertIn("announce(failureMessage)", clear_item)
        success_branch = clear_item[clear_item.index("state.history = outcome.history"):]
        self.assertIn("state.storageAvailable = true", success_branch)
        self.assertIn("nodes.historyMessage.focus()", success_branch)
        self.assertIn('announce("Detection history cleared.")', success_branch)

    def test_opening_saved_result_clears_stale_save_warning(self):
        app_js = (UI_DIR / "static" / "js" / "app.js").read_text(encoding="utf-8")
        open_item = app_js[app_js.index("function openHistoryItem"):app_js.index("function deleteHistoryItem")]
        self.assertIn("clearResultStorageMessage()", open_item)
        self.assertLess(open_item.index("clearResultStorageMessage()"), open_item.index("renderResult("))

    def test_history_actions_include_contextual_filename_labels(self):
        app_js = (UI_DIR / "static" / "js" / "app.js").read_text(encoding="utf-8")
        history_card = app_js[app_js.index("function historyCard"):app_js.index("function renderHistory")]
        self.assertIn('open.setAttribute("aria-label", `Open ${result.filename}`)', history_card)
        self.assertIn('remove.setAttribute("aria-label", `Delete ${result.filename}`)', history_card)

    def test_history_failure_message_is_a_non_blocking_status(self):
        body = self.client.get("/").get_data(as_text=True)
        self.assertIn('id="history-message" class="inline-message" role="status"', body)

    def test_image_selection_generation_rejects_and_revokes_stale_decodes(self):
        app_js = (UI_DIR / "static" / "js" / "app.js").read_text(encoding="utf-8")
        select_file = app_js[app_js.index("async function selectFile"):app_js.index("function resetAnalyze")]
        self.assertIn("const generation = ++state.selectionGeneration", select_file)
        self.assertIn("generation !== state.selectionGeneration", select_file)
        self.assertIn("URL.revokeObjectURL(decoded.url)", select_file)
        reset = app_js[app_js.index("function resetAnalyze"):app_js.index("function copyFor")]
        self.assertIn("state.selectionGeneration += 1", reset)

    def test_readme_documents_real_model_setup_and_limits(self):
        readme = (UI_DIR / "README.md").read_text(encoding="utf-8")
        requirements = (UI_DIR / "requirements.txt").read_text(encoding="utf-8")
        for text in (
            "python -m pip install -r requirements.txt",
            "python app.py",
            "http://127.0.0.1:5000",
            "runs/detect/fire_smoke_v2/weights/best.pt",
            "real YOLO inference",
            "first analysis request",
            "GPU",
            "CPU",
            "10 MB",
            "40,000,000 pixels",
            'python -m unittest discover -s tests -p "test_*.py" -v',
            "node --test tests/core.test.js",
            "node --check static/js/core.js",
            "python -m compileall -q .",
            "InferenceService",
            "AoF00001.jpg",
        ):
            self.assertIn(text, readme)
        for obsolete in ("UI prototype", "does not run YOLO", "simulateResult"):
            self.assertNotIn(obsolete, readme)
        self.assertIn("Pillow", requirements)
        self.assertIn("ultralytics", requirements)

    def test_declared_static_assets_resolve(self):
        for asset in ("/static/css/styles.css", "/static/js/core.js", "/static/js/app.js"):
            with self.subTest(asset=asset):
                response = self.client.get(asset)
                self.assertEqual(response.status_code, 200)
                response.close()


if __name__ == "__main__":
    unittest.main()
