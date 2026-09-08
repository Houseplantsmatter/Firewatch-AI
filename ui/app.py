from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from secrets import token_hex

from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from inference import InferenceFailure, InferenceService, ModelUnavailableError


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_DIR / "runs" / "detect" / "fire_smoke_v2" / "weights" / "best.pt"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
IMAGE_FORMATS = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}
GENERIC_INFERENCE_ERROR = "The image could not be analyzed. Please try again."


class UploadValidationError(ValueError):
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code


def _validated_upload(upload):
    if upload is None:
        raise UploadValidationError("Choose an image to analyze.")
    try:
        if not upload.filename:
            raise UploadValidationError("Choose an image to analyze.")
        filename = secure_filename(upload.filename)
        if not filename:
            raise UploadValidationError("Choose an image with a valid filename.")
        expected_format = IMAGE_FORMATS.get(upload.mimetype)
        if expected_format is None:
            raise UploadValidationError("Choose a JPG, PNG, or WebP image.")

        payload = upload.stream.read(MAX_IMAGE_BYTES + 1)
        if not payload:
            raise UploadValidationError(
                "This image is empty. Choose a different file."
            )
        if len(payload) > MAX_IMAGE_BYTES:
            raise UploadValidationError(
                "This image is larger than 10 MB. Choose a smaller file.", 413
            )

        try:
            with Image.open(BytesIO(payload)) as image:
                if image.format != expected_format:
                    raise UploadValidationError(
                        "The uploaded content does not match its image type."
                    )
                width, height = image.size
                if width <= 0 or height <= 0:
                    raise UploadValidationError(
                        "This image has invalid dimensions. Choose a different file."
                    )
                if width * height > MAX_IMAGE_PIXELS:
                    raise UploadValidationError(
                        "This image is too large to analyze. Choose a smaller image."
                    )
                image.verify()
        except UploadValidationError:
            raise
        except (
            UnidentifiedImageError,
            OSError,
            SyntaxError,
            ValueError,
            Image.DecompressionBombError,
        ) as exc:
            raise UploadValidationError(
                "This image could not be read. Choose a different file."
            ) from exc
        return filename, payload
    finally:
        upload.close()


def create_app(test_config=None, inference_service=None):
    app = Flask(__name__)
    app.config.from_mapping(
        MODEL_PATH=DEFAULT_MODEL_PATH,
        MAX_CONTENT_LENGTH=MAX_IMAGE_BYTES + 64 * 1024,
    )
    if test_config:
        app.config.update(test_config)
    service = (
        inference_service
        if inference_service is not None
        else InferenceService(Path(app.config["MODEL_PATH"]))
    )

    @app.errorhandler(RequestEntityTooLarge)
    def request_too_large(_error):
        return (
            jsonify(error="This image is larger than 10 MB. Choose a smaller file."),
            413,
        )

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/manifest.json")
    def manifest():
        return send_from_directory(app.static_folder.parent, "manifest.json")

    @app.get("/sw.js")
    def service_worker():
        return send_from_directory(app.root_path, "sw.js")

    @app.post("/api/analyze")
    def analyze():
        try:
            if (
                request.form
                or set(request.files.keys()) != {"image"}
                or len(request.files.getlist("image")) != 1
            ):
                raise UploadValidationError(
                    "Send exactly one image in the image field."
                )
            filename, payload = _validated_upload(request.files.get("image"))
            prediction = service.predict(payload)
        except UploadValidationError as exc:
            return jsonify(error=str(exc)), exc.status_code
        except ModelUnavailableError:
            app.logger.exception("The FireWatch inference model is unavailable.")
            return (
                jsonify(
                    error=(
                        "The trained model is unavailable. "
                        "Check the server setup and try again."
                    )
                ),
                503,
            )
        except InferenceFailure:
            app.logger.exception("The FireWatch model could not analyze an image.")
            return jsonify(error=GENERIC_INFERENCE_ERROR), 500
        except RequestEntityTooLarge:
            raise
        except Exception:
            app.logger.exception("Unexpected FireWatch inference failure.")
            return jsonify(error=GENERIC_INFERENCE_ERROR), 500
        timestamp = (
            datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
        return jsonify(
            result={
                "id": f"{timestamp}-{token_hex(8)}",
                "filename": filename,
                "timestamp": timestamp,
                **prediction,
            }
        )

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=True)
