"""
test_predict.py - Tests for the /predict endpoint
==================================================
Covers the full upload/inference/cleanup pipeline:
- valid image -> diagnosis + temp file cleanup
- confidence threshold -> "uncertain" state
- every validation failure path (missing file, empty filename, bad
  extension, corrupt image, oversized upload)
- graceful degradation when the model/labels are unavailable
- one optional test that loads the REAL trained model end to end

Uses the fake model from helpers.py (see conftest.py) for speed and
determinism; the real-model test is marked `slow` and skips if the
model file is not present.
"""

import io
import os

import pytest

import app as app_module
from helpers import FakeModel, PROJECT_ROOT, make_image_bytes


# =============================================================================
# Helpers
# =============================================================================

def post(client, image_bytes=None, filename="leaf.jpg", field="file", with_header=True):
    """POST an optional file to /predict, mirroring the frontend's request.

    Same-origin requests (the real frontend) include the CSRF-mitigation
    header; pass with_header=False to simulate a cross-origin request.
    """
    data = {}
    if image_bytes is not None:
        data = {field: (io.BytesIO(image_bytes), filename)}
    headers = {"X-Requested-With": "XMLHttpRequest"} if with_header else {}
    return client.post(
        "/predict", data=data, content_type="multipart/form-data", headers=headers
    )


# =============================================================================
# Happy path & cleanup
# =============================================================================

def test_valid_image_returns_diagnosis(client):
    r = post(client, make_image_bytes())
    j = r.get_json()
    assert r.status_code == 200
    assert j["success"] is True
    # Default fake probs [0.9, 0.05, 0.05] -> argmax 0 = Cassava Bacterial Blight
    assert j["prediction"] == "Cassava Bacterial Blight (CBB)"
    assert j["severity"] == "disease"
    assert j["confidence"] == 90.0
    assert j["description"] and j["advice"]


def test_temp_file_cleaned_up_after_prediction(client):
    post(client, make_image_bytes())
    uploads = app_module.app.config["UPLOAD_FOLDER"]
    assert os.path.isdir(uploads)
    assert os.listdir(uploads) == []


def test_healthy_prediction_gets_healthy_severity(client):
    # Index 2 = Healthy
    app_module.model.probs = [0.05, 0.05, 0.90]
    j = post(client, make_image_bytes()).get_json()
    assert j["severity"] == "healthy"
    assert j["prediction"] == "Healthy"


# =============================================================================
# Confidence threshold -> "uncertain"
# =============================================================================

def test_low_confidence_returns_uncertain(client):
    app_module.model.probs = [0.50, 0.30, 0.20]
    j = post(client, make_image_bytes()).get_json()
    assert j["success"] is True
    assert j["prediction"] == "Uncertain"
    assert j["severity"] == "uncertain"
    assert j["confidence"] == 50.0
    assert "extension officer" in j["advice"]


def test_confidence_just_below_threshold_is_uncertain(client):
    # Default threshold is 0.65 -> 64% must be uncertain
    app_module.model.probs = [0.64, 0.18, 0.18]
    j = post(client, make_image_bytes()).get_json()
    assert j["severity"] == "uncertain"


def test_confidence_above_threshold_is_definitive(client):
    app_module.model.probs = [0.66, 0.17, 0.17]
    j = post(client, make_image_bytes()).get_json()
    assert j["severity"] != "uncertain"
    assert j["prediction"] == "Cassava Bacterial Blight (CBB)"


# =============================================================================
# Input validation
# =============================================================================

def test_missing_file_field_returns_400(client):
    r = post(client, None)
    assert r.status_code == 400
    assert "No file uploaded" in r.get_json()["error"]


def test_empty_filename_returns_400(client):
    r = post(client, make_image_bytes(), filename="")
    assert r.status_code == 400


def test_disallowed_extension_returns_400(client):
    r = post(client, make_image_bytes(), filename="leaf.txt")
    assert r.status_code == 400
    assert "Invalid file type" in r.get_json()["error"]


def test_corrupt_image_returns_400(client):
    r = post(client, b"this is not an image at all", filename="fake.jpg")
    assert r.status_code == 400
    assert "corrupt" in r.get_json()["error"].lower()


def test_oversized_file_returns_413(client):
    huge = b"x" * (6 * 1024 * 1024)  # 6 MB > 5 MB MAX_CONTENT_LENGTH
    r = post(client, huge, filename="big.jpg")
    assert r.status_code == 413
    assert "too large" in r.get_json()["error"].lower()


# =============================================================================
# Graceful degradation
# =============================================================================

def test_missing_model_returns_503(client):
    app_module.model = None
    r = post(client, make_image_bytes())
    assert r.status_code == 503
    assert "Model not loaded" in r.get_json()["error"]


def test_missing_labels_returns_503(client):
    app_module.labels = None
    r = post(client, make_image_bytes())
    assert r.status_code == 503
    assert "Labels file not found" in r.get_json()["error"]


def test_unknown_route_returns_json_404(client):
    r = client.get("/does-not-exist")
    assert r.status_code == 404
    assert r.is_json


def test_predict_without_custom_header_returns_403(client):
    """Cross-origin requests (no custom header) must be rejected (M2)."""
    r = post(client, make_image_bytes(), with_header=False)
    assert r.status_code == 403
    assert "Missing required request header" in r.get_json()["error"]


def test_predict_with_custom_header_is_accepted(client):
    """Same-origin requests (custom header present) pass the CSRF check."""
    r = post(client, make_image_bytes())
    assert r.status_code == 200
    assert r.get_json()["success"] is True


# =============================================================================
# Health check & security headers (M6)
# =============================================================================

def test_health_endpoint_returns_ok(client):
    r = client.get("/health")
    j = r.get_json()
    assert r.status_code == 200
    assert j["status"] == "ok"
    assert j["model_loaded"] is True
    assert j["labels_loaded"] is True


def test_health_endpoint_reports_unloaded_model(client):
    app_module.model = None
    j = client.get("/health").get_json()
    assert j["status"] == "ok"  # process is up; /predict will 503
    assert j["model_loaded"] is False


def test_security_headers_on_html_and_health(client):
    for path in ("/", "/health"):
        r = client.get(path)
        assert r.headers["X-Content-Type-Options"] == "nosniff"
        assert r.headers["X-Frame-Options"] == "DENY"
        assert r.headers["Referrer-Policy"] == "no-referrer"
        assert r.headers["X-XSS-Protection"] == "1; mode=block"


def test_security_headers_on_predict_response(client):
    r = post(client, make_image_bytes())
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "DENY"


def test_missing_model_file_handled_gracefully():
    original = app_module.MODEL_PATH
    app_module.MODEL_PATH = os.path.join(PROJECT_ROOT, "does_not_exist.h5")
    try:
        model, labels = app_module.load_model_and_labels()
    finally:
        app_module.MODEL_PATH = original
    assert model is None
    assert labels is not None


def test_corrupt_model_file_handled_gracefully():
    original = app_module.load_model
    app_module.load_model = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("unable to parse the model file")
    )
    try:
        model, labels = app_module.load_model_and_labels()
    finally:
        app_module.load_model = original
    assert model is None
    assert labels is not None


def test_corrupt_labels_file_handled_gracefully(tmp_path):
    original = app_module.LABELS_PATH
    bad_labels = tmp_path / "labels.json"
    bad_labels.write_text("{ this is not valid json ")
    app_module.LABELS_PATH = str(bad_labels)
    try:
        model, labels = app_module.load_model_and_labels()
    finally:
        app_module.LABELS_PATH = original
    assert labels is None


def test_unhandled_error_returns_json_500(client):
    """Unexpected internal errors must come back as JSON, not HTML."""
    app_module.app.config["TESTING"] = False  # let Flask use error handlers
    saved_view = app_module.app.view_functions["index"]
    app_module.app.view_functions["index"] = lambda: 1 / 0
    try:
        r = client.get("/")
    finally:
        app_module.app.view_functions["index"] = saved_view
        app_module.app.config["TESTING"] = True
    assert r.status_code == 500
    assert r.is_json
    assert r.get_json()["success"] is False


# =============================================================================
# Real model (optional, slow)
# =============================================================================

@pytest.mark.slow
def test_real_model_end_to_end(client, tmp_path):
    """Run the REAL trained model through the full pipeline (~10s extra).

    Verifies the model output width matches labels.json and that a genuine
    image produces a well-formed response.
    """
    model_path = os.path.join(PROJECT_ROOT, "cassava_model.h5")
    if not os.path.exists(model_path):
        pytest.skip("cassava_model.h5 not found - skipping real-model test")

    import sys

    from helpers import FAKE_TF_MODULES

    # The fake tensorflow modules shadow the real package in sys.modules.
    # Pop them so we can load the real TensorFlow, then restore afterwards.
    saved = {k: sys.modules.pop(k) for k in FAKE_TF_MODULES if k in sys.modules}
    try:
        import tensorflow as tf

        real_model = tf.keras.models.load_model(model_path)
        # Model must output one probability per label in labels.json
        assert real_model.output_shape[-1] == len(app_module.labels)

        app_module.model = real_model
        img_path = tmp_path / "leaf.jpg"
        img_path.write_bytes(make_image_bytes(size=(224, 224)))

        with open(img_path, "rb") as f:
            r = client.post(
                "/predict",
                data={"file": (f, "leaf.jpg")},
                content_type="multipart/form-data",
                headers={"X-Requested-With": "XMLHttpRequest"},
            )
        j = r.get_json()
        assert r.status_code == 200
        assert j["success"] is True
        assert j["prediction"] in {
            "Healthy",
            "Uncertain",
            "Cassava Bacterial Blight (CBB)",
            "Cassava Mosaic Disease (CMD)",
        }
        assert 0 <= j["confidence"] <= 100
    finally:
        sys.modules.update(saved)
