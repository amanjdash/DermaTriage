from __future__ import annotations

from io import BytesIO

import pytest
from backend.app import main as main_module
from backend.app.main import app
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(main_module, "CHECKPOINT_PATH", str(tmp_path / "missing.pth"))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def ready_client(client, monkeypatch):
    service = app.state.inference_service
    service.predictor = {"checkpoint": {"model_version": "test-model", "dataset_reference": "test fixture"}}
    monkeypatch.setattr(
        service,
        "predict",
        lambda _image, _passes: {
            "prediction": "nv",
            "deterministic_prediction": "nv",
            "confidence": 0.8,
            "uncertainty": 0.5,
            "uncertainty_normalized": 0.25,
            "uncertainty_method": "predictive_entropy",
            "mc_passes": 30,
            "probabilities": {"akiec": 0.01, "bcc": 0.02, "bkl": 0.03, "df": 0.01, "mel": 0.08, "nv": 0.8, "vasc": 0.05},
            "class_probability_variance": {name: 0.0 for name in ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")},
            "inference_time_ms": 12.0,
            "preprocessing_time_ms": 2.0,
            "model_inference_time_ms": 9.0,
            "model_name": "EfficientNet-B3",
            "model_version": "test-model",
            "dataset_reference": "test fixture",
            "device": "cpu",
        },
    )
    return client


def _png_bytes():
    buffer = BytesIO()
    Image.new("RGB", (20, 20), "#aa7755").save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_and_model_info_without_checkpoint(client):
    health = client.get("/health")
    model = client.get("/model-info")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["model_loaded"] is False
    assert model.json()["model_available"] is False
    assert len(model.json()["classes"]) == 7


def test_predict_returns_typed_payload(ready_client):
    response = ready_client.post("/predict", files={"image": ("sample.png", _png_bytes(), "image/png")})
    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] == "nv"
    assert payload["deterministic_prediction"] == "nv"
    assert len(payload["probabilities"]) == 7


def test_predict_reports_missing_model(client):
    unavailable = client.post("/predict", files={"image": ("sample.png", _png_bytes(), "image/png")})
    assert unavailable.status_code == 503
    assert unavailable.json()["detail"]["code"] == "model_unavailable"


def test_predict_rejects_unsupported_and_corrupt_images(ready_client):
    unsupported = ready_client.post("/predict", files={"image": ("fake.svg", b"<svg/>", "image/svg+xml")})
    corrupt = ready_client.post("/predict", files={"image": ("fake.png", b"bad bytes", "image/png")})
    assert unsupported.status_code == 415
    assert unsupported.json()["detail"]["code"] == "unsupported_media_type"
    assert corrupt.status_code == 400
    assert corrupt.json()["detail"]["code"] == "invalid_image"


def test_predict_rejects_oversized_upload(ready_client):
    response = ready_client.post("/predict", files={"image": ("large.png", b"x" * (10 * 1024 * 1024 + 1), "image/png")})
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "file_too_large"


def test_predict_rejects_oversized_content_length_before_parsing(client):
    response = client.post(
        "/predict",
        content=b"",
        headers={"content-type": "multipart/form-data; boundary=test", "content-length": str(20 * 1024 * 1024)},
    )
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "file_too_large"


def test_predict_rejects_missing_multipart_file(client):
    response = client.post("/predict", data={})
    assert response.status_code == 422

