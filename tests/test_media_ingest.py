"""
tests/test_media_ingest.py  -  Persona Multimodal Media Ingestion Tests
=======================================================================
Tests image validation (png/jpg/webp, max 8MB, rejects others), Pillow aspect-ratio resizing
to 1024px max, SHA-256 caching, composite message assembly (max 600 chars), vision fallback,
and multipart POST /simulate endpoint integration with temporary file cleanup.
"""

import io
from PIL import Image
from fastapi.testclient import TestClient
import pytest

from backend.main import app
import backend.main as main_module
from backend.media_ingest import (
    MediaAnalysisResult,
    analyze_image_with_vision_model,
    build_message_description,
    compute_file_hash,
    resize_image_bytes,
    validate_image_metadata,
)


@pytest.fixture
def client():
    main_module.active_job_id = None
    return TestClient(app)


def create_test_image(width: int = 400, height: int = 300, fmt: str = "PNG") -> bytes:
    """Create in-memory image bytes."""
    img = Image.new("RGB", (width, height), color=(180, 200, 220))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_validate_image_metadata_allowed():
    """Accept PNG, JPG, JPEG, WEBP under 8 MB."""
    valid_bytes = create_test_image(100, 100, "PNG")
    validate_image_metadata("sample.png", "image/png", len(valid_bytes))
    validate_image_metadata("photo.jpg", "image/jpeg", len(valid_bytes))
    validate_image_metadata("photo.jpeg", "image/jpeg", len(valid_bytes))
    validate_image_metadata("graphic.webp", "image/webp", len(valid_bytes))


def test_validate_image_metadata_rejected_extensions():
    """Reject non-allowed image extensions with ValueError."""
    data = b"dummy"
    with pytest.raises(ValueError, match="Unsupported image file extension"):
        validate_image_metadata("file.gif", "image/gif", len(data))

    with pytest.raises(ValueError, match="Unsupported image file extension"):
        validate_image_metadata("script.sh", "text/plain", len(data))

    with pytest.raises(ValueError, match="Unsupported image file extension"):
        validate_image_metadata("document.pdf", "application/pdf", len(data))


def test_validate_image_metadata_rejected_size():
    """Reject images exceeding 8 MB with ValueError."""
    oversized = 8 * 1024 * 1024 + 1
    with pytest.raises(ValueError, match="exceeds maximum allowed limit of 8 MB"):
        validate_image_metadata("huge.png", "image/png", oversized)


def test_resize_image_bytes_aspect_ratio():
    """Ensure images larger than 1024px on the long side are resized preserving aspect ratio."""
    # 2000 x 1000 -> resized long side should be 1024, height should be 512
    large_img = create_test_image(width=2000, height=1000, fmt="PNG")
    resized_bytes, b64_str = resize_image_bytes(large_img, max_long_side=1024)

    assert len(resized_bytes) > 0
    assert len(b64_str) > 0

    with Image.open(io.BytesIO(resized_bytes)) as out_img:
        w, h = out_img.size
        assert max(w, h) == 1024
        assert w == 1024
        assert h == 512


def test_compute_file_hash_deterministic():
    """Verify SHA-256 hash is deterministic for caching."""
    img_bytes = create_test_image(50, 50, "PNG")
    h1 = compute_file_hash(img_bytes)
    h2 = compute_file_hash(img_bytes)
    assert h1 == h2
    assert len(h1) == 64


def test_build_message_description_limits():
    """Verify composite message builds from typed text + text_on_image + description capped at 600 chars."""
    media = MediaAnalysisResult(
        description="Electricity bill disconnection warning with Discom emblem",
        text_on_image="BILL OVERDUE: Call 9876543210",
        language="en",
        has_qr=True,
        has_official_look=True,
        visual_appeal=0.7,
        analyzed=True,
    )

    composite = build_message_description(
        typed_text="Check this notice I got today.",
        media_result=media,
        max_chars=600,
    )

    assert "Check this notice I got today." in composite
    assert "BILL OVERDUE: Call 9876543210" in composite
    assert "Electricity bill disconnection warning" in composite
    assert len(composite) <= 600


def test_vision_fallback_when_disabled(monkeypatch):
    """When vision/LLM is disabled, returns analyzed=False with 'image not analysed' note."""
    monkeypatch.setattr("backend.media_ingest.USE_LLM", False)
    img_bytes = create_test_image(80, 80, "PNG")
    result = analyze_image_with_vision_model(img_bytes, filename="fallback.png")
    assert not result.analyzed
    assert result.image_note == "image not analysed"


def test_vision_fallback_on_request_error(monkeypatch):
    """When vision model throws connection or timeout error, returns graceful fallback."""
    import requests
    def mock_post(*args, **kwargs):
        raise requests.ConnectionError("Connection refused to Ollama")
    monkeypatch.setattr("requests.post", mock_post)
    monkeypatch.setattr("backend.media_ingest.USE_LLM", True)
    img_bytes = create_test_image(85, 85, "PNG")
    result = analyze_image_with_vision_model(img_bytes, filename="error.png")
    assert not result.analyzed
    assert result.image_note == "image not analysed"


def test_api_simulate_multipart_success(client, monkeypatch):
    """POST /simulate accepts multipart with text and image, and returns job_id."""
    monkeypatch.setattr("backend.media_ingest.USE_LLM", False)
    img_bytes = create_test_image(120, 120, "PNG")
    form_data = {
        "message": "Notice regarding electricity disconnection",
        "pack_id": "scam_fraud",
        "audience_preset": "urban_seniors",
        "channel": "whatsapp",
        "demo_mode": "true",
    }
    files = {
        "image": ("bill.png", img_bytes, "image/png"),
    }

    res = client.post("/simulate", data=form_data, files=files)
    assert res.status_code == 200
    data = res.json()
    assert "job_id" in data
    assert data["status"] in ("running", "completed")


def test_api_simulate_multipart_image_only(client, monkeypatch):
    """POST /simulate accepts multipart with image only (empty text)."""
    monkeypatch.setattr("backend.media_ingest.USE_LLM", False)
    img_bytes = create_test_image(120, 120, "PNG")
    form_data = {
        "pack_id": "scam_fraud",
        "audience_preset": "urban_seniors",
        "channel": "whatsapp",
        "demo_mode": "true",
    }
    files = {
        "image": ("sample.png", img_bytes, "image/png"),
    }

    res = client.post("/simulate", data=form_data, files=files)
    assert res.status_code == 200
    data = res.json()
    assert "job_id" in data


def test_api_simulate_multipart_rejects_non_image(client):
    """POST /simulate rejects non-image attachments with HTTP 400."""
    form_data = {
        "message": "Valid text",
        "pack_id": "scam_fraud",
    }
    files = {
        "image": ("bad.pdf", b"%PDF-1.4 dummy content", "application/pdf"),
    }

    res = client.post("/simulate", data=form_data, files=files)
    assert res.status_code == 400
    assert "Unsupported image file extension" in res.json()["detail"]


def test_api_simulate_multipart_rejects_oversized(client):
    """POST /simulate rejects files over 8 MB with HTTP 400."""
    huge_bytes = b"0" * (8 * 1024 * 1024 + 10)
    form_data = {
        "message": "Valid text",
        "pack_id": "scam_fraud",
    }
    files = {
        "image": ("huge.png", huge_bytes, "image/png"),
    }

    res = client.post("/simulate", data=form_data, files=files)
    assert res.status_code == 400
    assert "exceeds maximum allowed limit of 8 MB" in res.json()["detail"]
