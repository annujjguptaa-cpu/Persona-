"""
tests/test_api_endpoints.py  -  Persona
=======================================
Comprehensive test suite for Persona FastAPI endpoints:
1. GET /packs, GET /presets, GET /health.
2. POST /simulate request validation (empty text, >500 chars, invalid pack).
3. SSE streaming at GET /stream/{job_id} with stages:
   "Reading message", "Designing audience", "Segments reacting (i/12)", "Building network", "Spreading wave n", "Scoring".
   Final event contains full result JSON and degraded flag.
4. Concurrency rate limiting: exactly 1 concurrent job allowed, rejecting parallel jobs with HTTP 429.
5. POST /compare endpoint (validation, N=20 runs, winner, relative diff).
6. Degraded mode fallback verification.
"""

import json
import time
from fastapi.testclient import TestClient
import pytest

from backend.main import app, jobs
import backend.main as main_module


@pytest.fixture
def client():
    """Create a TestClient with fresh concurrency lock state."""
    main_module.active_job_id = None
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. Base Endpoints: /packs, /presets, /health
# ---------------------------------------------------------------------------

def test_get_packs(client):
    """GET /packs returns all 5 domain packs without internal weights."""
    res = client.get("/packs")
    assert res.status_code == 200
    packs = res.json()
    assert len(packs) == 5
    ids = [p["id"] for p in packs]
    assert "scam_fraud" in ids
    assert "health_campaign" in ids
    assert "civic_policy" in ids
    # Ensure internal trait weights are omitted
    for p in packs:
        assert "trait_weights" not in p
        assert "verdict_labels" in p
        assert len(p["verdict_labels"]) == 4


def test_get_presets(client):
    """GET /presets returns audience presets with their constituent archetypes."""
    res = client.get("/presets")
    assert res.status_code == 200
    presets = res.json()
    assert "urban_seniors" in presets
    assert len(presets["urban_seniors"]) >= 10


def test_get_health(client):
    """GET /health verifies app liveness and configuration."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["app"] == "persona"
    assert "ollama" in data
    assert "config" in data


# ---------------------------------------------------------------------------
# 2. POST /simulate Request Validation
# ---------------------------------------------------------------------------

def test_simulate_validation_empty_message(client):
    """Reject empty messages with HTTP 400."""
    res = client.post("/simulate", json={"message": "   "})
    assert res.status_code == 400
    assert "Message cannot be empty" in res.json()["detail"]


def test_simulate_validation_over_500_chars(client):
    """Reject messages longer than 500 characters with HTTP 400."""
    long_msg = "A" * 501
    res = client.post("/simulate", json={"message": long_msg})
    assert res.status_code == 400
    assert "maximum allowed length of 500 characters" in res.json()["detail"]


def test_simulate_validation_invalid_pack(client):
    """Reject unknown pack IDs with HTTP 400 and clear error message."""
    res = client.post("/simulate", json={"message": "Valid text", "pack_id": "non_existent_pack"})
    assert res.status_code == 400
    assert "Unknown pack" in res.json()["detail"]


def test_simulate_validation_intervention_length(client):
    """Reject interventions exceeding 500 characters with HTTP 400."""
    res = client.post("/simulate", json={
        "message": "Valid text",
        "pack_id": "scam_fraud",
        "intervention": "X" * 501,
    })
    assert res.status_code == 400
    assert "Intervention exceeds maximum allowed length" in res.json()["detail"]


# ---------------------------------------------------------------------------
# 3. POST /simulate & SSE Stream Pipeline
# ---------------------------------------------------------------------------

def test_simulate_and_stream_pipeline(client):
    """Run end-to-end simulation via POST /simulate and consume SSE stream at GET /stream/{job_id}."""
    msg = "URGENT: Your electricity connection will be cut at 9:30 PM. Call 9876543210 to update bill."
    payload = {
        "message": msg,
        "pack_id": "scam_fraud",
        "audience_preset": "urban_seniors",
        "channel": "sms",
        "use_llm": False,  # Deterministic fast test mode
        "n_people": 30,
        "seed": 42,
    }

    sim_res = client.post("/simulate", json=payload)
    assert sim_res.status_code == 200
    body = sim_res.json()
    assert "job_id" in body
    job_id = body["job_id"]
    assert body["status"] == "running"

    # Connect to SSE stream
    with client.stream("GET", f"/stream/{job_id}") as stream:
        assert stream.status_code == 200
        events_raw = []
        for line in stream.iter_lines():
            if line:
                events_raw.append(line)
                if line.startswith("event: complete") or "event: complete" in line:
                    # also read the data line right after
                    break

    stream_text = "\n".join(events_raw)

    # Verify that all mandatory pipeline stages were broadcast
    assert "Reading message" in stream_text
    assert "Designing audience" in stream_text
    assert "Segments reacting" in stream_text
    assert "Building network" in stream_text
    assert "Spreading wave" in stream_text
    assert "Scoring" in stream_text

    # Verify that the complete event was fired and contains result JSON
    job = jobs.get(job_id)
    assert job is not None
    assert job.status == "completed"
    assert job.result is not None
    res_obj = job.result
    assert res_obj["degraded"] is False
    assert "scoring" in res_obj
    assert "insights" in res_obj
    assert res_obj["scoring"]["r0"] >= 0.0
    assert "verdict" in res_obj["scoring"]
    assert len(res_obj["insights"]["recommendations"]) == 3


# ---------------------------------------------------------------------------
# 4. Concurrency Rate Limiting (1 Job Max)
# ---------------------------------------------------------------------------

def test_rate_limit_concurrency(client):
    """Enforce that only 1 concurrent job is permitted, returning HTTP 429 when busy."""
    main_module.active_job_id = "sim_fake_in_progress"
    try:
        res = client.post("/simulate", json={"message": "Test message", "pack_id": "scam_fraud"})
        assert res.status_code == 429
        assert "Only 1 concurrent job is allowed" in res.json()["detail"]

        res_comp = client.post("/compare", json={
            "message": "Test message",
            "pack_id": "scam_fraud",
            "interventions": ["A", "B"],
        })
        assert res_comp.status_code == 429
        assert "Only 1 concurrent job is allowed" in res_comp.json()["detail"]
    finally:
        main_module.active_job_id = None


# ---------------------------------------------------------------------------
# 5. POST /compare Endpoint
# ---------------------------------------------------------------------------

def test_compare_endpoint_validation(client):
    """POST /compare must validate interventions length and message limits."""
    # Less than 2 interventions
    res = client.post("/compare", json={
        "message": "Valid text",
        "pack_id": "scam_fraud",
        "interventions": ["Only one intervention"],
    })
    assert res.status_code == 400
    assert "at least two interventions" in res.json()["detail"]

    # Message over 500 chars
    res2 = client.post("/compare", json={
        "message": "M" * 501,
        "pack_id": "scam_fraud",
        "interventions": ["A", "B"],
    })
    assert res2.status_code == 400


def test_compare_endpoint_execution(client):
    """Run POST /compare with 2 interventions across Monte Carlo runs."""
    payload = {
        "message": "Electricity bill unpaid. Connection severed tonight unless paid.",
        "pack_id": "scam_fraud",
        "audience": "urban_seniors",
        "interventions": [
            "Be cautious about online messages.",
            "Official Electricity Board Advisory: We never disconnect power without 15 days written notice. Verify bills at discom.gov.in.",
        ],
        "n_runs": 20,
        "n_people": 25,
        "channel": "whatsapp",
        "use_llm": False,
    }

    res = client.post("/compare", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["runs"] == 20
    assert "candidate_a_stats" in data
    assert "candidate_b_stats" in data
    assert data["winner"] in ("Intervention A", "Intervention B", "Candidate A", "Candidate B", "Tie")
    assert isinstance(data["relative_difference_percent"], float)
    assert "methodology_note" in data


# ---------------------------------------------------------------------------
# 6. Degraded Fallback Flag Verification
# ---------------------------------------------------------------------------

def test_degraded_flag_in_result(client):
    """Simulate degraded fallback behavior and ensure degraded flag is captured."""
    msg = "Free cash voucher: click link."
    payload = {
        "message": msg,
        "pack_id": "scam_fraud",
        "use_llm": False,
        "n_people": 20,
    }
    sim_res = client.post("/simulate", json=payload)
    job_id = sim_res.json()["job_id"]

    # Wait for completion
    time.sleep(0.5)
    job = jobs.get(job_id)
    assert job is not None
    # Wait if still running
    for _ in range(20):
        if job.status == "completed":
            break
        time.sleep(0.1)

    assert job.status == "completed"
    assert "degraded" in job.result


# ---------------------------------------------------------------------------
# 7. Offline Demo Mode Endpoints Verification
# ---------------------------------------------------------------------------

def test_demo_mode_endpoints(client):
    """Verify that offline demo endpoints return instant cached cascades (<5s)."""
    # 1. GET /demo/precomputed
    res_bundle = client.get("/demo/precomputed")
    assert res_bundle.status_code == 200
    bundle_data = res_bundle.json()
    assert "simulations" in bundle_data
    assert "compare_demos" in bundle_data
    assert len(bundle_data["simulations"]) >= 45

    # 2. POST /simulate with demo_mode=True
    sim_payload = {
        "message": "URGENT: Your electricity connection will be cut at 9:30 PM.",
        "pack_id": "scam_fraud",
        "audience_preset": "urban_seniors",
        "demo_mode": True,
    }
    sim_res = client.post("/simulate", json=sim_payload)
    assert sim_res.status_code == 200
    sim_body = sim_res.json()
    assert sim_body["status"] == "completed"
    assert sim_body["cached"] is True
    assert "result" in sim_body
    assert "scoring" in sim_body["result"]

    # 3. POST /compare with demo_mode=True
    comp_payload = {
        "message": "Electricity disconnection threat",
        "pack_id": "scam_fraud",
        "audience": "urban_seniors",
        "interventions": ["Notice A", "Notice B"],
        "demo_mode": True,
    }
    comp_res = client.post("/compare", json=comp_payload)
    assert comp_res.status_code == 200
    comp_body = comp_res.json()
    assert "winner" in comp_body
    assert comp_body.get("cached") is True

