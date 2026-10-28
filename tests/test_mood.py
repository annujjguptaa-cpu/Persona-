"""
tests/test_mood.py  -  Persona Community Mood & Macro Climate Tests
===================================================================
Tests MoodState model, json persistence, exponential temporal decay,
per-archetype sensitivity scaling, aggregate privacy guarantees, and /mood endpoints.
"""

from datetime import datetime, timedelta, timezone
import json
import math
from fastapi.testclient import TestClient
import pytest

from backend.main import app
from backend.models import Archetype
from backend.mood import (
    MoodState,
    compute_effective_trait,
    get_days_since_update,
    load_mood,
    save_mood,
)
from backend.population import generate_population


@pytest.fixture
def client():
    return TestClient(app)


def test_mood_state_bounds():
    """Verify that MoodState validates bounds [0.0, 1.0]."""
    # Valid bounds
    valid = MoodState(anxiety=0.0, trust_in_health_system=1.0, rumour_level=0.5)
    assert valid.anxiety == 0.0
    assert valid.trust_in_health_system == 1.0
    assert valid.rumour_level == 0.5
    assert valid.updated_at is not None

    # Invalid bounds
    with pytest.raises(Exception):
        MoodState(anxiety=-0.1)

    with pytest.raises(Exception):
        MoodState(trust_in_health_system=1.1)

    with pytest.raises(Exception):
        MoodState(rumour_level=-0.01)


def test_mood_persistence(tmp_path, monkeypatch):
    """Verify MoodState saves to and loads from mood.json."""
    fake_mood_file = tmp_path / "mood.json"
    fake_root_file = tmp_path / "root_mood.json"
    monkeypatch.setattr("backend.mood.MOOD_FILE", fake_mood_file)
    monkeypatch.setattr("backend.mood.ROOT_MOOD_FILE", fake_root_file)

    mood = MoodState(anxiety=0.82, trust_in_health_system=0.35, rumour_level=0.74)
    save_mood(mood)

    assert fake_mood_file.exists()
    loaded = load_mood()
    assert loaded.anxiety == 0.82
    assert loaded.trust_in_health_system == 0.35
    assert loaded.rumour_level == 0.74


def test_days_since_update():
    """Verify calculation of days since updated_at."""
    now = datetime.now(timezone.utc)
    two_days_ago = (now - timedelta(days=2.5)).isoformat()
    mood = MoodState(updated_at=two_days_ago)

    days = get_days_since_update(mood, now=now)
    assert abs(days - 2.5) < 0.01


def test_compute_effective_trait_formula_and_decay():
    """Verify formula: effective_trait = base + shift * sensitivity * exp(-days/5)."""
    # At days = 0, decay = 1.0
    # If anxiety = 0.8 -> shift = 0.8 - 0.5 = 0.3
    # base = 0.4, sensitivity = 1.5
    # expected = 0.4 + 0.3 * 1.5 * 1.0 = 0.85
    mood = MoodState(anxiety=0.8, trust_in_health_system=0.5, rumour_level=0.5)
    eff_0 = compute_effective_trait(
        base=0.4,
        trait_name="emotional_reactivity",
        sensitivity=1.5,
        mood=mood,
        days=0.0,
    )
    assert math.isclose(eff_0, 0.85, abs_tol=1e-5)

    # At days = 5.0, decay = exp(-1) = 0.367879...
    # expected = 0.4 + 0.3 * 1.5 * exp(-1) = 0.4 + 0.45 * 0.367879 = 0.565545...
    eff_5 = compute_effective_trait(
        base=0.4,
        trait_name="emotional_reactivity",
        sensitivity=1.5,
        mood=mood,
        days=5.0,
    )
    expected_5 = 0.4 + (0.3 * 1.5 * math.exp(-1.0))
    assert math.isclose(eff_5, expected_5, abs_tol=1e-5)

    # Clamping bounds: high shift does not exceed 1.0
    eff_clamped = compute_effective_trait(
        base=0.9,
        trait_name="emotional_reactivity",
        sensitivity=2.0,
        mood=mood,
        days=0.0,
    )
    assert eff_clamped == 1.0


def test_population_mood_integration():
    """Verify that population generation respects mood state and per-archetype sensitivity."""
    arch = Archetype(
        id="anxious_cohort",
        name="Anxious Cohort",
        label="Anxious Cohort",
        description="Highly sensitive to macro anxiety",
        share_of_population=1.0,
        traits={"emotional_reactivity": 0.4},
        sensitivity={"emotional_reactivity": 1.5},
    )

    # Low anxiety mood
    mood_calm = MoodState(anxiety=0.1, updated_at=datetime.now(timezone.utc).isoformat())
    pop_calm = generate_population([arch], n=30, seed=42, mood=mood_calm)
    avg_reactivity_calm = sum(p.traits["emotional_reactivity"] for p in pop_calm) / len(pop_calm)

    # High anxiety mood
    mood_high = MoodState(anxiety=0.9, updated_at=datetime.now(timezone.utc).isoformat())
    pop_high = generate_population([arch], n=30, seed=42, mood=mood_high)
    avg_reactivity_high = sum(p.traits["emotional_reactivity"] for p in pop_high) / len(pop_high)

    assert avg_reactivity_high > avg_reactivity_calm


def test_mood_endpoints(client):
    """Verify GET /mood and POST /mood."""
    # GET /mood
    res_get = client.get("/mood")
    assert res_get.status_code == 200
    data = res_get.json()
    assert "anxiety" in data
    assert "trust_in_health_system" in data
    assert "rumour_level" in data
    assert "updated_at" in data

    # POST /mood valid
    update_payload = {
        "anxiety": 0.72,
        "trust_in_health_system": 0.41,
        "rumour_level": 0.65,
    }
    res_post = client.post("/mood", json=update_payload)
    assert res_post.status_code == 200
    updated = res_post.json()
    assert updated["anxiety"] == 0.72
    assert updated["trust_in_health_system"] == 0.41
    assert updated["rumour_level"] == 0.65

    # POST /mood invalid (out of bounds)
    res_bad = client.post("/mood", json={"anxiety": 1.5, "trust_in_health_system": 0.5, "rumour_level": 0.5})
    assert res_bad.status_code == 422
