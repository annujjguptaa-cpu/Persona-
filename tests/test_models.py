"""Tests for Persona Pydantic models, configurations, and health endpoints."""

import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient

from backend.models import (
    Archetype,
    Person,
    MessageFeatures,
    Reaction,
    WaveState,
    SimulationResult,
)
from backend.config import (
    OLLAMA_URL,
    MODEL,
    USE_LLM,
    N_PEOPLE,
    SEED,
    MAX_WAVES,
)
from backend.main import app


# ---------------------------------------------------------------------------
# 1. Config Tests
# ---------------------------------------------------------------------------

def test_config_constants():
    """Verify that configuration defaults match specifications."""
    assert isinstance(OLLAMA_URL, str) and len(OLLAMA_URL) > 0
    assert isinstance(MODEL, str) and len(MODEL) > 0
    assert isinstance(USE_LLM, bool)
    assert N_PEOPLE == 120
    assert SEED == 42
    assert MAX_WAVES == 6


# ---------------------------------------------------------------------------
# 2. Archetype Tests
# ---------------------------------------------------------------------------

def test_archetype_valid():
    """Archetype model should instantiate cleanly with valid probability & traits."""
    arch = Archetype(
        id="skeptic",
        label="Analytical Skeptic",
        description="Scrutinizes sources, questions urgency triggers, rarely forwards.",
        share_of_population=0.25,
        traits={"skepticism": 0.85, "emotional_reactivity": 0.15},
    )
    assert arch.id == "skeptic"
    assert arch.share_of_population == 0.25
    assert arch.traits["skepticism"] == 0.85


@pytest.mark.parametrize("invalid_share", [-0.01, 1.01, 2.5, -1.0])
def test_archetype_invalid_share_of_population(invalid_share):
    """Archetype share_of_population must strictly be between 0.0 and 1.0."""
    with pytest.raises(ValidationError):
        Archetype(
            id="test",
            label="Test",
            description="Test",
            share_of_population=invalid_share,
            traits={},
        )


# ---------------------------------------------------------------------------
# 3. Person Tests
# ---------------------------------------------------------------------------

def test_person_valid():
    """Person model should instantiate cleanly with demographic and archetype references."""
    person = Person(
        id="person_001",
        name="Alex Chen",
        archetype_id="skeptic",
        age=34,
        language="en",
        area="metro",
        traits={"skepticism": 0.88},
    )
    assert person.id == "person_001"
    assert person.name == "Alex Chen"
    assert person.archetype_id == "skeptic"
    assert person.age == 34
    assert person.language == "en"
    assert person.area == "metro"
    assert person.traits["skepticism"] == 0.88


def test_person_invalid_age():
    """Person age must be non-negative."""
    with pytest.raises(ValidationError):
        Person(
            id="p_neg",
            name="Invalid",
            archetype_id="skeptic",
            age=-5,
        )


# ---------------------------------------------------------------------------
# 4. MessageFeatures Tests
# ---------------------------------------------------------------------------

def test_message_features_valid():
    """MessageFeatures should accept booleans, lists, and validated probabilities."""
    mf = MessageFeatures(
        is_suspicious=True,
        urgency=0.9,
        emotional_pull=0.75,
        authority_claim=0.8,
        call_to_action=True,
        has_link=True,
        language="en",
        channel_guess="sms",
        red_flags=["fake_bank_domain", "artificial_urgency"],
        legit_signals=[],
        topics=["banking", "security"],
    )
    assert mf.is_suspicious is True
    assert mf.urgency == 0.9
    assert mf.emotional_pull == 0.75
    assert mf.authority_claim == 0.8
    assert mf.call_to_action is True
    assert mf.has_link is True
    assert "fake_bank_domain" in mf.red_flags


def test_message_features_call_to_action_score():
    """call_to_action also accepts numeric strength scores in [0.0, 1.0]."""
    mf = MessageFeatures(call_to_action=0.65)
    assert mf.call_to_action == 0.65


@pytest.mark.parametrize("invalid_val", [-0.1, 1.1, 5.0])
def test_message_features_probability_bounds(invalid_val):
    """Urgency, emotional pull, and authority claim must be bounded in [0.0, 1.0]."""
    with pytest.raises(ValidationError):
        MessageFeatures(urgency=invalid_val)
    with pytest.raises(ValidationError):
        MessageFeatures(emotional_pull=invalid_val)
    with pytest.raises(ValidationError):
        MessageFeatures(authority_claim=invalid_val)
    with pytest.raises(ValidationError):
        MessageFeatures(call_to_action=invalid_val)


# ---------------------------------------------------------------------------
# 5. Reaction Tests
# ---------------------------------------------------------------------------

def test_reaction_valid():
    """Reaction probabilities (ignore, notice, engage, act, share, report) must validate."""
    reaction = Reaction(
        archetype_id="skeptic",
        ignore=0.4,
        notice=0.9,
        engage=0.3,
        act=0.05,
        share=0.02,
        report=0.45,
        top_trigger="artificial_urgency",
        reason="Detected mismatched sender address and pressure tactics.",
    )
    assert reaction.archetype_id == "skeptic"
    assert reaction.ignore == 0.4
    assert reaction.notice == 0.9
    assert reaction.engage == 0.3
    assert reaction.act == 0.05
    assert reaction.share == 0.02
    assert reaction.report == 0.45
    assert reaction.top_trigger == "artificial_urgency"


@pytest.mark.parametrize(
    "field_name",
    ["ignore", "notice", "engage", "act", "share", "report"]
)
@pytest.mark.parametrize("invalid_prob", [-0.05, 1.05, 2.0])
def test_reaction_probability_bounds(field_name, invalid_prob):
    """All 6 reaction probabilities must be strictly validated with Field(ge=0, le=1)."""
    kwargs = {
        "archetype_id": "test_arch",
        "ignore": 0.1,
        "notice": 0.5,
        "engage": 0.2,
        "act": 0.1,
        "share": 0.1,
        "report": 0.0,
    }
    kwargs[field_name] = invalid_prob
    with pytest.raises(ValidationError):
        Reaction(**kwargs)


# ---------------------------------------------------------------------------
# 6. WaveState and SimulationResult Tests
# ---------------------------------------------------------------------------

def test_wave_state_valid():
    """WaveState tracks diffusion progression at wave t."""
    ws = WaveState(
        wave=1,
        reached_ids=["p1", "p2", "p3"],
        newly_reached=3,
        cumulative_reached=3,
        reactions={"notice": 3, "share": 1, "ignore": 1},
        shares=1,
        acts=0,
        reports=0,
    )
    assert ws.wave == 1
    assert len(ws.reached_ids) == 3
    assert ws.cumulative_reached == 3
    assert ws.shares == 1


def test_simulation_result_valid():
    """SimulationResult aggregates waves, edges, totals, group stats, verdict, and insights."""
    ws0 = WaveState(wave=0, newly_reached=1, cumulative_reached=1)
    ws1 = WaveState(wave=1, newly_reached=3, cumulative_reached=4, shares=1)

    result = SimulationResult(
        waves=[ws0, ws1],
        edges=[{"source": "p0", "target": "p1"}, {"source": "p0", "target": "p2"}],
        totals={"total_reached": 4, "total_shared": 1, "total_reported": 0},
        group_stats={"skeptic": {"reached": 1, "act_rate": 0.0}},
        verdict="Low Viral Hazard: message was stopped by early analytical skepticism.",
        insights=[
            "High skepticism among metro cohorts halted diffusion at Wave 2.",
            "Absence of authenticating domain caused 45% of recipients to ignore.",
        ],
    )
    assert len(result.waves) == 2
    assert len(result.edges) == 2
    assert result.totals["total_reached"] == 4
    assert "skeptic" in result.group_stats
    assert "Low Viral Hazard" in result.verdict
    assert len(result.insights) == 2


# ---------------------------------------------------------------------------
# 7. FastAPI Endpoint & Static Files Tests
# ---------------------------------------------------------------------------

def test_health_endpoint():
    """GET /health should return 200, app metadata, and Ollama probe."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["app"] == "persona"
    assert "ollama" in data
    assert "config" in data
    assert data["config"]["n_people"] == 120
    assert data["config"]["seed"] == 42
    assert data["config"]["max_waves"] == 6


def test_static_index_serving():
    """GET / should serve frontend/index.html statically."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "Persona" in response.text
    assert "Synthetic Audience Message Spread Simulator" in response.text
