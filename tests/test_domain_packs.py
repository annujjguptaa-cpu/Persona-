"""Tests for Persona domain packs, public representations, and GET /packs endpoint."""

import pytest
from fastapi.testclient import TestClient

from backend.domain_packs import (
    PACKS,
    DomainPack,
    get_pack,
    list_public_packs,
    public_view,
    verdict_for,
    base_probabilities,
    TRAITS,
    ACTIONS,
    FEATURES,
)
from backend.main import app

EXPECTED_PACK_IDS = [
    "scam_fraud",
    "health_campaign",
    "product_marketing",
    "civic_policy",
    "misinformation",
]


# ---------------------------------------------------------------------------
# 1. Domain Pack Structure & Contents
# ---------------------------------------------------------------------------

def test_all_five_packs_exist():
    """Verify that all 5 required domain packs are registered."""
    assert set(PACKS.keys()) == set(EXPECTED_PACK_IDS)
    for pid in EXPECTED_PACK_IDS:
        pack = get_pack(pid)
        assert isinstance(pack, DomainPack)
        assert pack.id == pid


@pytest.mark.parametrize("pack_id", EXPECTED_PACK_IDS)
def test_pack_verdict_labels(pack_id):
    """Every pack must have exactly 4 verdict labels representing low to high spread tiers."""
    pack = get_pack(pack_id)
    assert len(pack.verdict_labels) == 4, f"{pack_id} must have exactly 4 verdict labels"
    for label in pack.verdict_labels:
        assert isinstance(label, str) and len(label.strip()) > 0


@pytest.mark.parametrize("pack_id", EXPECTED_PACK_IDS)
def test_pack_samples_count_and_legit_example(pack_id):
    """Every pack must have exactly 3 samples, with at least one legitimate example."""
    pack = get_pack(pack_id)
    assert len(pack.samples) == 3, f"{pack_id} must have exactly 3 samples"

    legit_samples = [s for s in pack.samples if s.kind == "legit"]
    assert len(legit_samples) >= 1, f"{pack_id} must include at least one 'legit' sample"
    for sample in pack.samples:
        assert isinstance(sample.label, str) and len(sample.label) > 0
        assert isinstance(sample.text, str) and len(sample.text) > 0
        assert sample.kind in ("typical", "legit")


@pytest.mark.parametrize("pack_id", EXPECTED_PACK_IDS)
def test_pack_audience_presets(pack_id):
    """Every pack must define exactly 3 default audience presets."""
    pack = get_pack(pack_id)
    assert len(pack.audience_presets) == 3, f"{pack_id} must have exactly 3 audience presets"
    for preset in pack.audience_presets:
        assert isinstance(preset.id, str) and len(preset.id) > 0
        assert isinstance(preset.label, str) and len(preset.label) > 0
        assert isinstance(preset.description, str) and len(preset.description) > 0


def test_valence_and_message_improvement_flags():
    """Harmful packs must forbid message improvement; beneficial packs allow it."""
    for pid, pack in PACKS.items():
        if pack.valence == "harmful":
            assert pack.allow_message_improvement is False, (
                f"Harmful pack '{pid}' must not allow message improvement"
            )
        elif pack.valence == "beneficial":
            assert pack.allow_message_improvement is True, (
                f"Beneficial pack '{pid}' should allow message improvement"
            )
        else:
            pytest.fail(f"Invalid valence '{pack.valence}' in pack '{pid}'")


def test_scam_fraud_verdict_labels():
    """Verify specific verdict tiers for scam_fraud."""
    pack = get_pack("scam_fraud")
    assert pack.verdict_labels == ["Contained", "Niche hit", "Wide spread", "Outbreak"]


def test_health_campaign_verdict_labels():
    """Verify specific verdict tiers for health_campaign."""
    pack = get_pack("health_campaign")
    assert pack.verdict_labels == ["Ignored", "Niche reach", "Strong reach", "Breakout"]


# ---------------------------------------------------------------------------
# 2. Public View & GET /packs Endpoint
# ---------------------------------------------------------------------------

def test_public_view_omits_internal_weights():
    """Public representation must not leak internal trait weights or base rates."""
    for pack in PACKS.values():
        pub = public_view(pack)
        assert "trait_weights" not in pub
        assert "feature_weights" not in pub
        assert "base_rates" not in pub
        assert "tier_thresholds" not in pub
        assert "llm_context" not in pub

        assert pub["id"] == pack.id
        assert pub["title"] == pack.title
        assert pub["description"] == pack.description
        assert pub["act_label"] == pack.act_label
        assert pub["valence"] == pack.valence
        assert pub["allow_message_improvement"] == pack.allow_message_improvement
        assert len(pub["verdict_labels"]) == 4
        assert len(pub["audience_presets"]) == 3
        assert len(pub["samples"]) == 3


def test_get_packs_endpoint():
    """GET /packs returns 200 with all 5 packs in sanitized public format."""
    client = TestClient(app)
    response = client.get("/packs")
    assert response.status_code == 200

    packs = response.json()
    assert isinstance(packs, list)
    assert len(packs) == 5

    ids = [p["id"] for p in packs]
    assert set(ids) == set(EXPECTED_PACK_IDS)

    for p in packs:
        # Check presence of public attributes
        assert "id" in p
        assert "title" in p
        assert "description" in p
        assert "act_label" in p
        assert "valence" in p
        assert "allow_message_improvement" in p
        assert "verdict_labels" in p
        assert "audience_presets" in p
        assert "samples" in p

        # Check absence of internal weights
        assert "trait_weights" not in p
        assert "feature_weights" not in p
        assert "base_rates" not in p
        assert "tier_thresholds" not in p
        assert "llm_context" not in p

        assert len(p["verdict_labels"]) == 4
        assert len(p["samples"]) == 3
        assert any(s["kind"] == "legit" for s in p["samples"])


# ---------------------------------------------------------------------------
# 3. Verdict and Probability Calculation Logic
# ---------------------------------------------------------------------------

def test_verdict_for_tiers():
    """Verdict calculation must map reach & R0 to proper tier and label."""
    pack = get_pack("scam_fraud")
    # Low reach and low R0 -> Tier 0 "Contained"
    v0 = verdict_for(pack, reach=0.05, r0=0.5)
    assert v0["tier"] == 0
    assert v0["label"] == "Contained"

    # High reach -> Tier 3 "Outbreak"
    v3 = verdict_for(pack, reach=0.60, r0=0.5)
    assert v3["tier"] == 3
    assert v3["label"] == "Outbreak"


def test_base_probabilities_ordering_and_bounds():
    """All action probabilities must be between 0 and 1, and act <= engage."""
    for pack in PACKS.values():
        mid_traits = {t: 0.5 for t in TRAITS}
        mid_features = {f: 0.5 for f in FEATURES}
        probs = base_probabilities(pack, mid_traits, mid_features)

        for a in ACTIONS:
            assert 0.0 <= probs[a] <= 1.0, f"{pack.id} action {a} out of bounds"
        assert probs["act"] <= probs["engage"], f"{pack.id} act must be <= engage"


def test_scam_vulnerable_vs_skeptical_calibration():
    """Vulnerable individuals must have higher act probability on scams than skeptics."""
    scam = get_pack("scam_fraud")
    vuln_traits = {t: 0.5 for t in TRAITS} | {
        "skepticism": 0.2,
        "digital_literacy": 0.25,
        "trust_in_source": 0.8,
        "time_pressure": 0.7,
    }
    skep_traits = {t: 0.5 for t in TRAITS} | {
        "skepticism": 0.9,
        "digital_literacy": 0.8,
    }
    scam_features = {
        "urgency": 0.9,
        "emotional_pull": 0.6,
        "authority_claim": 0.8,
        "call_to_action": 0.9,
        "has_link": 1.0,
    }

    p_vuln = base_probabilities(scam, vuln_traits, scam_features)
    p_skep = base_probabilities(scam, skep_traits, scam_features)

    assert p_vuln["act"] > p_skep["act"]
    assert p_skep["act"] < 0.15
