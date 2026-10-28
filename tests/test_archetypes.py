"""Tests for audience presets, LLM-driven archetype generation, validation, and caching."""

from unittest.mock import patch, MagicMock
import pytest

from backend.archetypes import (
    PRESETS,
    generate_archetypes,
    validate_and_normalize_archetypes,
    find_closest_preset,
)
from backend.domain_packs import TRAITS
from backend.models import Archetype
from backend.cache import clear_cache, hash_key

EXPECTED_PRESETS = [
    "urban_seniors",
    "college_students",
    "rural_farmers",
    "working_parents",
    "small_shopkeepers",
    "community_health_workers",
    "gig_workers",
    "urban_professionals",
    "homemakers",
]


# ---------------------------------------------------------------------------
# 1. Preset Structure and Trait Calibrations
# ---------------------------------------------------------------------------

def test_all_presets_exist():
    """Verify that all 9 expected audience presets are populated."""
    assert set(PRESETS.keys()) == set(EXPECTED_PRESETS)


@pytest.mark.parametrize("preset_id", EXPECTED_PRESETS)
def test_preset_archetypes_count_and_shares(preset_id):
    """Each preset must contain 10-12 archetypes whose shares sum to 1.0."""
    archetypes = PRESETS[preset_id]
    assert 10 <= len(archetypes) <= 12, f"{preset_id} should have 10-12 archetypes"

    total_share = sum(a.share_of_population for a in archetypes)
    assert total_share == pytest.approx(1.0, abs=1e-4), (
        f"{preset_id} shares must sum to 1.0, got {total_share}"
    )


@pytest.mark.parametrize("preset_id", EXPECTED_PRESETS)
def test_preset_traits_validity(preset_id):
    """Every archetype in each preset must contain all 10 universal traits bounded in [0, 1]."""
    for arch in PRESETS[preset_id]:
        assert isinstance(arch.id, str) and len(arch.id) > 0
        assert isinstance(arch.label, str) and len(arch.label) > 0
        assert isinstance(arch.description, str) and len(arch.description) > 0

        # Must have all 10 universal traits
        for trait in TRAITS:
            assert trait in arch.traits, f"{arch.id} missing trait {trait}"
            val = arch.traits[trait]
            assert 0.0 <= val <= 1.0, f"{arch.id} trait {trait}={val} out of bounds"


@pytest.mark.parametrize("preset_id", EXPECTED_PRESETS)
def test_preset_diversity_skeptic_and_vulnerable(preset_id):
    """Each preset must contain at least 2 skeptical (>0.6) and 2 vulnerable (<0.4) segments."""
    archetypes = PRESETS[preset_id]
    skeptics = [a for a in archetypes if a.traits["skepticism"] > 0.6]
    vulnerable = [a for a in archetypes if a.traits["digital_literacy"] < 0.4]

    assert len(skeptics) >= 2, f"{preset_id} needs >= 2 skeptical segments, got {len(skeptics)}"
    assert len(vulnerable) >= 2, f"{preset_id} needs >= 2 vulnerable segments, got {len(vulnerable)}"


# ---------------------------------------------------------------------------
# 2. Closest Preset Matching
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "query,expected_preset",
    [
        ("Retired elderly grandparents in Delhi colony", "urban_seniors"),
        ("Engineering college students in campus hostel", "college_students"),
        ("Cotton and paddy farmers in rural Vidarbha", "rural_farmers"),
        ("Working mothers and fathers with toddler children", "working_parents"),
        ("Small kirana store traders in wholesale market", "small_shopkeepers"),
        ("ASHA workers and anganwadi healthcare volunteers", "community_health_workers"),
        ("Zomato and Swiggy delivery riders on daily wages", "gig_workers"),
        ("Corporate software engineers and tech managers", "urban_professionals"),
        ("Stay-at-home mothers in society neighborhood", "homemakers"),
    ],
)
def test_find_closest_preset(query, expected_preset):
    """Test heuristic keyword mapping to relevant presets."""
    assert find_closest_preset(query) == expected_preset


# ---------------------------------------------------------------------------
# 3. Validation and Normalization Logic
# ---------------------------------------------------------------------------

def test_validate_and_normalize_valid_data():
    """Valid raw dicts should be clamped, normalized to sum=1.0, and converted to Archetypes."""
    raw_dicts = [
        {
            "id": "skeptic_1",
            "label": "Deep Skeptic",
            "description": "Never trusts links",
            "share_of_population": 0.5,
            "traits": {"skepticism": 0.9, "digital_literacy": 0.8},
        },
        {
            "id": "skeptic_2",
            "label": "Audit Checker",
            "description": "Double-checks all sources",
            "share_of_population": 0.5,
            "traits": {"skepticism": 0.75, "digital_literacy": 0.8},
        },
        {
            "id": "vulnerable_1",
            "label": "Low Literacy Elder",
            "description": "Struggles with OTPs",
            "share_of_population": 0.5,
            "traits": {"skepticism": 0.2, "digital_literacy": 0.15},
        },
        {
            "id": "vulnerable_2",
            "label": "Rushed Novice",
            "description": "Clicks without checking",
            "share_of_population": 0.5,
            "traits": {"skepticism": 0.3, "digital_literacy": 0.25},
        },
    ]

    result = validate_and_normalize_archetypes(raw_dicts)
    assert result is not None
    assert len(result) == 4
    total_share = sum(a.share_of_population for a in result)
    assert total_share == pytest.approx(1.0, abs=1e-4)
    # Check trait clamping and default filling
    for a in result:
        assert set(a.traits.keys()) == set(TRAITS)
        assert all(0.0 <= v <= 1.0 for v in a.traits.values())


def test_validate_and_normalize_fails_insufficient_diversity():
    """Fails if there are fewer than 2 skeptical or fewer than 2 vulnerable segments."""
    # Only 1 skeptic, 1 vulnerable
    raw_dicts = [
        {
            "id": "skep_only",
            "label": "One Skeptic",
            "description": "Desc",
            "traits": {"skepticism": 0.8, "digital_literacy": 0.8},
        },
        {
            "id": "vuln_only",
            "label": "One Vulnerable",
            "description": "Desc",
            "traits": {"skepticism": 0.3, "digital_literacy": 0.2},
        },
        {
            "id": "mid_1",
            "label": "Average",
            "description": "Desc",
            "traits": {"skepticism": 0.5, "digital_literacy": 0.5},
        },
        {
            "id": "mid_2",
            "label": "Average",
            "description": "Desc",
            "traits": {"skepticism": 0.5, "digital_literacy": 0.5},
        },
    ]
    assert validate_and_normalize_archetypes(raw_dicts) is None


# ---------------------------------------------------------------------------
# 4. LLM Generation, Mocking, and Caching
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_archetypes_cache():
    """Ensure cache is reset between test cases."""
    clear_cache("archetypes")
    yield
    clear_cache("archetypes")


@patch("backend.archetypes.query_ollama")
def test_generate_archetypes_success_and_caching(mock_query):
    """Successful LLM generation caches results by description hash and reuses them."""
    mock_data = {
        "archetypes": [
            {
                "id": "tech_skeptic_1",
                "label": "Tech Skeptic 1",
                "description": "Desc",
                "share_of_population": 0.3,
                "traits": {"skepticism": 0.85, "digital_literacy": 0.9},
            },
            {
                "id": "tech_skeptic_2",
                "label": "Tech Skeptic 2",
                "description": "Desc",
                "share_of_population": 0.2,
                "traits": {"skepticism": 0.75, "digital_literacy": 0.8},
            },
            {
                "id": "novice_user_1",
                "label": "Novice User 1",
                "description": "Desc",
                "share_of_population": 0.25,
                "traits": {"skepticism": 0.3, "digital_literacy": 0.2},
            },
            {
                "id": "novice_user_2",
                "label": "Novice User 2",
                "description": "Desc",
                "share_of_population": 0.25,
                "traits": {"skepticism": 0.2, "digital_literacy": 0.15},
            },
        ]
    }
    mock_query.return_value = mock_data

    desc = "Custom tech startup employees in Gurgaon"
    res1 = generate_archetypes(desc, n=4)
    assert len(res1) == 4
    assert mock_query.call_count == 1
    assert sum(a.share_of_population for a in res1) == pytest.approx(1.0)

    # Calling again with identical description should hit the cache without calling mock_query
    res2 = generate_archetypes(desc, n=4)
    assert len(res2) == 4
    assert mock_query.call_count == 1  # Unchanged!
    assert [a.id for a in res1] == [a.id for a in res2]


@patch("backend.archetypes.query_ollama")
def test_generate_archetypes_retry_then_success(mock_query):
    """When the first call is invalid, it retries and succeeds on the second attempt."""
    invalid_data = {
        "archetypes": [
            {"id": "a1", "label": "L", "description": "D", "traits": {"skepticism": 0.1, "digital_literacy": 0.9}}
        ]
    }
    valid_data = {
        "archetypes": [
            {"id": "s1", "label": "S1", "description": "D", "share_of_population": 0.25, "traits": {"skepticism": 0.9, "digital_literacy": 0.7}},
            {"id": "s2", "label": "S2", "description": "D", "share_of_population": 0.25, "traits": {"skepticism": 0.8, "digital_literacy": 0.7}},
            {"id": "v1", "label": "V1", "description": "D", "share_of_population": 0.25, "traits": {"skepticism": 0.3, "digital_literacy": 0.1}},
            {"id": "v2", "label": "V2", "description": "D", "share_of_population": 0.25, "traits": {"skepticism": 0.2, "digital_literacy": 0.2}},
        ]
    }
    mock_query.side_effect = [invalid_data, valid_data]

    res = generate_archetypes("Auto rickshaw drivers in Mumbai", n=4)
    assert len(res) == 4
    assert mock_query.call_count == 2
    assert res[0].id == "s1"


@patch("backend.archetypes.query_ollama")
def test_generate_archetypes_fallback_to_closest_preset_after_retries(mock_query):
    """If all 3 attempts fail (or raise exceptions), it falls back to the closest preset."""
    mock_query.side_effect = Exception("Ollama connection timeout")

    res = generate_archetypes("Elderly retired grandfathers in South Delhi", n=10)
    assert mock_query.call_count == 3  # Initial + 2 retries
    assert len(res) == 10
    # Should fall back to urban_seniors
    assert res[0].id in [a.id for a in PRESETS["urban_seniors"]]


# ---------------------------------------------------------------------------
# 5. FastAPI Endpoints for Presets and Archetype Generation
# ---------------------------------------------------------------------------

def test_api_get_presets():
    """GET /presets returns all presets and their archetypes."""
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)
    response = client.get("/presets")
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == set(EXPECTED_PRESETS)
    assert len(data["urban_seniors"]) == 10


@patch("backend.archetypes.query_ollama")
def test_api_generate_archetypes(mock_query):
    """POST /archetypes/generate creates or falls back to segments."""
    from fastapi.testclient import TestClient
    from backend.main import app

    mock_query.return_value = {
        "archetypes": [
            {"id": "s1", "label": "S1", "description": "D", "share_of_population": 0.25, "traits": {"skepticism": 0.9, "digital_literacy": 0.7}},
            {"id": "s2", "label": "S2", "description": "D", "share_of_population": 0.25, "traits": {"skepticism": 0.8, "digital_literacy": 0.7}},
            {"id": "v1", "label": "V1", "description": "D", "share_of_population": 0.25, "traits": {"skepticism": 0.3, "digital_literacy": 0.1}},
            {"id": "v2", "label": "V2", "description": "D", "share_of_population": 0.25, "traits": {"skepticism": 0.2, "digital_literacy": 0.2}},
        ]
    }
    client = TestClient(app)
    response = client.post(
        "/archetypes/generate",
        json={"audience_description": "Custom test group in Pune", "n": 4},
    )
    assert response.status_code == 200
    payload = response.json()
    assert "archetypes" in payload
    assert len(payload["archetypes"]) == 4
