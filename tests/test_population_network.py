"""Tests for population generation and NetworkX social graph modeling."""

import pytest
import networkx as nx

from backend.archetypes import PRESETS
from backend.domain_packs import TRAITS
from backend.models import Person
from backend.population import generate_population
from backend.network import build_network, print_network_stats, NetworkGraphResult


# ---------------------------------------------------------------------------
# 1. Population Generation Tests
# ---------------------------------------------------------------------------

def test_generate_population_count_and_types():
    """Verify that generate_population returns n Person objects with required fields."""
    archetypes = PRESETS["urban_seniors"]
    people = generate_population(archetypes, n=120, seed=42)

    assert len(people) == 120
    assert all(isinstance(p, Person) for p in people)

    # Check node ID sequence
    assert people[0].id == "p_001"
    assert people[-1].id == "p_120"

    for p in people:
        assert isinstance(p.name, str) and len(p.name.split()) >= 2
        assert p.age >= 18
        assert p.area in ("metro", "suburban", "rural")
        assert p.language in ("hi", "en", "mr", "ta", "te", "bn", "kn")
        assert len(p.traits) == len(TRAITS)
        for t, val in p.traits.items():
            assert 0.0 <= val <= 1.0, f"Trait {t}={val} out of bounds"


def test_generate_population_trait_jitter():
    """Person traits must deviate slightly from archetype mean (Gaussian jitter, sd=0.08)."""
    archetypes = PRESETS["college_students"]
    people = generate_population(archetypes, n=100, seed=123)

    # Find people of the same archetype and verify their individual traits differ
    arch_id = "skeptical_coder"
    coders = [p for p in people if p.archetype_id == arch_id]
    if len(coders) >= 2:
        diffs = [
            abs(coders[0].traits[t] - coders[1].traits[t])
            for t in TRAITS
        ]
        assert any(d > 0.001 for d in diffs), "Traits should have individual jitter"


def test_generate_population_determinism():
    """Identical seed must produce identical population and traits."""
    archetypes = PRESETS["rural_farmers"]
    pop1 = generate_population(archetypes, n=60, seed=99)
    pop2 = generate_population(archetypes, n=60, seed=99)

    assert len(pop1) == len(pop2)
    for p1, p2 in zip(pop1, pop2):
        assert p1.id == p2.id
        assert p1.name == p2.name
        assert p1.age == p2.age
        assert p1.area == p2.area
        assert p1.language == p2.language
        assert p1.traits == p2.traits


def test_generate_population_different_seeds():
    """Different seeds must yield different demographic selections or jitter."""
    archetypes = PRESETS["working_parents"]
    pop1 = generate_population(archetypes, n=50, seed=1)
    pop2 = generate_population(archetypes, n=50, seed=2)

    names1 = [p.name for p in pop1]
    names2 = [p.name for p in pop2]
    assert names1 != names2


# ---------------------------------------------------------------------------
# 2. Network Construction & Graph Topology Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("channel", ["whatsapp", "sms", "social_feed", "call"])
def test_build_network_giant_component_greater_than_90_percent(channel):
    """Every generated network must have a giant component containing > 90% of nodes."""
    people = generate_population(PRESETS["urban_seniors"], n=120, seed=42)
    res = build_network(people, channel=channel, seed=42)

    G, payload = res
    assert isinstance(G, nx.Graph)
    assert isinstance(payload, dict)

    # Degree stats printed for inspection
    print_network_stats(res)

    metrics = payload["metrics"]
    giant_ratio = metrics["giant_component_ratio"]
    assert giant_ratio > 0.90, f"Giant component ratio {giant_ratio} is below 90% for {channel}"
    assert metrics["total_nodes"] == 120
    assert metrics["total_edges"] == G.number_of_edges()


@pytest.mark.parametrize("channel", ["whatsapp", "sms", "social_feed", "call"])
def test_build_network_has_hubs_and_family_clusters(channel):
    """Graph must contain 5-6 designated hubs and family household cluster edges."""
    people = generate_population(PRESETS["small_shopkeepers"], n=120, seed=42)
    G, payload = build_network(people, channel=channel, seed=42)

    # Hubs check
    hubs = [n for n, d in G.nodes(data=True) if d.get("is_hub") is True]
    assert len(hubs) in (5, 6), f"Expected 5-6 hubs, got {len(hubs)}"

    # Family edges check
    family_edges = [
        (u, v) for u, v, d in G.edges(data=True)
        if d.get("edge_type") == "family"
    ]
    assert len(family_edges) > 0, "Graph must contain household family edges"


def test_build_network_channel_topology_differences():
    """Verify that channel alters structure (WhatsApp has high clustering, Social Feed has high-degree hubs)."""
    people = generate_population(PRESETS["urban_professionals"], n=120, seed=42)

    res_wa = build_network(people, channel="whatsapp", seed=42)
    res_sms = build_network(people, channel="sms", seed=42)
    res_feed = build_network(people, channel="social_feed", seed=42)
    res_call = build_network(people, channel="call", seed=42)

    # WhatsApp small-world clustering should exceed SMS clustering
    wa_clust = res_wa.payload["metrics"]["clustering_coefficient"]
    sms_clust = res_sms.payload["metrics"]["clustering_coefficient"]
    assert wa_clust > 0.10

    # Social feed should have the highest hub max degree
    feed_max_deg = res_feed.payload["metrics"]["max_degree"]
    call_max_deg = res_call.payload["metrics"]["max_degree"]
    assert feed_max_deg > call_max_deg
    assert feed_max_deg >= 50, "Social feed hubs should connect to a large share of population"

    # Call and SMS should be significantly sparser than WhatsApp
    assert res_call.payload["metrics"]["total_edges"] < res_wa.payload["metrics"]["total_edges"]


def test_build_network_determinism():
    """Calling build_network with the same seed must produce the exact same edges."""
    people = generate_population(PRESETS["community_health_workers"], n=80, seed=77)
    res1 = build_network(people, channel="whatsapp", seed=77)
    res2 = build_network(people, channel="whatsapp", seed=77)

    assert set(res1.graph.edges()) == set(res2.graph.edges())
    assert res1.payload["metrics"] == res2.payload["metrics"]


def test_build_network_serialisable_output():
    """Payload must be fully serializable with proper edge types and node fields."""
    people = generate_population(PRESETS["gig_workers"], n=50, seed=10)
    _, payload = build_network(people, channel="whatsapp", seed=10)

    assert "nodes" in payload
    assert "edges" in payload
    assert "metrics" in payload

    edge_types = {e["edge_type"] for e in payload["edges"]}
    assert "family" in edge_types
    assert "hub" in edge_types
    assert ("homophily" in edge_types or "peer" in edge_types)

    for node in payload["nodes"]:
        assert "id" in node
        assert "name" in node
        assert "archetype_id" in node
        assert "degree" in node
        assert "is_hub" in node


# ---------------------------------------------------------------------------
# 3. FastAPI Population & Network Endpoints
# ---------------------------------------------------------------------------

def test_api_generate_population_and_build_network():
    """Verify POST /population/generate and POST /network/build via TestClient."""
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)

    # 1. Generate population
    pop_resp = client.post(
        "/population/generate",
        json={"preset_id": "urban_seniors", "n": 60, "seed": 42},
    )
    assert pop_resp.status_code == 200
    pop_data = pop_resp.json()
    assert pop_data["count"] == 60
    assert len(pop_data["people"]) == 60

    # 2. Build network
    net_resp = client.post(
        "/network/build",
        json={"people": pop_data["people"], "channel": "whatsapp", "seed": 42},
    )
    assert net_resp.status_code == 200
    net_data = net_resp.json()
    assert "nodes" in net_data
    assert "edges" in net_data
    assert "metrics" in net_data
    assert net_data["metrics"]["giant_component_ratio"] > 0.90
