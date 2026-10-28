"""
tests/test_engine.py  -  Persona
================================
Test suite for engine.py:
1. Intervention analysis and strength scoring.
2. Intervention reduction on person-level act/engage.
3. Diffusion cascade execution across multi-wave NetworkX graphs.
4. Mandatory test requirements:
   - Deterministic with a seed.
   - Zero forwarding on 'call' channel.
   - Stronger warning gives <= victims.
5. Community immunity and early stopping.
"""

import pytest

from backend.archetypes import PRESETS
from backend.domain_packs import SCAM_FRAUD, CIVIC_POLICY
from backend.engine import (
    compute_intervention_reduction,
    evaluate_intervention,
    run_simulation,
)
from backend.models import MessageFeatures, Person, Reaction
from backend.network import build_network
from backend.population import generate_population
from backend.reactions import get_reactions


@pytest.fixture
def test_setup():
    """Build a deterministic test environment with 60 agents and WhatsApp network."""
    archetypes = PRESETS["urban_seniors"]
    people = generate_population(archetypes, n=60, seed=42)
    net_res = build_network(people, channel="whatsapp", seed=42)

    scam_features = MessageFeatures(
        is_suspicious=True,
        urgency=0.90,
        emotional_pull=0.85,
        authority_claim="bank",
        call_to_action=0.85,
        has_link=True,
        language="en",
        channel_guess="whatsapp",
    )
    scam_msg = (
        "URGENT: Your SBI account has been suspended! "
        "Click http://sbi-unblock.top/kyc now or your funds will be seized in 24 hours!"
    )
    reactions = get_reactions(scam_features, scam_msg, archetypes, SCAM_FRAUD, k=1, use_cache=True)

    return {
        "people": people,
        "graph": net_res.graph,
        "reactions": reactions,
        "features": scam_features,
        "msg": scam_msg,
    }


# ---------------------------------------------------------------------------
# 1. Intervention Evaluation Tests
# ---------------------------------------------------------------------------

def test_evaluate_intervention_strength():
    """Strong official warning with verification step must score high strength."""
    # Strong intervention: Clear official source, verification steps, helpline, plain language
    strong_inv = (
        "OFFICIAL WARNING from RBI and Cyber Crime Department: "
        "Banks NEVER request OTP or PIN over SMS. Never click unverified links. "
        "Verify your account status by visiting your nearest branch or call the national cyber helpline 1930."
    )
    strength_strong = evaluate_intervention(strong_inv, message_language="en")
    assert strength_strong >= 0.70, f"Expected high strength for official warning: {strength_strong}"

    # Weak/vague intervention
    weak_inv = "Be careful with messages."
    strength_weak = evaluate_intervention(weak_inv, message_language="en")
    assert strength_weak < strength_strong
    assert strength_weak < 0.40

    # Empty intervention
    assert evaluate_intervention(None) == 0.0
    assert evaluate_intervention("") == 0.0


def test_intervention_reduction_modifiers():
    """Intervention reduction must be higher for trusting agents and lower with language barriers."""
    trusting_person = Person(
        id="p_trust",
        name="Asha Sharma",
        archetype_id="senior",
        traits={"trust_in_source": 0.90, "language_barrier": 0.05},
    )
    skeptical_barrier_person = Person(
        id="p_barrier",
        name="John Doe",
        archetype_id="worker",
        traits={"trust_in_source": 0.20, "language_barrier": 0.80},
    )

    act_trust, _ = compute_intervention_reduction(trusting_person, intervention_strength=0.85)
    act_barrier, _ = compute_intervention_reduction(skeptical_barrier_person, intervention_strength=0.85)

    # Trusting person without language barrier adheres more, so act multiplier is smaller
    assert act_trust < act_barrier, f"Expected act_trust ({act_trust}) < act_barrier ({act_barrier})"


# ---------------------------------------------------------------------------
# 2. Mandatory Test: Deterministic with a Seed
# ---------------------------------------------------------------------------

def test_simulation_deterministic(test_setup):
    """Running simulation twice with the exact same seed must produce identical outputs."""
    p = test_setup["people"]
    g = test_setup["graph"]
    r = test_setup["reactions"]
    f = test_setup["features"]

    res1 = run_simulation(p, g, r, f, SCAM_FRAUD, seed=42)
    res2 = run_simulation(p, g, r, f, SCAM_FRAUD, seed=42)

    assert res1.totals["cumulative_reached"] == res2.totals["cumulative_reached"]
    assert res1.totals["total_acts"] == res2.totals["total_acts"]
    assert res1.totals["total_shares"] == res2.totals["total_shares"]
    assert res1.totals["total_reports"] == res2.totals["total_reports"]
    assert res1.totals["r0"] == res2.totals["r0"]
    assert len(res1.waves) == len(res2.waves)
    assert len(res1.edges) == len(res2.edges)


# ---------------------------------------------------------------------------
# 3. Mandatory Test: Zero Forwarding on 'Call' Channel
# ---------------------------------------------------------------------------

def test_simulation_zero_forwarding_on_call(test_setup):
    """On the 'call' channel, forwarding probability is zero so only Wave 0 seeds are exposed."""
    p = test_setup["people"]
    r = test_setup["reactions"]
    f = test_setup["features"]

    # Build call network
    call_net = build_network(p, channel="call", seed=42)
    res = run_simulation(p, call_net.graph, r, f, SCAM_FRAUD, seed=42)

    # Zero forwarding transmission edges
    assert len(res.edges) == 0
    # Exactly one wave executed (Wave 0 seed)
    assert len(res.waves) == 1
    assert res.totals["waves_completed"] == 1
    # Only the initial seed nodes reached (approx 10% of 60 = 6)
    assert res.totals["cumulative_reached"] <= 10


# ---------------------------------------------------------------------------
# 4. Mandatory Test: Stronger Warning Gives <= Victims
# ---------------------------------------------------------------------------

def test_simulation_stronger_warning_reduces_victims(test_setup):
    """Simulating with a strong official intervention must yield <= victims than with no warning."""
    p = test_setup["people"]
    g = test_setup["graph"]
    r = test_setup["reactions"]
    f = test_setup["features"]

    # 1. Baseline simulation without intervention
    res_no_warning = run_simulation(p, g, r, f, SCAM_FRAUD, intervention=None, seed=99)

    # 2. Simulation with strong verified warning
    strong_warning = (
        "OFFICIAL WARNING from RBI and Cyber Crime Cell: "
        "This is an urgent phishing scam. NEVER enter your bank OTP or credentials online. "
        "Verify your account status by calling your bank branch or the cyber helpline 1930."
    )
    res_strong_warning = run_simulation(p, g, r, f, SCAM_FRAUD, intervention=strong_warning, seed=99)

    # 3. Simulation with weak vague warning
    weak_warning = "Be careful."
    res_weak_warning = run_simulation(p, g, r, f, SCAM_FRAUD, intervention=weak_warning, seed=99)

    victims_none = res_no_warning.totals["total_acts"]
    victims_weak = res_weak_warning.totals["total_acts"]
    victims_strong = res_strong_warning.totals["total_acts"]

    # Strong warning must give <= victims than no warning and <= weak warning
    assert victims_strong <= victims_none, (
        f"Strong warning victims ({victims_strong}) should be <= no warning ({victims_none})"
    )
    assert victims_strong <= victims_weak, (
        f"Strong warning victims ({victims_strong}) should be <= weak warning ({victims_weak})"
    )


# ---------------------------------------------------------------------------
# 5. Multi-Wave Mechanics & Group Statistics
# ---------------------------------------------------------------------------

def test_simulation_wave_progression(test_setup):
    """Simulation progresses across waves, respects early stopping and tracks group stats."""
    p = test_setup["people"]
    g = test_setup["graph"]
    r = test_setup["reactions"]
    f = test_setup["features"]

    res = run_simulation(p, g, r, f, SCAM_FRAUD, seed=42)

    assert 1 <= len(res.waves) <= 6
    assert res.totals["cumulative_reached"] > 0
    assert len(res.group_stats) > 0

    # Cumulative reach must be monotonically non-decreasing
    reaches = [w.cumulative_reached for w in res.waves]
    for i in range(1, len(reaches)):
        assert reaches[i] >= reaches[i - 1]

    # Verdict and insights populated
    assert res.verdict != ""
    assert len(res.insights) >= 2
