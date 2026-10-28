"""
tests/test_reactions.py  -  Persona
===================================
Test suite for reactions.py:
1. Rule baseline extraction (Step B).
2. Hybrid blending and ordering constraints (Step C).
3. Caching by hash(archetype set + message + pack).
4. Individual personalisation (Step D).
5. Mandatory Sanity Tests:
   - Scam sample gives higher act-rate for low-literacy segments than skeptical ones.
   - Legit sample produces low act rate overall.
"""

import pytest

from backend.domain_packs import SCAM_FRAUD, CIVIC_POLICY, HEALTH_CAMPAIGN
from backend.models import Archetype, MessageFeatures, Person, Reaction
from backend.reactions import (
    blend_reactions,
    compute_reactions_cache_key,
    extract_rule_reactions,
    get_reactions,
    personalise_for_population,
    personalise_reaction,
)


@pytest.fixture
def sample_archetypes() -> list[Archetype]:
    """Provide a diverse pair of archetypes: vulnerable senior vs tech skeptic."""
    vulnerable = Archetype(
        id="vulnerable_senior",
        label="Vulnerable Senior",
        description="Elderly pensioner with low digital literacy and high trust in institutions.",
        share_of_population=0.5,
        traits={
            "attention": 0.60,
            "digital_literacy": 0.20,
            "trust_in_source": 0.85,
            "skepticism": 0.15,
            "emotional_reactivity": 0.80,
            "social_sharing": 0.45,
            "time_pressure": 0.70,
            "domain_interest": 0.50,
            "financial_cushion": 0.30,
            "language_barrier": 0.30,
        },
    )
    skeptic = Archetype(
        id="tech_skeptic",
        label="Tech Skeptic",
        description="Digital professional who verifies sources and questions unusual requests.",
        share_of_population=0.5,
        traits={
            "attention": 0.70,
            "digital_literacy": 0.85,
            "trust_in_source": 0.25,
            "skepticism": 0.90,
            "emotional_reactivity": 0.20,
            "social_sharing": 0.20,
            "time_pressure": 0.30,
            "domain_interest": 0.50,
            "financial_cushion": 0.75,
            "language_barrier": 0.05,
        },
    )
    return [vulnerable, skeptic]


@pytest.fixture
def scam_features() -> MessageFeatures:
    """Features representing an urgent scam message."""
    return MessageFeatures(
        is_suspicious=True,
        urgency=0.90,
        emotional_pull=0.85,
        authority_claim="bank",
        call_to_action=0.90,
        has_link=True,
        language="en",
        channel_guess="sms",
        red_flags=["fake_kyc", "immediate_threat"],
        legit_signals=[],
        topics=["banking", "security"],
    )


@pytest.fixture
def legit_features() -> MessageFeatures:
    """Features representing a benign low-stakes civic message."""
    return MessageFeatures(
        is_suspicious=False,
        urgency=0.10,
        emotional_pull=0.10,
        authority_claim="govt",
        call_to_action=0.10,
        has_link=False,
        language="en",
        channel_guess="sms",
        red_flags=[],
        legit_signals=["official_gov"],
        topics=["civic"],
    )


# ---------------------------------------------------------------------------
# 1. Rule Baseline Extraction (Step B)
# ---------------------------------------------------------------------------

def test_extract_rule_reactions_ordering(sample_archetypes, scam_features):
    """Rule baseline must enforce act <= engage <= notice for every archetype."""
    msg = "URGENT: Your account will be closed in 24 hours. Verify now."
    reactions = extract_rule_reactions(scam_features, msg, sample_archetypes, SCAM_FRAUD)

    assert len(reactions) == len(sample_archetypes)
    for aid, r in reactions.items():
        assert 0.0 <= r.act <= 1.0
        assert 0.0 <= r.engage <= 1.0
        assert 0.0 <= r.notice <= 1.0
        assert r.act <= r.engage <= r.notice, f"Failed for {aid}: {r.act} <= {r.engage} <= {r.notice}"
        assert len(r.reason.split()) <= 15
        assert r.top_trigger in ("urgency", "authority", "fear", "reward", "curiosity", "social_proof", "none")


# ---------------------------------------------------------------------------
# 2. Hybrid Blending (Step C)
# ---------------------------------------------------------------------------

def test_blend_reactions_math(sample_archetypes, scam_features):
    """blend_reactions must compute 0.6 * LLM + 0.4 * Rules for continuous probabilities."""
    llm_r = {
        "vulnerable_senior": Reaction(
            archetype_id="vulnerable_senior",
            ignore=0.10,
            notice=0.90,
            engage=0.70,
            act=0.50,
            share=0.40,
            report=0.05,
            top_trigger="authority",
            reason="Trusts bank claims easily.",
        )
    }
    rule_r = {
        "vulnerable_senior": Reaction(
            archetype_id="vulnerable_senior",
            ignore=0.20,
            notice=0.80,
            engage=0.50,
            act=0.30,
            share=0.20,
            report=0.05,
            top_trigger="fear",
            reason="Rule baseline reaction.",
        )
    }

    blended = blend_reactions(llm_r, rule_r, scam_features)
    res = blended["vulnerable_senior"]

    # Notice: 0.6*0.90 + 0.4*0.80 = 0.54 + 0.32 = 0.86
    assert pytest.approx(res.notice, 0.001) == 0.86
    # Engage: 0.6*0.70 + 0.4*0.50 = 0.42 + 0.20 = 0.62
    assert pytest.approx(res.engage, 0.001) == 0.62
    # Act: 0.6*0.50 + 0.4*0.30 = 0.30 + 0.12 = 0.42
    assert pytest.approx(res.act, 0.001) == 0.42
    # Share: 0.6*0.40 + 0.4*0.20 = 0.24 + 0.08 = 0.32
    assert pytest.approx(res.share, 0.001) == 0.32

    # Ordering holds
    assert res.act <= res.engage <= res.notice


def test_blend_reactions_legit_suppression(sample_archetypes, legit_features):
    """Legitimate low-stakes messages must suppress act rate across all segments."""
    rule_r = extract_rule_reactions(legit_features, "Water maintenance tomorrow.", sample_archetypes, CIVIC_POLICY)
    blended = blend_reactions(None, rule_r, legit_features)

    for aid, r in blended.items():
        assert r.act <= 0.12
        assert r.report <= 0.05


# ---------------------------------------------------------------------------
# 3. Caching
# ---------------------------------------------------------------------------

def test_reactions_caching(sample_archetypes, scam_features):
    """get_reactions should cache results and return identical models from cache."""
    msg = "Test caching message string for verification."
    res1 = get_reactions(scam_features, msg, sample_archetypes, SCAM_FRAUD, k=1, use_cache=True)
    res2 = get_reactions(scam_features, msg, sample_archetypes, SCAM_FRAUD, k=1, use_cache=True)

    assert set(res1.keys()) == set(res2.keys())
    for aid in res1:
        assert res1[aid].act == res2[aid].act
        assert res1[aid].notice == res2[aid].notice
        assert res1[aid].reason == res2[aid].reason


# ---------------------------------------------------------------------------
# 4. Individual Personalisation (Step D)
# ---------------------------------------------------------------------------

def test_personalise_reaction(sample_archetypes):
    """Personalised probability must respond to stress, skepticism, and trust deltas."""
    arch = sample_archetypes[0]  # vulnerable senior
    arch_reaction = Reaction(
        archetype_id=arch.id,
        ignore=0.20,
        notice=0.80,
        engage=0.50,
        act=0.35,
        share=0.20,
        report=0.05,
        top_trigger="authority",
        reason="Base archetype reaction",
    )

    # Person with higher stress (+0.15) and higher trust (+0.10), lower skepticism (-0.10)
    person_high_risk = Person(
        id="person_stressed",
        name="Ramesh Kumar",
        archetype_id=arch.id,
        traits={
            "emotional_reactivity": arch.traits["emotional_reactivity"] + 0.15,
            "trust_in_source": arch.traits["trust_in_source"] + 0.10,
            "skepticism": max(0.0, arch.traits["skepticism"] - 0.10),
        },
    )

    # Person with higher skepticism (+0.30)
    person_skeptical = Person(
        id="person_cautious",
        name="Suresh Verma",
        archetype_id=arch.id,
        traits={
            "emotional_reactivity": arch.traits["emotional_reactivity"],
            "trust_in_source": arch.traits["trust_in_source"],
            "skepticism": min(1.0, arch.traits["skepticism"] + 0.30),
        },
    )

    r_high = personalise_reaction(person_high_risk, arch, arch_reaction, seed=42)
    r_cautious = personalise_reaction(person_skeptical, arch, arch_reaction, seed=42)

    # Stressed/trusting person should have higher act probability than cautious person
    assert r_high.act > r_cautious.act
    assert r_high.act <= r_high.engage <= r_high.notice
    assert r_cautious.act <= r_cautious.engage <= r_cautious.notice


def test_personalise_for_population(sample_archetypes):
    """Batch personalisation must assign reactions for all people in population."""
    arch = sample_archetypes[0]
    arch_r = {
        arch.id: Reaction(
            archetype_id=arch.id,
            ignore=0.25,
            notice=0.75,
            engage=0.45,
            act=0.30,
            share=0.15,
            report=0.05,
            top_trigger="authority",
            reason="Sample",
        )
    }
    people = [
        Person(id=f"p_{i}", name=f"Person {i}", archetype_id=arch.id, traits=arch.traits.copy())
        for i in range(10)
    ]

    pop_reactions = personalise_for_population(people, sample_archetypes, arch_r, seed=123)
    assert len(pop_reactions) == 10
    for pid in pop_reactions:
        r = pop_reactions[pid]
        assert r.act <= r.engage <= r.notice


# ---------------------------------------------------------------------------
# 5. Mandatory Sanity Tests
# ---------------------------------------------------------------------------

def test_sanity_scam_vulnerable_vs_skeptical(sample_archetypes, scam_features):
    """Sanity Test: Scam sample must give higher act-rate for low-literacy segments than skeptical ones."""
    scam_msg = (
        "URGENT: Your bank account is BLOCKED due to pending KYC. "
        "Click http://bank-update.top/kyc to complete verification immediately!"
    )
    reactions = get_reactions(scam_features, scam_msg, sample_archetypes, SCAM_FRAUD, k=1)

    r_vulnerable = reactions["vulnerable_senior"]
    r_skeptic = reactions["tech_skeptic"]

    # Low-literacy, high-trust segment must have noticeably higher act rate than skeptic
    assert r_vulnerable.act > r_skeptic.act, (
        f"Expected vulnerable act ({r_vulnerable.act}) > skeptical act ({r_skeptic.act})"
    )
    # Skeptic act rate should be low
    assert r_skeptic.act < 0.20, f"Skeptic act rate too high: {r_skeptic.act}"
    # Vulnerable act rate should be elevated (at least 1.5x skeptic)
    assert r_vulnerable.act > r_skeptic.act * 1.5, (
        f"Expected vulnerable act to be > 1.5x skeptic: {r_vulnerable.act} vs {r_skeptic.act}"
    )
    assert r_vulnerable.act >= 0.20, f"Vulnerable act rate too low: {r_vulnerable.act}"


def test_sanity_legit_sample_produces_low_act_overall(sample_archetypes, legit_features):
    """Sanity Test: Legit sample must produce low act rate overall."""
    legit_msg = (
        "Municipal advisory: Routine water pipeline maintenance scheduled for Ward 4 tomorrow "
        "from 10 AM to 2 PM. Please store sufficient water."
    )
    reactions = get_reactions(legit_features, legit_msg, sample_archetypes, CIVIC_POLICY, k=1)

    act_rates = [r.act for r in reactions.values()]
    avg_act = sum(act_rates) / len(act_rates)

    # Legit sample must produce low act overall
    assert all(act < 0.20 for act in act_rates), f"An archetype had unexpectedly high act: {act_rates}"
    assert avg_act < 0.12, f"Average act rate too high for legit message: {avg_act}"
