"""
tests/test_scoring_insights.py  -  Persona
==========================================
Unit test suite for scoring.py and insights.py:
1. R0 calculation with synthetic edges and share counts (edge cases: 0 shares, no edges, multi-exposures).
2. Time-to-peak wave identification under different wave velocity patterns.
3. Verdict evaluation across all 4 tiers using threshold cuts on reach and R0 for both harmful and beneficial packs.
4. Granular demographic breakdowns (archetype, age bands <25, 25-40, 41-60, 60+, area, language) with act-rate and exposure-rate.
5. Weakest group identification (highest act_rate for harmful, lowest for beneficial).
6. Strongest trigger identification among actors vs general reached population.
7. Defensive suggestions matrix: keyed by trigger x group, deduplicated to 3 tips, safety assurance (never scam text).
8. Beneficial improvement suggestions for positive campaigns.
9. Comparative Monte Carlo evaluations (compare / compare_interventions) across N=20 seeds with mean, p10-p90, winner, and relative difference.
"""

from typing import Dict, List
import pytest

from backend.config import DEFAULT_R0_THRESHOLDS, DEFAULT_REACH_THRESHOLDS
from backend.domain_packs import CIVIC_POLICY, HEALTH_CAMPAIGN, SCAM_FRAUD, verdict_for
from backend.insights import (
    DEFENSIVE_TEMPLATES,
    compare,
    compare_interventions,
    generate_beneficial_tips,
    generate_defensive_suggestions,
    generate_strategic_insights,
)
from backend.models import Archetype, MessageFeatures, Person, Reaction, WaveState
from backend.scoring import (
    compute_group_stats,
    compute_r0,
    compute_time_to_peak,
    find_strongest_trigger,
    find_weakest_group,
    get_age_band,
    score_cascade,
)


# ---------------------------------------------------------------------------
# 1. R0 and Time-to-Peak Tests
# ---------------------------------------------------------------------------

def test_compute_r0_zero_shares():
    """R0 must be 0.0 when there are zero shares or empty transmission edges."""
    assert compute_r0([], 0) == 0.0
    assert compute_r0([{"source": "p1", "target": "p2"}], 0) == 0.0
    assert compute_r0([], 5) == 0.0


def test_compute_r0_synthetic_cases():
    """R0 must be len(edges) / total_shares rounded to 2 decimal places."""
    edges = [
        {"source": "p0", "target": "p1"},
        {"source": "p0", "target": "p2"},
        {"source": "p1", "target": "p3"},
    ]
    # 3 transmissions among 2 sharers -> R0 = 1.50
    assert compute_r0(edges, 2) == 1.50

    # 3 transmissions among 4 sharers -> R0 = 0.75
    assert compute_r0(edges, 4) == 0.75

    # 10 transmissions among 3 sharers -> R0 = 3.33
    synthetic_10 = [{"source": f"s{i}", "target": f"t{i}"} for i in range(10)]
    assert compute_r0(synthetic_10, 3) == 3.33


def test_compute_time_to_peak():
    """Identify the wave index with maximum newly_reached count."""
    assert compute_time_to_peak([]) == 0

    waves_single = [
        WaveState(wave=0, newly_reached=10, cumulative_reached=10, acts=2, shares=1, reports=0, reached_ids=["p0"])
    ]
    assert compute_time_to_peak(waves_single) == 0

    # Velocity peaks at wave 2
    waves_multi = [
        WaveState(wave=0, newly_reached=5, cumulative_reached=5, acts=1, shares=1, reports=0, reached_ids=[]),
        WaveState(wave=1, newly_reached=12, cumulative_reached=17, acts=3, shares=2, reports=0, reached_ids=[]),
        WaveState(wave=2, newly_reached=25, cumulative_reached=42, acts=6, shares=4, reports=1, reached_ids=[]),
        WaveState(wave=3, newly_reached=8, cumulative_reached=50, acts=1, shares=0, reports=0, reached_ids=[]),
    ]
    assert compute_time_to_peak(waves_multi) == 2


# ---------------------------------------------------------------------------
# 2. Verdict Thresholds Tests
# ---------------------------------------------------------------------------

def test_verdict_thresholds_harmful_pack():
    """Verify verdict labels and tiers for scam_fraud based on reach and R0 thresholds."""
    pack = SCAM_FRAUD
    assert pack.verdict_labels == ["Contained", "Niche hit", "Wide spread", "Outbreak"]

    # Tier 0: reach < 0.10 and r0 < 0.70
    v0 = verdict_for(pack, reach=0.05, r0=0.50)
    assert v0["tier"] == 0
    assert v0["label"] == "Contained"

    # Tier 1: reach >= 0.10 or r0 >= 0.70
    v1_reach = verdict_for(pack, reach=0.15, r0=0.40)
    assert v1_reach["tier"] == 1
    assert v1_reach["label"] == "Niche hit"

    v1_r0 = verdict_for(pack, reach=0.08, r0=0.85)
    assert v1_r0["tier"] == 1
    assert v1_r0["label"] == "Niche hit"

    # Tier 2: reach >= 0.30 or r0 >= 1.00
    v2_reach = verdict_for(pack, reach=0.35, r0=0.60)
    assert v2_reach["tier"] == 2
    assert v2_reach["label"] == "Wide spread"

    v2_r0 = verdict_for(pack, reach=0.15, r0=1.10)
    assert v2_r0["tier"] == 2
    assert v2_r0["label"] == "Wide spread"

    # Tier 3: reach >= 0.55 or r0 >= 1.40
    v3_reach = verdict_for(pack, reach=0.60, r0=0.80)
    assert v3_reach["tier"] == 3
    assert v3_reach["label"] == "Outbreak"

    v3_r0 = verdict_for(pack, reach=0.20, r0=1.55)
    assert v3_r0["tier"] == 3
    assert v3_r0["label"] == "Outbreak"


def test_verdict_thresholds_beneficial_pack():
    """Verify beneficial verdict labels (Ignored, Niche reach, Strong reach, Breakout)."""
    pack = HEALTH_CAMPAIGN
    assert pack.verdict_labels == ["Ignored", "Niche reach", "Strong reach", "Breakout"]

    # HEALTH_CAMPAIGN thresholds: reach=[0.15, 0.35, 0.60], r0=[0.7, 1.0, 1.3]
    assert verdict_for(pack, reach=0.05, r0=0.20)["label"] == "Ignored"
    assert verdict_for(pack, reach=0.18, r0=0.30)["label"] == "Niche reach"
    assert verdict_for(pack, reach=0.40, r0=0.80)["label"] == "Strong reach"
    assert verdict_for(pack, reach=0.65, r0=1.45)["label"] == "Breakout"


# ---------------------------------------------------------------------------
# 3. Demographic Group Stats & Age Bands
# ---------------------------------------------------------------------------

def test_get_age_band():
    """Verify standardized age band classifications."""
    assert get_age_band(18) == "<25"
    assert get_age_band(24) == "<25"
    assert get_age_band(25) == "25-40"
    assert get_age_band(40) == "25-40"
    assert get_age_band(41) == "41-60"
    assert get_age_band(60) == "41-60"
    assert get_age_band(61) == "60+"
    assert get_age_band(85) == "60+"


def test_compute_group_stats():
    """Test group stats computation across archetype, age band, area, and language."""
    people = [
        Person(id="p1", name="Ramesh", archetype_id="pensioners", age=68, language="hindi", area="urban_metro", traits={}),
        Person(id="p2", name="Suresh", archetype_id="pensioners", age=72, language="hindi", area="urban_metro", traits={}),
        Person(id="p3", name="Ananya", archetype_id="students", age=21, language="english", area="urban_metro", traits={}),
        Person(id="p4", name="Sunita", archetype_id="homemakers", age=35, language="hindi", area="tier2_city", traits={}),
    ]

    waves = [
        WaveState(wave=0, newly_reached=3, cumulative_reached=3, acts=1, shares=1, reports=0, reached_ids=["p1", "p2", "p3"]),
    ]
    node_states = {
        "p1": "act",
        "p2": "ignore",
        "p3": "share",
        # p4 is unexposed
    }

    stats = compute_group_stats(people, waves, node_states)

    assert "by_archetype" in stats
    assert "by_age_band" in stats
    assert "by_area" in stats
    assert "by_language" in stats

    # Check pensioners archetype
    pensioner_stats = stats["by_archetype"]["pensioners"]
    assert pensioner_stats["count"] == 2
    assert pensioner_stats["reached"] == 2
    assert pensioner_stats["acts"] == 1
    assert pensioner_stats["shares"] == 0
    assert pensioner_stats["exposure_rate"] == 1.0
    assert pensioner_stats["act_rate"] == 0.5

    # Check age band 60+ (p1 and p2)
    seniors_stats = stats["by_age_band"]["60+"]
    assert seniors_stats["count"] == 2
    assert seniors_stats["reached"] == 2
    assert seniors_stats["acts"] == 1
    assert seniors_stats["act_rate"] == 0.5

    # Check age band <25 (p3)
    youth_stats = stats["by_age_band"]["<25"]
    assert youth_stats["count"] == 1
    assert youth_stats["reached"] == 1
    assert youth_stats["shares"] == 1
    assert youth_stats["acts"] == 0
    assert youth_stats["act_rate"] == 0.0

    # Check unexposed p4 in tier2_city
    tier2_stats = stats["by_area"]["tier2_city"]
    assert tier2_stats["count"] == 1
    assert tier2_stats["reached"] == 0
    assert tier2_stats["acts"] == 0
    assert tier2_stats["exposure_rate"] == 0.0
    assert tier2_stats["act_rate"] == 0.0


# ---------------------------------------------------------------------------
# 4. Weakest Group & Strongest Trigger
# ---------------------------------------------------------------------------

def test_find_weakest_group_harmful():
    """In harmful packs, weakest group is the one with highest act_rate."""
    group_stats = {
        "by_archetype": {
            "skeptics": {"count": 10, "reached": 10, "acts": 1, "act_rate": 0.10},
            "pensioners": {"count": 10, "reached": 10, "acts": 6, "act_rate": 0.60},
            "students": {"count": 10, "reached": 10, "acts": 3, "act_rate": 0.30},
        }
    }
    weakest = find_weakest_group(group_stats, valence="harmful")
    assert weakest["group"] == "pensioners"
    assert weakest["act_rate"] == 0.60


def test_find_weakest_group_beneficial():
    """In beneficial packs, weakest group is the one with lowest act_rate (needs outreach)."""
    group_stats = {
        "by_archetype": {
            "active_parents": {"count": 10, "reached": 10, "acts": 8, "act_rate": 0.80},
            "busy_workers": {"count": 10, "reached": 10, "acts": 2, "act_rate": 0.20},
        }
    }
    weakest = find_weakest_group(group_stats, valence="beneficial")
    assert weakest["group"] == "busy_workers"
    assert weakest["act_rate"] == 0.20


def test_find_strongest_trigger_actors_priority():
    """Strongest trigger must prioritize actors over passive observers."""
    people = [
        Person(id="p1", name="A", archetype_id="arch_a", age=30, language="en", area="metro", traits={}),
        Person(id="p2", name="B", archetype_id="arch_b", age=40, language="en", area="metro", traits={}),
        Person(id="p3", name="C", archetype_id="arch_c", age=50, language="en", area="metro", traits={}),
    ]
    reactions = {
        "p1": Reaction(archetype_id="arch_a", ignore=0.1, notice=0.9, engage=0.8, act=0.7, share=0.1, report=0.0, top_trigger="urgency", reason="r"),
        "p2": Reaction(archetype_id="arch_b", ignore=0.1, notice=0.9, engage=0.8, act=0.7, share=0.1, report=0.0, top_trigger="urgency", reason="r"),
        "p3": Reaction(archetype_id="arch_c", ignore=0.1, notice=0.9, engage=0.5, act=0.1, share=0.1, report=0.0, top_trigger="curiosity", reason="r"),
    }
    node_states = {
        "p1": "act",
        "p2": "act",
        "p3": "ignore",
    }
    top = find_strongest_trigger(people, reactions, node_states)
    assert top == "urgency"


# ---------------------------------------------------------------------------
# 5. Defensive & Beneficial Insights
# ---------------------------------------------------------------------------

def test_generate_defensive_suggestions_structure():
    """Harmful packs must generate exactly 3 defensive suggestions and never scam text."""
    pack = SCAM_FRAUD
    features = MessageFeatures(
        is_suspicious=True,
        urgency=0.9,
        emotional_pull=0.8,
        authority_claim="bank",
        call_to_action=0.9,
        has_link=True,
        language="hi",
        channel_guess="whatsapp",
        red_flags=["fake link"],
        legit_signals=[],
        topics=["finance"],
    )
    weakest_group = {"group": "urban_seniors", "act_rate": 0.55}

    suggestions = generate_defensive_suggestions(
        pack=pack,
        strongest_trigger="urgency",
        weakest_group=weakest_group,
        features=features,
    )

    assert len(suggestions) == 3
    # Check that seniors + urgency matched
    assert any("call-back verify" in s.lower() or "cool-down" in s.lower() for s in suggestions)
    # Check that language recommendation or link defense is included
    assert any("regional" in s.lower() or "domain" in s.lower() or "friction" in s.lower() for s in suggestions)
    # Crucial safety rule: none of the suggestions contain scam rewriting
    for s in suggestions:
        assert "click here" not in s.lower()
        assert "otp" not in s.lower() or "never" in s.lower() or "pin" in s.lower()


def test_generate_beneficial_tips():
    """Beneficial packs should provide actionable improvement tips."""
    pack = HEALTH_CAMPAIGN
    features = MessageFeatures(
        is_suspicious=False,
        urgency=0.2,
        emotional_pull=0.4,
        authority_claim="hospital",
        call_to_action=0.3,
        has_link=False,
        language="en",
        channel_guess="whatsapp",
        red_flags=[],
        legit_signals=["official health info"],
        topics=["vaccine"],
    )
    tips = generate_beneficial_tips(pack, features)
    assert len(tips) == 3
    assert any("call-to-action" in t.lower() or "register" in t.lower() for t in tips)


def test_generate_strategic_insights_wrapper():
    """Verify strategic insights payload format."""
    pack = SCAM_FRAUD
    features = MessageFeatures(
        is_suspicious=True,
        urgency=0.8,
        emotional_pull=0.7,
        authority_claim="police",
        call_to_action=0.6,
        has_link=False,
        language="en",
        channel_guess="sms",
        red_flags=["threat"],
        legit_signals=[],
        topics=["legal"],
    )
    from backend.models import SimulationResult
    sim_res = SimulationResult(waves=[], edges=[], totals={}, group_stats={}, verdict="Contained", verdict_tier=0)
    weakest = {"group": "students", "act_rate": 0.4}

    insights = generate_strategic_insights(
        simulation_result=sim_res,
        pack=pack,
        strongest_trigger="authority",
        weakest_group=weakest,
        features=features,
    )

    assert insights["mode"] == "defensive"
    assert insights["valence"] == "harmful"
    assert len(insights["recommendations"]) == 3
    assert insights["allow_message_improvement"] is False


# ---------------------------------------------------------------------------
# 6. Comparative Monte Carlo Simulation (N=20)
# ---------------------------------------------------------------------------

def test_compare_monte_carlo_runs():
    """Run compare across N=20 seeds and verify statistical output format."""
    archetypes = [
        Archetype(
            id="vulnerable_seniors",
            label="Vulnerable Seniors",
            description="Elderly citizens with high trust",
            share_of_population=0.5,
            traits={
                "attention": 0.6,
                "digital_literacy": 0.2,
                "trust_in_source": 0.8,
                "skepticism": 0.2,
                "emotional_reactivity": 0.7,
                "social_sharing": 0.5,
                "time_pressure": 0.6,
                "domain_interest": 0.5,
                "financial_cushion": 0.4,
                "language_barrier": 0.2,
            },
        ),
        Archetype(
            id="tech_skeptics",
            label="Tech Skeptics",
            description="Digitally savvy critical thinkers",
            share_of_population=0.5,
            traits={
                "attention": 0.8,
                "digital_literacy": 0.85,
                "trust_in_source": 0.2,
                "skepticism": 0.85,
                "emotional_reactivity": 0.2,
                "social_sharing": 0.2,
                "time_pressure": 0.3,
                "domain_interest": 0.5,
                "financial_cushion": 0.7,
                "language_barrier": 0.1,
            },
        ),
    ]

    msg = "Urgent: Your electricity bill is unpaid. Power will be disconnected in 2 hours. Pay immediately."
    inv_weak = "Be careful with payment messages."
    inv_strong = "Official advisory from State Electricity Board: DISCOM never demands immediate payment via messaging apps or threatens instant disconnection. Verify bill status only at https://discom.gov.in."

    res = compare(
        a=inv_weak,
        b=inv_strong,
        message=msg,
        pack=SCAM_FRAUD,
        archetypes=archetypes,
        n_runs=20,
        n_people=30,  # Fast test size
        channel="whatsapp",
        base_seed=123,
    )

    assert res["runs"] == 20
    assert "candidate_a_stats" in res
    assert "candidate_b_stats" in res
    stats_a = res["candidate_a_stats"]["acts"]
    stats_b = res["candidate_b_stats"]["acts"]

    # Verify distribution percentiles exist
    for k in ("mean", "std", "p10", "p50", "p90"):
        assert k in stats_a
        assert k in stats_b

    # Verify winner determination
    assert res["winner"] in ("Intervention A", "Intervention B", "Candidate A", "Candidate B", "Tie")
    assert isinstance(res["relative_difference_percent"], float)
    assert "methodology_note" in res
    assert "identical synthetic population" in res["methodology_note"]
