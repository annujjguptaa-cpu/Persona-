"""
insights.py  -  Persona
======================
Generates human-readable strategic insights, defensive policy recommendations,
and comparative Monte Carlo evaluations.

Features:
- Defensive suggestions for harmful packs from template matrix keyed by trigger x group.
- Optimization tips for beneficial packs (messaging clarity, localized tone).
- Strict safety guardrail: NEVER generates or improves scam/misinformation text.
- compare(a, b): Runs each intervention/message N=20 times with different seeds,
  returning mean, p10-p90 range, winner, and relative difference percentage.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from backend.domain_packs import PACKS, SCAM_FRAUD, DomainPack
from backend.engine import run_simulation
from backend.models import Archetype, MessageFeatures, Person, Reaction, SimulationResult
from backend.network import build_network
from backend.population import generate_population
from backend.reactions import get_reactions

logger = logging.getLogger(__name__)

# Template matrix for defensive suggestions keyed by (trigger, demographic/context)
DEFENSIVE_TEMPLATES: Dict[Tuple[str, str], str] = {
    ("urgency", "seniors"): (
        "Add a call-back verify step with trusted family contacts or an automated 15-minute "
        "cool-down before high-value payment authorizations."
    ),
    ("urgency", "students"): (
        "Display explicit in-app warning banners explaining that legitimate recruiters "
        "and exam boards never set arbitrary 2-hour payment countdowns."
    ),
    ("urgency", "any"): (
        "Introduce cognitive friction at payment checkout: require users to re-type the recipient's "
        "name and state before transferring funds."
    ),
    ("authority", "seniors"): (
        "Deploy targeted public warnings clarifying that RBI, police, and tax officials NEVER "
        "issue digital arrests, video summons, or demand clearing fees over messaging apps."
    ),
    ("authority", "shopkeepers"): (
        "Equip merchant UPI soundboxes with clear voice confirmation: 'Receiving payments never "
        "requires scanning a QR code or entering your UPI PIN.'"
    ),
    ("authority", "any"): (
        "Display certified institutional badges and prominent links to verify notices directly on "
        "official .gov.in and .nic.in portals."
    ),
    ("fear", "seniors"): (
        "Provide immediate 1-click access to the National Cybercrime Reporting Portal helpline (1930) "
        "with regional language operators."
    ),
    ("fear", "any"): (
        "Provide direct hotlines and fact-checking bots that verify account suspension threats in seconds."
    ),
    ("reward", "gig_workers"): (
        "Broadcast localized SMS alerts illustrating typical cashback bait schemes: legitimate rewards "
        "are credited automatically and never require an upfront processing deposit."
    ),
    ("reward", "any"): (
        "Flag lottery and cashback offers that demand upfront processing fees with automated browser warnings."
    ),
    ("social_proof", "any"): (
        "Add visual forwarding friction ('Forwarded many times') paired with verified fact-check cards "
        "in active family and neighborhood group chats."
    ),
    ("curiosity", "any"): (
        "Integrate automated domain URL previews that flag unverified top-level domains (.top, .xyz, .cc) "
        "before navigation."
    ),
    ("language_barrier", "any"): (
        "Issue official defense advisories in local regional languages and vernacular voice notes, "
        "bridging the accessibility gap for non-English speakers."
    ),
}

BENEFICIAL_IMPROVEMENT_TIPS: List[str] = [
    "Position the direct call-to-action (clinic address, helpline, or registration link) in the first two sentences.",
    "Localize messaging into regional languages with conversational, community-centered tone to bridge comprehension barriers.",
    "Lower the perceived cost of action by explicitly emphasizing 'free', 'walk-in', or 'no appointment necessary'.",
    "Leverage trusted community intermediaries (ASHA workers, local group coordinators) as the credited message source.",
    "Add social endorsement ('Over 10,000 neighborhood families already enrolled') to encourage peer forwarding.",
]


def generate_defensive_suggestions(
    pack: DomainPack,
    strongest_trigger: str,
    weakest_group: Dict[str, Any],
    features: MessageFeatures,
) -> List[str]:
    """Generate 3 defensive recommendations tailored to the observed trigger and vulnerable group."""
    trigger = str(strongest_trigger).lower()
    group_name = str(weakest_group.get("group", "")).lower()

    # Determine demographic archetype category
    group_category = "any"
    if any(k in group_name for k in ("senior", "elder", "pensioner")):
        group_category = "seniors"
    elif any(k in group_name for k in ("student", "youth", "college")):
        group_category = "students"
    elif any(k in group_name for k in ("shopkeeper", "trader", "merchant")):
        group_category = "shopkeepers"
    elif any(k in group_name for k in ("gig", "wage", "driver", "rider")):
        group_category = "gig_workers"

    suggestions: List[str] = []

    # 1. Primary trigger x group match
    key1 = (trigger, group_category)
    if key1 in DEFENSIVE_TEMPLATES:
        suggestions.append(DEFENSIVE_TEMPLATES[key1])
    elif (trigger, "any") in DEFENSIVE_TEMPLATES:
        suggestions.append(DEFENSIVE_TEMPLATES[(trigger, "any")])

    # 2. Language or authority defense
    if getattr(features, "language", "en") != "en" or any("language" in k for k in group_name):
        suggestions.append(DEFENSIVE_TEMPLATES[("language_barrier", "any")])
    elif features.authority_score > 0.40 and ("authority", "any") not in suggestions:
        suggestions.append(DEFENSIVE_TEMPLATES[("authority", "any")])
    elif ("fear", "any") not in suggestions:
        suggestions.append(DEFENSIVE_TEMPLATES[("fear", "any")])

    # 3. Channel / forwarding friction or verification
    if features.has_link and ("curiosity", "any") not in suggestions:
        suggestions.append(DEFENSIVE_TEMPLATES[("curiosity", "any")])
    elif ("social_proof", "any") not in suggestions:
        suggestions.append(DEFENSIVE_TEMPLATES[("social_proof", "any")])
    else:
        suggestions.append(DEFENSIVE_TEMPLATES[("urgency", "any")])

    # Ensure deduplicated and exactly 3 suggestions
    unique_suggestions = list(dict.fromkeys(suggestions))
    while len(unique_suggestions) < 3:
        for tpl in DEFENSIVE_TEMPLATES.values():
            if tpl not in unique_suggestions:
                unique_suggestions.append(tpl)
            if len(unique_suggestions) == 3:
                break

    return unique_suggestions[:3]


def generate_beneficial_tips(pack: DomainPack, features: MessageFeatures) -> List[str]:
    """Provide constructive suggestions to maximize reach and conversion for positive campaigns."""
    tips = list(BENEFICIAL_IMPROVEMENT_TIPS)
    if features.call_to_action < 0.5:
        tips.insert(0, "Strengthen the direct call-to-action with clear verbs like 'Register today' or 'Visit your nearest clinic'.")
    if getattr(features, "language", "en") == "en":
        tips.insert(1, "Translate and broadcast into regional languages to drastically increase rural and suburban reach.")
    return tips[:3]


def generate_strategic_insights(
    simulation_result: SimulationResult,
    pack: DomainPack,
    strongest_trigger: str,
    weakest_group: Dict[str, Any],
    features: MessageFeatures,
) -> Dict[str, Any]:
    """Produce comprehensive defense and strategy guidance based on simulation cascade."""
    if pack.valence == "harmful":
        recommendations = generate_defensive_suggestions(
            pack=pack,
            strongest_trigger=strongest_trigger,
            weakest_group=weakest_group,
            features=features,
        )
        mode = "defensive"
    else:
        recommendations = generate_beneficial_tips(pack=pack, features=features)
        mode = "improvement"

    return {
        "mode": mode,
        "valence": pack.valence,
        "strongest_trigger": strongest_trigger,
        "weakest_group": weakest_group,
        "recommendations": recommendations,
        "allow_message_improvement": pack.allow_message_improvement,
    }


# ---------------------------------------------------------------------------
# Comparative Monte Carlo Evaluation (N=20 runs)
# ---------------------------------------------------------------------------

def compare_interventions(
    message: str,
    pack: Union[DomainPack, str],
    archetypes: Optional[List[Archetype]] = None,
    interventions: Optional[List[Optional[str]]] = None,
    n_runs: int = 20,
    n_people: int = 60,
    channel: str = "whatsapp",
    base_seed: int = 42,
    messages: Optional[List[str]] = None,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """Run comparative Monte Carlo simulations across two interventions or two messages.

    Runs each candidate N=20 times with distinct seeds, computing:
    - mean, standard deviation, and p10-p90 percentile range.
    - winner selection with relative difference percentage.
    - explicitly notes that projections are relative under identical assumptions.
    """
    from backend.archetypes import PRESETS
    from backend.analyzer import analyze_message

    domain_pack = PACKS.get(pack, SCAM_FRAUD) if isinstance(pack, str) else pack

    if archetypes is None:
        archetypes = PRESETS["urban_seniors"]

    # Determine comparison mode: comparing two messages OR comparing two interventions
    comparing_messages = bool(messages and len(messages) >= 2)

    if comparing_messages:
        msg_a = messages[0]
        msg_b = messages[1]
        inv_a = interventions[0] if (interventions and len(interventions) > 0) else None
        inv_b = interventions[1] if (interventions and len(interventions) > 1) else None

        features_a = analyze_message(msg_a, pack_id=domain_pack.id, use_llm=use_llm)
        reactions_a = get_reactions(features_a, msg_a, archetypes, domain_pack, k=1, use_cache=True, use_llm=use_llm)

        features_b = analyze_message(msg_b, pack_id=domain_pack.id, use_llm=use_llm)
        reactions_b = get_reactions(features_b, msg_b, archetypes, domain_pack, k=1, use_cache=True, use_llm=use_llm)
    else:
        if not interventions or len(interventions) < 2:
            raise ValueError("Must provide at least two interventions (e.g. [A, B]) or two messages to compare.")
        msg_a = message
        msg_b = message
        inv_a = interventions[0]
        inv_b = interventions[1]

        features_a = analyze_message(message, pack_id=domain_pack.id, use_llm=use_llm)
        reactions_a = get_reactions(features_a, message, archetypes, domain_pack, k=1, use_cache=True, use_llm=use_llm)
        features_b = features_a
        reactions_b = reactions_a

    results_a: List[Dict[str, float]] = []
    results_b: List[Dict[str, float]] = []

    for i in range(n_runs):
        run_seed = (base_seed + i * 17) % (2**31 - 1)
        people = generate_population(archetypes, n=n_people, seed=run_seed)
        net_res = build_network(people, channel=channel, seed=run_seed)

        # Simulation with Candidate A
        sim_a = run_simulation(
            people=people,
            graph=net_res.graph,
            reactions=reactions_a,
            features=features_a,
            pack=domain_pack,
            intervention=inv_a,
            seed=run_seed,
        )
        results_a.append({
            "acts": float(sim_a.totals.get("total_acts", 0)),
            "reached": float(sim_a.totals.get("cumulative_reached", 0)),
            "r0": float(sim_a.totals.get("r0", 0.0)),
        })

        # Simulation with Candidate B
        sim_b = run_simulation(
            people=people,
            graph=net_res.graph,
            reactions=reactions_b,
            features=features_b,
            pack=domain_pack,
            intervention=inv_b,
            seed=run_seed,
        )
        results_b.append({
            "acts": float(sim_b.totals.get("total_acts", 0)),
            "reached": float(sim_b.totals.get("cumulative_reached", 0)),
            "r0": float(sim_b.totals.get("r0", 0.0)),
        })

    def calc_stats(run_data: List[Dict[str, float]], metric_key: str) -> Dict[str, float]:
        vals = np.array([item[metric_key] for item in run_data], dtype=float)
        return {
            "mean": round(float(np.mean(vals)), 2),
            "std": round(float(np.std(vals)), 2),
            "p10": round(float(np.percentile(vals, 10)), 2),
            "p50": round(float(np.median(vals)), 2),
            "p90": round(float(np.percentile(vals, 90)), 2),
        }

    stats_a = {
        "acts": calc_stats(results_a, "acts"),
        "reached": calc_stats(results_a, "reached"),
        "r0": calc_stats(results_a, "r0"),
    }
    stats_b = {
        "acts": calc_stats(results_b, "acts"),
        "reached": calc_stats(results_b, "reached"),
        "r0": calc_stats(results_b, "r0"),
    }

    # Determine Winner
    mean_acts_a = stats_a["acts"]["mean"]
    mean_acts_b = stats_b["acts"]["mean"]

    label_a = "Candidate A" if not comparing_messages else "Message A"
    label_b = "Candidate B" if not comparing_messages else "Message B"
    if not comparing_messages and inv_a is not None:
        label_a = "Intervention A"
        label_b = "Intervention B"

    if domain_pack.valence == "harmful":
        # For harmful scam/fraud packs, lower victims/acts is better
        if mean_acts_a < mean_acts_b:
            winner = label_a
            diff = mean_acts_b - mean_acts_a
            rel_diff = round((diff / max(1e-4, mean_acts_b)) * 100, 1)
        elif mean_acts_b < mean_acts_a:
            winner = label_b
            diff = mean_acts_a - mean_acts_b
            rel_diff = round((diff / max(1e-4, mean_acts_a)) * 100, 1)
        else:
            winner = "Tie"
            rel_diff = 0.0
    else:
        # For beneficial health/civic packs, higher reach and action is better
        if mean_acts_a > mean_acts_b:
            winner = label_a
            diff = mean_acts_a - mean_acts_b
            rel_diff = round((diff / max(1e-4, mean_acts_b)) * 100, 1)
        elif mean_acts_b > mean_acts_a:
            winner = label_b
            diff = mean_acts_b - mean_acts_a
            rel_diff = round((diff / max(1e-4, mean_acts_a)) * 100, 1)
        else:
            winner = "Tie"
            rel_diff = 0.0

    return {
        "runs": n_runs,
        "pack_id": domain_pack.id,
        "valence": domain_pack.valence,
        "comparison_mode": "messages" if comparing_messages else "interventions",
        "candidate_a": msg_a if comparing_messages else inv_a,
        "candidate_b": msg_b if comparing_messages else inv_b,
        "candidate_a_stats": stats_a,
        "candidate_b_stats": stats_b,
        "winner": winner,
        "relative_difference_percent": rel_diff,
        "methodology_note": (
            "Results are relative projections evaluated under identical synthetic population, "
            "NetworkX topology, and behavioral trait assumptions across 20 randomized seeds."
        ),
    }


def compare(
    a: Optional[str] = None,
    b: Optional[str] = None,
    *,
    message: Optional[str] = None,
    interventions: Optional[List[Optional[str]]] = None,
    messages: Optional[List[str]] = None,
    pack: Union[DomainPack, str] = SCAM_FRAUD,
    archetypes: Optional[List[Archetype]] = None,
    n_runs: int = 20,
    n_people: int = 60,
    channel: str = "whatsapp",
    base_seed: int = 42,
    mode: str = "intervention",
    use_llm: bool = True,
) -> Dict[str, Any]:
    """Convenience wrapper for compare(a, b) as requested in specification."""
    if a is not None and b is not None:
        if mode == "message" or message is None:
            messages = [a, b]
            msg = a
            invs = None
        else:
            invs = [a, b]
            msg = message
    else:
        msg = message or "Default message"
        invs = interventions

    return compare_interventions(
        message=msg,
        pack=pack,
        archetypes=archetypes,
        interventions=invs,
        messages=messages,
        n_runs=n_runs,
        n_people=n_people,
        channel=channel,
        base_seed=base_seed,
        use_llm=use_llm,
    )
