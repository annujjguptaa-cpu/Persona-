"""
scoring.py  -  Persona
======================
Calculates cascade reach, virality metrics (R0, time-to-peak), demographic
breakdowns, vulnerability rankings, and verdict classifications.

Metrics:
- reach %, act count, share count, report count, R0 (new exposures per sharer),
  waves reached, time-to-peak.
- Group stats by: archetype, age band (<25, 25-40, 41-60, 60+), area, language,
  with act-rate and exposure-rate.
- Verdict from the pack's verdict_labels using thresholds on reach and R0.
- weakest_group and strongest_trigger (most common top_trigger among actors).
"""

from __future__ import annotations

from collections import Counter
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from backend.config import DEFAULT_R0_THRESHOLDS, DEFAULT_REACH_THRESHOLDS
from backend.domain_packs import PACKS, SCAM_FRAUD, DomainPack, verdict_for
from backend.models import Person, Reaction, WaveState

logger = logging.getLogger(__name__)


def compute_r0(edges: List[Dict[str, Any]], total_shares: int) -> float:
    """Calculate R0 (basic reproduction number): new transmissions per sharer.

    Args:
        edges: List of transmission edges traversed during diffusion.
        total_shares: Total count of agents who shared/forwarded.

    Returns:
        Float R0 rounded to 2 decimal places (0.0 if no shares).
    """
    if total_shares <= 0 or not edges:
        return 0.0
    return round(len(edges) / float(total_shares), 2)


def compute_time_to_peak(waves: List[WaveState]) -> int:
    """Identify the wave index (0-indexed) with the peak newly reached exposure velocity.

    Args:
        waves: Chronological list of WaveState objects.

    Returns:
        Wave index of peak exposure velocity (0 if empty or single wave).
    """
    if not waves:
        return 0

    max_newly_reached = -1
    peak_wave = 0
    for w in waves:
        if w.newly_reached > max_newly_reached:
            max_newly_reached = w.newly_reached
            peak_wave = w.wave

    return peak_wave


def get_age_band(age: int) -> str:
    """Categorize age into standardized demographic bands."""
    if age < 25:
        return "<25"
    if age <= 40:
        return "25-40"
    if age <= 60:
        return "41-60"
    return "60+"


def compute_group_stats(
    people: List[Person],
    waves: List[WaveState],
    node_states: Dict[str, str],
) -> Dict[str, Dict[str, Any]]:
    """Compute granular breakdowns across archetype, age band, area, and language.

    Each subgroup calculates:
    - count: total members in population
    - reached: members exposed during simulation
    - acts: members who took action ('act')
    - shares: members who forwarded ('share')
    - reports: members who flagged ('report')
    - exposure_rate: reached / count
    - act_rate: acts / max(1, reached)
    """
    exposed_ids: Set[str] = set()
    for w in waves:
        exposed_ids.update(w.reached_ids)

    categories: Dict[str, Dict[str, Dict[str, Any]]] = {
        "by_archetype": {},
        "by_age_band": {},
        "by_area": {},
        "by_language": {},
    }

    def init_bucket() -> Dict[str, Any]:
        return {
            "count": 0,
            "reached": 0,
            "acts": 0,
            "shares": 0,
            "reports": 0,
            "exposure_rate": 0.0,
            "act_rate": 0.0,
        }

    for p in people:
        is_reached = p.id in exposed_ids
        st = node_states.get(p.id, "ignore") if is_reached else "unexposed"

        # 1. Archetype
        arch_bucket = categories["by_archetype"].setdefault(p.archetype_id, init_bucket())
        arch_bucket["count"] += 1
        if is_reached:
            arch_bucket["reached"] += 1
            if st == "act":
                arch_bucket["acts"] += 1
            elif st == "share":
                arch_bucket["shares"] += 1
            elif st == "report":
                arch_bucket["reports"] += 1

        # 2. Age band
        age_band = get_age_band(p.age)
        age_bucket = categories["by_age_band"].setdefault(age_band, init_bucket())
        age_bucket["count"] += 1
        if is_reached:
            age_bucket["reached"] += 1
            if st == "act":
                age_bucket["acts"] += 1
            elif st == "share":
                age_bucket["shares"] += 1
            elif st == "report":
                age_bucket["reports"] += 1

        # 3. Area
        area_key = str(p.area).lower()
        area_bucket = categories["by_area"].setdefault(area_key, init_bucket())
        area_bucket["count"] += 1
        if is_reached:
            area_bucket["reached"] += 1
            if st == "act":
                area_bucket["acts"] += 1
            elif st == "share":
                area_bucket["shares"] += 1
            elif st == "report":
                area_bucket["reports"] += 1

        # 4. Language
        lang_key = str(p.language).lower()
        lang_bucket = categories["by_language"].setdefault(lang_key, init_bucket())
        lang_bucket["count"] += 1
        if is_reached:
            lang_bucket["reached"] += 1
            if st == "act":
                lang_bucket["acts"] += 1
            elif st == "share":
                lang_bucket["shares"] += 1
            elif st == "report":
                lang_bucket["reports"] += 1

    # Finalize rates
    for cat_name, cat_dict in categories.items():
        for key, b in cat_dict.items():
            b["exposure_rate"] = round(b["reached"] / max(1, b["count"]), 3)
            b["act_rate"] = round(b["acts"] / max(1, b["reached"]), 3)

    return categories


def find_weakest_group(
    group_stats: Dict[str, Dict[str, Any]],
    valence: str = "harmful",
) -> Dict[str, Any]:
    """Identify the most vulnerable subgroup.

    For harmful packs (scams): highest act_rate (most exploited).
    For beneficial packs (health/civic): lowest act_rate or exposure_rate (needs outreach).
    """
    by_arch = group_stats.get("by_archetype", {})
    if not by_arch:
        return {"group": "unknown", "act_rate": 0.0, "reason": "No data"}

    if valence == "harmful":
        # Highest act_rate among reached
        candidates = [
            (aid, data["act_rate"], data["acts"], data["reached"])
            for aid, data in by_arch.items()
            if data["reached"] > 0
        ]
        if not candidates:
            first_aid = next(iter(by_arch.keys()))
            return {"group": first_aid, "act_rate": 0.0, "reason": "Zero reach"}
        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
        top = candidates[0]
        return {
            "group": top[0],
            "act_rate": top[1],
            "acts": top[2],
            "reached": top[3],
            "category": "archetype",
            "reason": f"Highest victim conversion rate ({top[1] * 100:.1f}%) among exposed",
        }
    else:
        # Lowest act_rate among reached
        candidates = [
            (aid, data["act_rate"], data["acts"], data["reached"])
            for aid, data in by_arch.items()
            if data["reached"] > 0
        ]
        if not candidates:
            first_aid = next(iter(by_arch.keys()))
            return {"group": first_aid, "act_rate": 0.0, "reason": "Zero reach"}
        candidates.sort(key=lambda x: x[1])
        top = candidates[0]
        return {
            "group": top[0],
            "act_rate": top[1],
            "acts": top[2],
            "reached": top[3],
            "category": "archetype",
            "reason": f"Lowest engagement rate ({top[1] * 100:.1f}%) requiring targeted campaign outreach",
        }


def find_strongest_trigger(
    people: List[Person],
    reactions: Dict[str, Reaction],
    node_states: Dict[str, str],
) -> str:
    """Find the most prevalent top_trigger among agents who took action ('act').

    Falls back to most prevalent trigger among all reached agents if none acted.
    """
    actor_triggers: List[str] = []
    reached_triggers: List[str] = []

    for p in people:
        st = node_states.get(p.id)
        if st is None:
            continue

        # Look up trigger from personalized reaction or archetype reaction
        r = reactions.get(p.id) or reactions.get(p.archetype_id)
        trigger = r.top_trigger if r and r.top_trigger not in ("none", "") else None

        if trigger:
            reached_triggers.append(trigger)
            if st == "act":
                actor_triggers.append(trigger)

    if actor_triggers:
        return Counter(actor_triggers).most_common(1)[0][0]
    if reached_triggers:
        return Counter(reached_triggers).most_common(1)[0][0]

    return "none"


def score_cascade(
    people: List[Person],
    waves: List[WaveState],
    edges: List[Dict[str, Any]],
    reactions: Dict[str, Reaction],
    pack: Union[DomainPack, str],
    node_states: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Compile comprehensive scoring payload from simulation trajectory."""
    domain_pack = PACKS.get(pack, SCAM_FRAUD) if isinstance(pack, str) else pack

    n_people = len(people)
    all_exposed: Set[str] = set()
    for w in waves:
        all_exposed.update(w.reached_ids)

    reach_count = len(all_exposed)
    reach_fraction = round(reach_count / max(1, n_people), 4)

    total_acts = sum(w.acts for w in waves)
    total_shares = sum(w.shares for w in waves)
    total_reports = sum(w.reports for w in waves)

    r0 = compute_r0(edges, total_shares)
    time_to_peak = compute_time_to_peak(waves)

    # Reconstruct node states if not passed directly
    states_map = node_states or {}
    if not states_map:
        for w in waves:
            for pid in w.reached_ids:
                if pid not in states_map:
                    states_map[pid] = "notice only"

    group_stats = compute_group_stats(people, waves, states_map)
    weakest_group = find_weakest_group(group_stats, valence=domain_pack.valence)
    strongest_trigger = find_strongest_trigger(people, reactions, states_map)

    # Verdict evaluation
    v_res = verdict_for(domain_pack, reach=reach_fraction, r0=r0)
    verdict_label = v_res.get("label", domain_pack.verdict_labels[0])
    verdict_tier = v_res.get("tier", 0)

    return {
        "reach_count": reach_count,
        "reach_fraction": reach_fraction,
        "reach_percent": round(reach_fraction * 100, 2),
        "act_count": total_acts,
        "act_fraction": round(total_acts / max(1, n_people), 4),
        "act_percent": round((total_acts / max(1, n_people)) * 100, 2),
        "share_count": total_shares,
        "report_count": total_reports,
        "r0": r0,
        "waves_reached": len(waves),
        "time_to_peak": time_to_peak,
        "group_stats": group_stats,
        "weakest_group": weakest_group,
        "strongest_trigger": strongest_trigger,
        "verdict": verdict_label,
        "verdict_tier": verdict_tier,
    }
