"""
engine.py  -  Persona
=====================
Multi-wave social graph diffusion simulator tracking propagation, state paths,
and cascade dynamics across synthetic audience populations.

Features:
- Wave 0 Seeding: Seeds 10% of people; hubs 50% likely depending on channel.
- State Path Sampling: Each newly exposed node samples exactly one state path:
  ignore | notice only | engage | act | share | report.
- Transmission Decay: Sharers propagate to unexposed neighbors with probability
  decaying by wave (0.85^wave) and modulated by edge type:
  family: 1.0, hub: 0.8, peer/neighbour: 0.6.
- Zero Forwarding on 'call' channel.
- Community Immunity: Reporters flag messages, reducing neighbor engage/act
  by 0.05 in subsequent waves.
- Intervention Dynamics: Warnings/clarifications analysed with analyzer.py
  and converted to a strength score (0-1) that mitigates engage/act proportionally
  to trust and language barrier.
- Early Stopping: Halts when cascade naturally terminates or reaches MAX_WAVES (6).
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np

from backend.config import MAX_WAVES, SEED
from backend.domain_packs import PACKS, SCAM_FRAUD, DomainPack, verdict_for
from backend.models import (
    Archetype,
    MessageFeatures,
    Person,
    Reaction,
    SimulationResult,
    WaveState,
)
from backend.reactions import personalise_reaction

logger = logging.getLogger(__name__)

# Primary state paths that an exposed individual can sample
STATE_PATHS = ["ignore", "notice only", "engage", "act", "share", "report"]


def evaluate_intervention(
    intervention: Optional[str],
    message_language: str = "en",
) -> float:
    """Analyze a warning or clarification text and compute an intervention strength in [0.0, 1.0].

    Evaluates:
    1. Clear source named (+0.30): Official institution, RBI, police, fact-checker.
    2. Verification step (+0.25): Concrete helpline, official portal, or verification action.
    3. Plain language (+0.25): Clear length (40-350 chars), readable structure, no shouting.
    4. Local language match (+0.20): Matches message language or Devanagari/Hindi cues.
    """
    if not intervention or not isinstance(intervention, str) or not intervention.strip():
        return 0.0

    text_clean = intervention.strip()
    text_lower = text_clean.lower()

    # 1. Clear source named (0.0 or 0.30)
    source_terms = [
        "rbi", "police", "pib", "fact check", "government", "official",
        "bank", "ministry", "cyber crime", "cert-in", "helpline", "authorized",
        "department", "sbi", "rbi kehta hai", "press information bureau",
    ]
    has_source = any(t in text_lower for t in source_terms)
    source_score = 0.30 if has_source else 0.0

    # 2. Actionable verification step (0.0 or 0.30)
    verify_terms = [
        "verify", "check", "official website", "helpline", "toll free",
        "do not share", "never share", "do not click", "report to", "1930",
        "portal", "confirm", "branch counter", "call your bank",
    ]
    has_verify = any(t in text_lower for t in verify_terms)
    verify_score = 0.30 if has_verify else 0.0

    # 3. Plain language (0.10 or 0.25)
    alphas = [c for c in text_clean if c.isalpha()]
    caps_ratio = sum(1 for c in alphas if c.isupper()) / max(1, len(alphas))
    is_plain = (35 <= len(text_clean) <= 400) and (caps_ratio < 0.22)
    plain_score = 0.25 if is_plain else 0.10

    # 4. Local language match (0.05 or 0.15)
    has_devanagari = bool(re.search(r"[\u0900-\u097F]", text_clean))
    is_hi_msg = message_language.lower() in ("hi", "hindi", "mr")
    lang_match = (is_hi_msg and has_devanagari) or (not is_hi_msg and not has_devanagari)
    lang_score = 0.15 if lang_match else 0.05

    strength = source_score + verify_score + plain_score + lang_score
    return round(max(0.10, min(1.0, strength)), 3)


def compute_intervention_reduction(
    person: Person,
    intervention_strength: float,
) -> Tuple[float, float]:
    """Calculate multiplier reductions for act and engage based on trust and language barrier.

    Reduces engage and act proportionally to trust and inversely to language barrier mismatch.
    Returns:
        (act_multiplier, engage_multiplier) in (0.0, 1.0].
    """
    if intervention_strength <= 0.0:
        return 1.0, 1.0

    trust = person.traits.get("trust_in_source", 0.5)
    lang_barrier = person.traits.get("language_barrier", 0.0)

    # High trust in institutions increases adherence; language barrier impedes understanding
    receptivity = (0.50 + 0.50 * trust) * (1.0 - 0.55 * lang_barrier)
    receptivity = max(0.10, min(1.0, receptivity))

    effective_reduction = intervention_strength * receptivity

    act_multiplier = max(0.05, 1.0 - 0.70 * effective_reduction)
    engage_multiplier = max(0.10, 1.0 - 0.45 * effective_reduction)

    return act_multiplier, engage_multiplier


def run_simulation(
    people: List[Person],
    graph: nx.Graph,
    reactions: Dict[str, Reaction],
    features: MessageFeatures,
    pack: Union[DomainPack, str],
    intervention: Optional[str] = None,
    seed: int = SEED,
    wave_callback: Optional[Callable[[int], None]] = None,
) -> SimulationResult:
    """Run an agent-based multi-wave social graph diffusion cascade simulation.

    Args:
        people: List of Person synthetic agent nodes in the population.
        graph: NetworkX graph connecting people with typed edges.
        reactions: Archetype or person-level Reaction probability profiles.
        features: Extracted MessageFeatures model.
        pack: DomainPack instance or pack ID string.
        intervention: Optional warning or clarification text.
        seed: Random seed for deterministic reproducibility.

    Returns:
        Fully populated SimulationResult model.
    """
    domain_pack = PACKS.get(pack, SCAM_FRAUD) if isinstance(pack, str) else pack
    rng = np.random.RandomState(seed % (2**31 - 1))

    n_people = len(people)
    if n_people == 0:
        return SimulationResult(verdict="No population provided")

    person_by_id: Dict[str, Person] = {p.id: p for p in people}
    node_ids = list(person_by_id.keys())

    # Detect transmission channel
    channel = str(
        graph.graph.get("channel", getattr(features, "channel_guess", "whatsapp"))
    ).lower()

    # Precompute intervention strength if an intervention text is provided
    inv_strength = evaluate_intervention(
        intervention,
        message_language=getattr(features, "language", "en"),
    )

    # Resolve person-level reactions (if archetype reactions supplied, personalize them)
    person_reactions: Dict[str, Reaction] = {}
    archetypes_by_id: Dict[str, Archetype] = {}
    for aid, r in reactions.items():
        # Build lightweight archetype representations if needed
        archetypes_by_id[aid] = Archetype(
            id=aid,
            label=aid,
            description="",
            share_of_population=0.0,
            traits={},
        )

    for i, p in enumerate(people):
        if p.id in reactions:
            person_reactions[p.id] = reactions[p.id]
        elif p.archetype_id in reactions:
            base_r = reactions[p.archetype_id]
            arch_obj = archetypes_by_id.get(p.archetype_id)
            if arch_obj is None:
                arch_obj = Archetype(
                    id=p.archetype_id,
                    label=p.archetype_id,
                    description="",
                    share_of_population=0.0,
                    traits=p.traits,
                )
            p_seed = (seed + i * 43) % (2**31 - 1)
            person_reactions[p.id] = personalise_reaction(
                p, arch_obj, base_r, seed=p_seed
            )
        else:
            # Fallback default reaction
            first_r = next(iter(reactions.values())) if reactions else Reaction(archetype_id=p.id)
            person_reactions[p.id] = first_r

    # -----------------------------------------------------------------------
    # Wave 0 Seeding
    # -----------------------------------------------------------------------
    # Default 10% of population as initial followers
    target_seed_count = max(1, int(round(0.10 * n_people)))

    # Identify central hubs (from node attributes or highest degrees)
    hub_ids: List[str] = [
        pid for pid in node_ids
        if graph.nodes.get(pid, {}).get("is_hub", False)
    ]
    if not hub_ids and len(graph) > 0:
        deg_sorted = sorted(graph.degree(), key=lambda x: x[1], reverse=True)
        hub_ids = [node for node, _ in deg_sorted[: max(3, n_people // 20)]]

    # Hub probability depending on channel
    if channel in ("social_feed", "whatsapp"):
        hub_prob = 0.50
    elif channel == "sms":
        hub_prob = 0.30
    else:  # call
        hub_prob = 0.20

    seed_set: Set[str] = set()
    for hid in hub_ids:
        if rng.random() < hub_prob:
            seed_set.add(hid)

    # Fill remaining seed positions with non-hub followers to satisfy 10%
    non_hubs = [pid for pid in node_ids if pid not in seed_set]
    if len(seed_set) < target_seed_count and non_hubs:
        needed = target_seed_count - len(seed_set)
        chosen = rng.choice(non_hubs, size=min(needed, len(non_hubs)), replace=False)
        seed_set.update(chosen)

    # Simulation State Tracking
    all_exposed: Set[str] = set()
    newly_exposed: List[str] = sorted(list(seed_set))
    immunity_penalties: Dict[str, float] = {pid: 0.0 for pid in node_ids}
    infection_edges: List[Dict[str, Any]] = []
    waves: List[WaveState] = []
    node_sampled_states: Dict[str, str] = {}

    total_shares = 0
    total_acts = 0
    total_reports = 0
    total_ignores = 0
    total_engages = 0
    total_notice_only = 0

    # -----------------------------------------------------------------------
    # Multi-Wave Propagation Loop
    # -----------------------------------------------------------------------
    for wave_idx in range(MAX_WAVES):
        if not newly_exposed:
            break

        if wave_callback is not None:
            try:
                wave_callback(wave_idx)
            except Exception:
                pass

        all_exposed.update(newly_exposed)

        wave_tallies = {path: 0 for path in STATE_PATHS}
        wave_sharers: List[str] = []
        wave_reporters: List[str] = []

        # 1. State Path Sampling for each newly exposed person
        for pid in newly_exposed:
            person = person_by_id[pid]
            r = person_reactions[pid]

            # Incorporate community immunity from flagged neighbors
            imm = immunity_penalties.get(pid, 0.0)
            base_act = max(0.0, r.act - imm)
            base_engage = max(0.0, r.engage - imm)

            # Incorporate intervention reduction if active
            if inv_strength > 0.0:
                act_mult, eng_mult = compute_intervention_reduction(person, inv_strength)
                base_act *= act_mult
                base_engage *= eng_mult

            base_act = min(base_act, base_engage)
            notice = max(base_engage, r.notice)

            # Categorical distribution over exactly one mutually exclusive state path
            p_report = r.report
            p_act = base_act
            p_share = r.share
            p_engage = max(0.0, base_engage - base_act)
            p_notice_only = max(0.0, notice - base_engage)
            p_ignore = max(0.05, 1.0 - notice)

            path_probs = np.array(
                [p_ignore, p_notice_only, p_engage, p_act, p_share, p_report],
                dtype=float,
            )
            total_p = path_probs.sum()
            if total_p > 0:
                path_probs /= total_p
            else:
                path_probs = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            chosen_path = str(rng.choice(STATE_PATHS, p=path_probs))
            node_sampled_states[pid] = chosen_path
            wave_tallies[chosen_path] += 1

            if chosen_path == "share":
                wave_sharers.append(pid)
            elif chosen_path == "report":
                wave_reporters.append(pid)

        # Update cumulative tallies
        total_shares += wave_tallies["share"]
        total_acts += wave_tallies["act"]
        total_reports += wave_tallies["report"]
        total_ignores += wave_tallies["ignore"]
        total_engages += wave_tallies["engage"]
        total_notice_only += wave_tallies["notice only"]

        # 2. Community Immunity Propagation from Reporters
        # Each report reduces belief/engagement probability of neighbors by 0.05 next wave
        for rep_id in wave_reporters:
            if graph.has_node(rep_id):
                for nbr in graph.neighbors(rep_id):
                    immunity_penalties[nbr] = immunity_penalties.get(nbr, 0.0) + 0.05

        # 3. Snapshot WaveState
        wave_state = WaveState(
            wave=wave_idx,
            reached_ids=newly_exposed,
            newly_reached=len(newly_exposed),
            cumulative_reached=len(all_exposed),
            reactions=dict(wave_tallies),
            shares=wave_tallies["share"],
            acts=wave_tallies["act"],
            reports=wave_tallies["report"],
            details={
                "channel": channel,
                "wave_sharers": len(wave_sharers),
                "wave_reporters": len(wave_reporters),
                "wave_decay": round(0.85 ** wave_idx, 4),
            },
        )
        waves.append(wave_state)

        # 4. Transmission to Unexposed Neighbors
        # Channels like 'call' exhibit zero forwarding
        if channel == "call":
            newly_exposed = []
            break

        next_wave_candidates: Set[str] = set()
        wave_decay = 0.85 ** wave_idx

        for sharer_id in wave_sharers:
            if not graph.has_node(sharer_id):
                continue

            for nbr in graph.neighbors(sharer_id):
                if nbr in all_exposed or nbr in next_wave_candidates:
                    continue

                edge_data = graph.get_edge_data(sharer_id, nbr, {})
                edge_type = edge_data.get("edge_type", "peer")

                # Edge type transmission probability
                if edge_type == "family":
                    base_p = 1.00
                elif edge_type == "hub":
                    base_p = 0.80
                else:  # neighbour, peer, homophily
                    base_p = 0.60

                p_transmission = base_p * wave_decay

                if rng.random() < p_transmission:
                    next_wave_candidates.add(nbr)
                    infection_edges.append(
                        {
                            "source": sharer_id,
                            "target": nbr,
                            "wave": wave_idx,
                            "edge_type": edge_type,
                        }
                    )

        newly_exposed = sorted(list(next_wave_candidates))

    # -----------------------------------------------------------------------
    # Cascade Summary & Performance Metrics
    # -----------------------------------------------------------------------
    reach_fraction = round(len(all_exposed) / max(1, n_people), 4)
    act_fraction = round(total_acts / max(1, n_people), 4)
    r0 = round(len(infection_edges) / max(1, total_shares), 2)

    # Granular Group Breakdown by Archetype
    group_stats: Dict[str, Dict[str, Any]] = {}
    for p in people:
        aid = p.archetype_id
        if aid not in group_stats:
            group_stats[aid] = {
                "count": 0,
                "reached": 0,
                "acts": 0,
                "shares": 0,
                "reports": 0,
                "ignores": 0,
                "engages": 0,
            }
        group_stats[aid]["count"] += 1
        if p.id in all_exposed:
            group_stats[aid]["reached"] += 1
            st = node_sampled_states.get(p.id, "ignore")
            if st == "act":
                group_stats[aid]["acts"] += 1
            elif st == "share":
                group_stats[aid]["shares"] += 1
            elif st == "report":
                group_stats[aid]["reports"] += 1
            elif st == "ignore":
                group_stats[aid]["ignores"] += 1
            elif st in ("engage", "notice only"):
                group_stats[aid]["engages"] += 1

    for aid, st in group_stats.items():
        st["reach_rate"] = round(st["reached"] / max(1, st["count"]), 3)
        st["act_rate"] = round(st["acts"] / max(1, st["reached"]), 3)

    # Determine Verdict
    verdict_info = verdict_for(domain_pack, reach=reach_fraction, r0=r0)
    verdict_label = verdict_info.get("label", domain_pack.verdict_labels[0])

    # Narrative Strategic Insights
    insights: List[str] = [
        f"Cascade reached {len(all_exposed)} of {n_people} agents ({reach_fraction * 100:.1f}%) across {len(waves)} waves.",
        f"Virality coefficient R0 = {r0:.2f} resulting in {total_shares} total forwards.",
    ]
    if total_acts > 0:
        insights.append(
            f"{total_acts} agents ({act_fraction * 100:.1f}%) converted or took action ('{domain_pack.act_label}')."
        )
    if total_reports > 0:
        insights.append(
            f"{total_reports} vigilant agents reported the message, creating localized community immunity."
        )
    if inv_strength > 0.0:
        insights.append(
            f"Active defense intervention (strength {inv_strength:.2f}) attenuated conversion and transmission rates."
        )

    totals = {
        "total_people": n_people,
        "cumulative_reached": len(all_exposed),
        "reach_fraction": reach_fraction,
        "total_acts": total_acts,
        "act_fraction": act_fraction,
        "total_shares": total_shares,
        "total_reports": total_reports,
        "total_ignores": total_ignores,
        "total_engages": total_engages,
        "total_notice_only": total_notice_only,
        "r0": r0,
        "waves_completed": len(waves),
        "channel": channel,
        "intervention_applied": bool(intervention),
        "intervention_strength": inv_strength,
    }

    return SimulationResult(
        waves=waves,
        edges=infection_edges,
        totals=totals,
        group_stats=group_stats,
        verdict=verdict_label,
        insights=insights,
        node_states=node_sampled_states,
    )
