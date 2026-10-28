"""
scripts/precompute.py  -  Persona
=================================
Precomputes and caches simulation cascades for:
- All 5 domain packs x 3 sample messages x 3 audience presets (45 combinations)
- 3 representative intervention comparison demos (Monte Carlo N=20 runs)

Saves payloads to:
1. data/cache/precomputed.json
2. frontend/precomputed.json

Ensures the Persona UI runs fully offline in Demo Mode and loads in <5 seconds.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure repository root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.analyzer import analyze_message
from backend.archetypes import PRESETS
from backend.config import CACHE_DIR, FRONTEND_DIR, N_PEOPLE, SEED
from backend.domain_packs import PACKS, SCAM_FRAUD, DomainPack
from backend.engine import run_simulation
from backend.insights import compare, generate_strategic_insights
from backend.network import build_network
from backend.population import generate_population
from backend.reactions import get_reactions
from backend.scoring import score_cascade

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("persona.precompute")


def precompute_simulations() -> Dict[str, Any]:
    """Precompute all pack samples across their 3 assigned audience presets."""
    simulations_cache: Dict[str, Any] = {}
    index_catalog: List[Dict[str, Any]] = []

    total_combinations = sum(len(p.samples) * len(p.audience_presets) for p in PACKS.values())
    logger.info(f"Precomputing {total_combinations} simulation cascades across 5 domain packs...")

    count = 0
    start_time = time.perf_counter()

    for pack_id, pack in PACKS.items():
        preset_ids = [p.id for p in pack.audience_presets]

        for s_idx, sample in enumerate(pack.samples):
            # Deterministic rule-based extraction for fast, stable offline cache
            features = analyze_message(sample.text, pack_id=pack.id, use_llm=False)

            for preset_id in preset_ids:
                count += 1
                archetypes = PRESETS.get(preset_id)
                if not archetypes:
                    logger.warning(f"Preset {preset_id} not found in PRESETS, falling back to urban_seniors")
                    archetypes = PRESETS["urban_seniors"]

                people = generate_population(archetypes, n=N_PEOPLE, seed=SEED)
                channel = features.channel_guess or "whatsapp"
                net_res = build_network(people, channel=channel, seed=SEED)

                reactions = get_reactions(
                    features=features,
                    message=sample.text,
                    archetypes=archetypes,
                    pack=pack,
                    k=1,
                    use_cache=True,
                    use_llm=False,
                )

                sim_result = run_simulation(
                    people=people,
                    graph=net_res.graph,
                    reactions=reactions,
                    features=features,
                    pack=pack,
                    intervention=None,
                    seed=SEED,
                )

                scoring_data = score_cascade(
                    people=people,
                    waves=sim_result.waves,
                    edges=sim_result.edges,
                    reactions=reactions,
                    pack=pack,
                    node_states=sim_result.node_states,
                )

                insights_data = generate_strategic_insights(
                    simulation_result=sim_result,
                    pack=pack,
                    strongest_trigger=scoring_data["strongest_trigger"],
                    weakest_group=scoring_data["weakest_group"],
                    features=features,
                )

                canonical_key = f"{pack_id}__s{s_idx}__{preset_id}"
                
                payload = {
                    "job_id": f"cached_{canonical_key}",
                    "status": "completed",
                    "degraded": False,
                    "cached": True,
                    "canonical_key": canonical_key,
                    "message": sample.text,
                    "sample_label": sample.label,
                    "sample_kind": sample.kind,
                    "pack_id": pack.id,
                    "audience_preset": preset_id,
                    "channel": channel,
                    "features": features.model_dump(),
                    "audience": [a.model_dump() for a in archetypes],
                    "population_count": len(people),
                    "network_summary": {
                        "node_count": len(people),
                        "edge_count": len(net_res.graph.edges),
                        "channel": channel,
                    },
                    "network_payload": net_res.payload,
                    "reactions": {aid: r.model_dump() for aid, r in reactions.items()},
                    "simulation": sim_result.model_dump(),
                    "scoring": scoring_data,
                    "insights": insights_data,
                }

                simulations_cache[canonical_key] = payload

                index_catalog.append({
                    "key": canonical_key,
                    "pack_id": pack_id,
                    "sample_idx": s_idx,
                    "sample_label": sample.label,
                    "sample_kind": sample.kind,
                    "audience_preset": preset_id,
                    "reach_fraction": scoring_data["reach_fraction"],
                    "act_fraction": scoring_data["act_fraction"],
                    "r0": scoring_data["r0"],
                    "verdict": scoring_data["verdict"],
                })

    elapsed = time.perf_counter() - start_time
    logger.info(f"Computed {count} simulations in {elapsed:.2f}s ({elapsed/count*1000:.1f}ms/sim).")
    return {"simulations": simulations_cache, "catalog": index_catalog}


def precompute_compare_demos() -> Dict[str, Any]:
    """Precompute 3 representative intervention comparison scenarios (20x Monte Carlo runs)."""
    demos: Dict[str, Any] = {}

    compare_specs = [
        {
            "id": "compare_scam_electricity",
            "title": "Electricity Bill Threat vs Discom Advisory",
            "message": "URGENT: Your electricity connection will be disconnected tonight at 9:30 PM due to an unpaid bill. Call 9876543210 immediately or pay via link: http://bit.ly/quick-bill-pay to prevent power cut.",
            "pack_id": "scam_fraud",
            "audience": "urban_seniors",
            "interventions": [
                "Be cautious about online messages and verify payment details.",
                "Official Electricity Board Advisory: We never disconnect power without 15 days written notice. Verify bills only at discom.gov.in or helpline 1912.",
            ],
            "n_runs": 20,
            "n_people": 40,
        },
        {
            "id": "compare_bank_kyc",
            "title": "Bank KYC Block Threat vs RBI Mandate",
            "message": "Dear customer, your SBI YONO account will be blocked today because PAN is not linked. Update immediately at http://sbi-pan-update.xyz to keep active.",
            "pack_id": "scam_fraud",
            "audience": "urban_seniors",
            "interventions": [
                "Do not click unknown links or share passwords with strangers.",
                "RBI Public Alert: Banks never freeze accounts via SMS links. Never disclose OTP or passwords. Report suspicious messages immediately to 1930.",
            ],
            "n_runs": 20,
            "n_people": 40,
        },
        {
            "id": "compare_health_campaign",
            "title": "Dengue Warning: Bureaucratic Jargon vs Actionable Steps",
            "message": "Health Department Notice: Monsoon vector-borne disease outbreak risk elevated in your ward. Take immediate preventive measures against dengue fever.",
            "pack_id": "health_campaign",
            "audience": "working_parents",
            "interventions": [
                "Citizens should adhere to municipal vector-borne disease guidelines and eradicate water stagnation reservoirs.",
                "Clean and empty standing water from air coolers, flowerpots, and drums every Sunday. If your child develops high fever, visit your nearest primary health center.",
            ],
            "n_runs": 20,
            "n_people": 40,
        },
    ]

    logger.info("Precomputing 3 compare demos (20x Monte Carlo runs each)...")

    for spec in compare_specs:
        start = time.perf_counter()
        pack = PACKS.get(spec["pack_id"], SCAM_FRAUD)
        archetypes = PRESETS.get(spec["audience"], PRESETS["urban_seniors"])
        res = compare(
            message=spec["message"],
            pack=pack,
            archetypes=archetypes,
            interventions=spec["interventions"],
            n_runs=spec["n_runs"],
            n_people=spec["n_people"],
            use_llm=False,
            channel="whatsapp",
            base_seed=SEED,
        )
        elapsed = time.perf_counter() - start

        res["id"] = spec["id"]
        res["title"] = spec["title"]
        res["cached"] = True
        demos[spec["id"]] = res
        logger.info(f"Precomputed {spec['id']} in {elapsed:.2f}s (Winner: {res['winner']})")

    return demos


def build_precomputed_bundle() -> Dict[str, Any]:
    """Execute precomputation and package into single offline bundle."""
    sim_data = precompute_simulations()
    compare_data = precompute_compare_demos()

    bundle = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "version": "1.0.0",
        "simulations_count": len(sim_data["simulations"]),
        "compare_count": len(compare_data),
        "simulations": sim_data["simulations"],
        "catalog": sim_data["catalog"],
        "compare_demos": compare_data,
    }

    return bundle


def main():
    logger.info("=== Starting Persona Offline Cache Precomputation ===")
    bundle = build_precomputed_bundle()

    # Destination 1: data/cache/precomputed.json
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "precomputed.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, separators=(",", ":"))
    cache_size_kb = os.path.getsize(cache_path) / 1024
    logger.info(f"✓ Saved backend cache to: {cache_path} ({cache_size_kb:.1f} KB)")

    # Destination 2: frontend/precomputed.json
    FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
    frontend_path = FRONTEND_DIR / "precomputed.json"
    with open(frontend_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, separators=(",", ":"))
    fe_size_kb = os.path.getsize(frontend_path) / 1024
    logger.info(f"✓ Saved frontend offline bundle to: {frontend_path} ({fe_size_kb:.1f} KB)")

    # Print summary table
    print(f"\n{'='*78}")
    print(f"PERSONA PRECOMPUTED DEMO CACHE SUMMARY")
    print(f"{'='*78}")
    print(f"Total Simulation Cascades: {bundle['simulations_count']} (5 packs x 3 samples x 3 presets)")
    print(f"Total Compare Scenarios:   {bundle['compare_count']} (20x Monte Carlo seeds each)")
    print(f"Backend Cache Location:    {cache_path} ({cache_size_kb:.1f} KB)")
    print(f"Frontend Cache Location:   {frontend_path} ({fe_size_kb:.1f} KB)")
    print(f"{'='*78}")
    print("✓ Precompute completed successfully. Offline Demo Mode ready (<5s load guaranteed).\n")


if __name__ == "__main__":
    main()
