"""
network.py  -  Persona
======================
Builds synthetic social graphs over Person nodes using NetworkX.
Incorporates:
  * Homophily (archetype and geographic affinity)
  * Household clusters of 3-5 people (family edges)
  * 5-6 central community hubs (leader, shopkeeper, doctor, influencer, admin)
  * Channel-specific structural dynamics:
      - whatsapp: dense small-world clusters & group chats
      - sms: sparse, lower clustering
      - social_feed: scale-free stars (few hubs reaching many)
      - call: near-zero forwarding, mostly dyadic & household links
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
import networkx as nx
import numpy as np

from backend.models import Person


class NetworkGraphResult(NamedTuple):
    """Container holding both NetworkX Graph and serializable dictionary payload."""

    graph: nx.Graph
    payload: Dict[str, Any]

    def __iter__(self):
        yield self.graph
        yield self.payload


# Hub role definitions and priority keyword matches
HUB_ROLES = [
    ("community_leader", ["secretary", "panchayat", "elder", "mobilizer", "leader"]),
    ("shopkeeper", ["shopkeeper", "kirana", "trader", "dealer", "merchant"]),
    ("doctor", ["doctor", "health", "nurse", "supervisor", "asha", "chemist"]),
    ("influencer", ["influencer", "streamer", "coder", "successor", "creative"]),
    ("group_admin", ["admin", "coordinator", "repper", "facilitator", "volunteer"]),
    ("trade_activist", ["activist", "union", "association", "advocate"]),
]


def _select_hubs(people: List[Person], py_rng: random.Random) -> List[Tuple[Person, str]]:
    """Select 5-6 distinctive hub nodes from the population representing key community roles."""
    selected_hubs: List[Tuple[Person, str]] = []
    used_ids = set()

    for role_name, keywords in HUB_ROLES:
        best_candidate: Optional[Person] = None
        for p in people:
            if p.id in used_ids:
                continue
            arch_lower = p.archetype_id.lower()
            if any(k in arch_lower for k in keywords):
                best_candidate = p
                break

        if best_candidate is None:
            # Fallback: pick available node with highest social_sharing or random
            candidates = [p for p in people if p.id not in used_ids]
            if candidates:
                candidates.sort(key=lambda x: x.traits.get("social_sharing", 0.5), reverse=True)
                best_candidate = candidates[0]

        if best_candidate and len(selected_hubs) < 6:
            selected_hubs.append((best_candidate, role_name))
            used_ids.add(best_candidate.id)

    return selected_hubs


def build_network(
    people: List[Person],
    channel: str = "whatsapp",
    seed: int = 42
) -> NetworkGraphResult:
    """Construct a synthetic social network over population agents based on transmission channel.

    Args:
        people: List of Person objects representing nodes.
        channel: Delivery medium ('whatsapp', 'sms', 'social_feed', 'call').
        seed: Deterministic seed for reproducible graph structure.

    Returns:
        NetworkGraphResult containing (networkx.Graph, serializable_payload).
    """
    if not people:
        empty_g = nx.Graph()
        return NetworkGraphResult(graph=empty_g, payload={"nodes": [], "edges": [], "metrics": {}})

    norm_channel = channel.strip().lower()
    if norm_channel in ("x", "twitter", "x_twitter", "feed", "broadcast"):
        norm_channel = "social_feed"
    elif norm_channel in ("phone", "phone_call", "voice"):
        norm_channel = "call"
    elif norm_channel not in ("whatsapp", "sms", "social_feed", "call"):
        norm_channel = "whatsapp"

    np_rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    G = nx.Graph(channel=norm_channel)
    n = len(people)

    # 1. Identify 5-6 community hub nodes
    hubs_info = _select_hubs(people, py_rng)
    hub_map: Dict[str, str] = {p.id: role for p, role in hubs_info}

    # Add nodes with metadata to graph
    for p in people:
        is_hub = p.id in hub_map
        hub_type = hub_map.get(p.id)
        G.add_node(
            p.id,
            name=p.name,
            archetype_id=p.archetype_id,
            age=p.age,
            language=p.language,
            area=p.area,
            traits=p.traits,
            is_hub=is_hub,
            hub_type=hub_type,
        )

    # 2. Form household clusters of 3-5 people (Family Edges)
    # Group people by area to make household clustering geographically coherent
    shuffled_people = list(people)
    py_rng.shuffle(shuffled_people)
    shuffled_people.sort(key=lambda p: p.area)

    idx = 0
    household_id = 0
    while idx < n:
        remaining = n - idx
        if remaining <= 5:
            cluster_size = remaining
        else:
            cluster_size = py_rng.randint(3, 5)

        cluster = shuffled_people[idx : idx + cluster_size]
        household_id += 1

        # Form family clique edges within household
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                u, v = cluster[i].id, cluster[j].id
                G.add_edge(u, v, edge_type="family", weight=0.95, household=household_id)

        idx += cluster_size

    # 3. Channel-Specific Parameters
    # Channel controls peer edge probabilities, hub reach, and group chat structures
    if norm_channel == "whatsapp":
        # Dense small-world groups: high clustering, active community & school chat cliques
        base_peer_prob = 0.025
        homophily_arch_boost = 0.045
        homophily_area_boost = 0.035
        hub_reach_fraction = 0.30
        group_clique_count = max(4, n // 15)
        group_clique_size_range = (5, 9)
    elif norm_channel == "sms":
        # Sparse direct messaging, lower clustering
        base_peer_prob = 0.008
        homophily_arch_boost = 0.020
        homophily_area_boost = 0.015
        hub_reach_fraction = 0.18
        group_clique_count = 0
        group_clique_size_range = (0, 0)
    elif norm_channel == "social_feed":
        # Few hubs reaching many (scale-free / power law distribution)
        base_peer_prob = 0.009
        homophily_arch_boost = 0.015
        homophily_area_boost = 0.015
        hub_reach_fraction = 0.55  # Hubs connect to over half the population
        group_clique_count = max(2, n // 25)
        group_clique_size_range = (4, 7)
    else:  # "call"
        # Near-zero forwarding: mostly family & very direct 1-to-1 contacts
        base_peer_prob = 0.003
        homophily_arch_boost = 0.008
        homophily_area_boost = 0.006
        hub_reach_fraction = 0.12
        group_clique_count = 0
        group_clique_size_range = (0, 0)

    # 4. Connect Hub Nodes across the population
    for hub_person, hub_role in hubs_info:
        hub_id = hub_person.id
        # Determine number of connections for this hub
        target_connections = int(n * hub_reach_fraction * py_rng.uniform(0.85, 1.25))
        target_connections = max(5, min(n - 1, target_connections))

        # Sample targets with affinity towards same area or high interest
        all_other_ids = [p.id for p in people if p.id != hub_id and not G.has_edge(hub_id, p.id)]
        if all_other_ids:
            py_rng.shuffle(all_other_ids)
            connect_to = all_other_ids[:target_connections]
            for target_id in connect_to:
                weight = round(py_rng.uniform(0.70, 0.90), 3)
                G.add_edge(hub_id, target_id, edge_type="hub", weight=weight, hub_role=hub_role)

    # 5. WhatsApp / Social Feed Group Chats (Small-World Cliques)
    if group_clique_count > 0:
        for _ in range(group_clique_count):
            g_size = py_rng.randint(*group_clique_size_range)
            # Pick an anchor node and assemble nearby nodes with same area or archetype
            anchor = py_rng.choice(people)
            cohort = [
                p for p in people
                if p.area == anchor.area or p.archetype_id == anchor.archetype_id
            ]
            if len(cohort) < g_size:
                cohort = list(people)
            py_rng.shuffle(cohort)
            members = cohort[:g_size]

            # Connect members in group chat clique
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    u, v = members[i].id, members[j].id
                    if not G.has_edge(u, v):
                        weight = round(py_rng.uniform(0.65, 0.85), 3)
                        G.add_edge(u, v, edge_type="peer", weight=weight)

    # 6. Homophily and Peer Links across general pairs
    person_by_id = {p.id: p for p in people}
    node_ids = list(person_by_id.keys())

    for i in range(len(node_ids)):
        p1 = person_by_id[node_ids[i]]
        for j in range(i + 1, len(node_ids)):
            p2 = person_by_id[node_ids[j]]
            u, v = p1.id, p2.id

            if G.has_edge(u, v):
                continue

            # Calculate pairwise connection probability based on homophily
            prob = base_peer_prob
            same_arch = p1.archetype_id == p2.archetype_id
            same_area = p1.area == p2.area
            same_lang = p1.language == p2.language

            if same_arch:
                prob += homophily_arch_boost
            if same_area:
                prob += homophily_area_boost
            if same_lang:
                prob += 0.015

            # Social sharing trait bonus
            sharing_bonus = 0.02 * ((p1.traits.get("social_sharing", 0.5) + p2.traits.get("social_sharing", 0.5)) / 2.0)
            prob += sharing_bonus

            if np_rng.random() < prob:
                edge_type = "homophily" if (same_arch or same_area) else "peer"
                weight = round(py_rng.uniform(0.50, 0.75), 3)
                G.add_edge(u, v, edge_type=edge_type, weight=weight)

    # 7. Ensure Connectivity: Giant Component > 90%
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    if components:
        giant = set(components[0])
        giant_ratio = len(giant) / n

        # Bridge disconnected islands to the giant component
        if giant_ratio < 0.92 and len(components) > 1:
            hub_ids_list = [p.id for p, _ in hubs_info]
            for small_comp in components[1:]:
                comp_node = py_rng.choice(list(small_comp))
                # Connect to a hub or random node in giant component
                target_node = py_rng.choice(hub_ids_list) if hub_ids_list else py_rng.choice(list(giant))
                G.add_edge(comp_node, target_node, edge_type="peer", weight=0.60)
                giant.update(small_comp)

    # Re-evaluate components
    final_components = sorted(nx.connected_components(G), key=len, reverse=True)
    giant_ratio = len(final_components[0]) / n if final_components else 0.0

    # 8. Compute Degree and Graph Statistics
    degrees = [d for _, d in G.degree()]
    avg_deg = float(np.mean(degrees)) if degrees else 0.0
    min_deg = int(np.min(degrees)) if degrees else 0
    max_deg = int(np.max(degrees)) if degrees else 0
    try:
        clustering_coeff = float(nx.average_clustering(G))
    except Exception:
        clustering_coeff = 0.0

    # 9. Build Serialisable Output Payload
    serialisable_nodes: List[Dict[str, Any]] = []
    for p in people:
        node_data = {
            "id": p.id,
            "name": p.name,
            "archetype_id": p.archetype_id,
            "age": p.age,
            "language": p.language,
            "area": p.area,
            "is_hub": p.id in hub_map,
            "hub_type": hub_map.get(p.id),
            "degree": G.degree(p.id),
            "traits": p.traits,
        }
        serialisable_nodes.append(node_data)

    serialisable_edges: List[Dict[str, Any]] = []
    for u, v, data in G.edges(data=True):
        edge_data = {
            "source": u,
            "target": v,
            "edge_type": data.get("edge_type", "peer"),
            "weight": data.get("weight", 0.5),
        }
        serialisable_edges.append(edge_data)

    payload: Dict[str, Any] = {
        "channel": norm_channel,
        "nodes": serialisable_nodes,
        "edges": serialisable_edges,
        "metrics": {
            "channel": norm_channel,
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "giant_component_ratio": round(giant_ratio, 4),
            "avg_degree": round(avg_deg, 2),
            "min_degree": min_deg,
            "max_degree": max_deg,
            "clustering_coefficient": round(clustering_coeff, 4),
            "num_hubs": len(hubs_info),
            "hub_ids": [p.id for p, _ in hubs_info],
        },
    }

    return NetworkGraphResult(graph=G, payload=payload)


def print_network_stats(result: NetworkGraphResult) -> None:
    """Print degree statistics and topology metrics for inspection."""
    m = result.payload.get("metrics", {})
    print(f"\n================ Network Topology: {m.get('channel', '').upper()} ================")
    print(f"  Nodes: {m.get('total_nodes')} | Edges: {m.get('total_edges')}")
    print(f"  Giant Component Ratio: {m.get('giant_component_ratio', 0.0)*100:.1f}%")
    print(f"  Average Degree: {m.get('avg_degree')} | Min: {m.get('min_degree')} | Max: {m.get('max_degree')}")
    print(f"  Clustering Coefficient: {m.get('clustering_coefficient', 0.0):.4f}")
    print(f"  Hubs ({m.get('num_hubs')}): {m.get('hub_ids')}")
    print("==================================================================\n")
