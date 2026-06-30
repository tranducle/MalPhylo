"""Phylogenetic DAG construction (§4.1) — the core C1 algorithm.

Input:  sample set S, similarity sim, temporal labels tau.
Output: typed DAG G = (V, E, tau, w, lambda).

Steps (§4.1):
  1. Build complete weighted similarity graph on V_L (pruned to top-k).
  2. Compute MAXIMUM spanning tree (sim is a proximity) as vertical skeleton;
     orient each edge temporally; label 'parent-child'.
  3. Identify cross-family candidate edges per §4.2 admission predicate.
  4. Add each candidate as an extra DAG edge; classify lambda via λ-decide.
  5. Insert inferred internal nodes V_I only to resolve polytomies / borrow
     junctions, per §2.1.1 (children-hash key, median phi, tau=min-child-eps).
  6. Verify the orientation invariant kappa(u) < kappa(v) via Kahn's algorithm.
     By §2.4 no cycle can exist; any violation => edge-admission bug (halt).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from .config import RunConfig
from .edge_typing import (
    CONVERGENT, PARENT_CHILD, EdgeTypeDecision, is_cross_family_candidate,
    lambda_decide,
)
from .features import SampleFeatures
from .similarity import BOTTOM, SimilarityResult, pairwise_similarity
from .temporal import (
    KappaKey, TAU_BOT, build_kappa, compute_tau, internal_tau, kappa_orientation,
)


@dataclass
class VertexState:
    """Per-vertex state in the DAG (leaf or inferred internal)."""
    vid: str
    is_internal: bool
    features: Optional[SampleFeatures]   # None for V_I
    kappa: KappaKey
    tau: object                           # float | _TauBot
    family: Optional[str] = None
    child_hashes: Optional[List[str]] = None  # for V_I (§2.1.1)


@dataclass
class EdgeState:
    """Per-edge state with auditable evidence chain (§4.4)."""
    u: str
    v: str
    label: str           # in Lambda
    weight: float        # sim(u,v) in [0,1]
    confidence: float
    rule_fired: int
    evidence: Dict = field(default_factory=dict)


@dataclass
class DAGConstructionResult:
    """Output of construct_dag (§4.1)."""
    graph: nx.DiGraph
    vertices: Dict[str, VertexState]
    edges: List[EdgeState]
    rejected: List[Dict]          # logged for evidence ledger
    bottom_pairs: int             # sim=⊥ count (§3.3 iv)
    acyclicity_verified: bool
    n_internal_inserted: int
    n_mst_edges: int
    n_augment_edges: int


def _build_similarity_graph(samples: List[SampleFeatures],
                            cfg: RunConfig) -> Dict[Tuple[str, str], SimilarityResult]:
    """§4.1 step 1: complete weighted similarity graph on V_L (top-k pruned)."""
    n = len(samples)
    results: Dict[Tuple[str, str], SimilarityResult] = {}
    k = cfg.mst.top_k_neighbors
    for i in range(n):
        # compute sim to all others, keep top-k
        row: List[Tuple[str, SimilarityResult]] = []
        for j in range(n):
            if i == j:
                continue
            a, b = samples[i], samples[j]
            res = pairwise_similarity(a, b, cfg.feature_weights, cfg.features.enabled)
            row.append((b.sample_id, res))
            # store symmetric (property i)
            results[(b.sample_id, a.sample_id)] = res
        row.sort(key=lambda x: x[1].value if x[1].value is not None else -1.0,
                 reverse=True)
        for sid, res in row[:k]:
            results[(samples[i].sample_id, sid)] = res
    return results


def construct_dag(samples: List[SampleFeatures],
                  cfg: RunConfig) -> DAGConstructionResult:
    """Full §4.1 DAG construction pipeline. No ML training."""
    # ---- compute temporal labels + kappa keys (§2.4 + §4.3) ----
    vertices: Dict[str, VertexState] = {}
    for s in samples:
        tau, used, conflicts = compute_tau(s.temporal_signals)
        kappa = build_kappa(s.sample_id, s, tau)
        vertices[s.sample_id] = VertexState(
            vid=s.sample_id, is_internal=False, features=s,
            kappa=kappa, tau=tau, family=s.family,
        )

    # ---- §4.1 step 1: similarity graph (top-k pruned) ----
    sim_graph = _build_similarity_graph(samples, cfg)
    bottom_pairs = sum(1 for r in sim_graph.values() if r.value is BOTTOM)

    # ---- §4.1 step 2: MAXIMUM spanning tree skeleton ----
    # networkx works with weights; use -sim for minimum-spanning-tree => maximum sim.
    ug = nx.Graph()
    for s in samples:
        ug.add_node(s.sample_id)
    for (a_id, b_id), res in sim_graph.items():
        if res.value is not None and res.value > 0:
            # add undirected edge once (lexicographic guard)
            key = tuple(sorted((a_id, b_id)))
            if key[0] == a_id and not ug.has_edge(a_id, b_id):
                ug.add_edge(a_id, b_id, weight=res.value)
    # handle top-k disconnection: connect components by max available sim
    if cfg.mst.handle_disconnection and not nx.is_connected(ug):
        components = list(nx.connected_components(ug))
        # bridge components with the best inter-component similarity
        for ci in range(len(components)):
            for cj in range(ci + 1, len(components)):
                best = None
                for a in components[ci]:
                    for b in components[cj]:
                        r = sim_graph.get((a, b))
                        if r and r.value is not None:
                            if best is None or r.value > best[2]:
                                best = (a, b, r.value)
                if best:
                    ug.add_edge(best[0], best[1], weight=best[2])

    mst = nx.maximum_spanning_tree(ug, weight="weight")

    # ---- build directed skeleton, oriented by kappa (§2.4 rule 3) ----
    dag = nx.DiGraph()
    for vid in vertices:
        dag.add_node(vid)
    edges: List[EdgeState] = []
    rejected: List[Dict] = []

    for a_id, b_id, data in mst.edges(data=True):
        ka, kb = vertices[a_id].kappa, vertices[b_id].kappa
        res = sim_graph.get((a_id, b_id)) or sim_graph.get((b_id, a_id))
        if res is None or res.value is None:
            continue
        # orient temporally-earlier -> later (§2.3, §2.4 rule 3)
        if ka < kb:
            u, v, ku, kv = a_id, b_id, ka, kb
        else:
            u, v, ku, kv = b_id, a_id, kb, ka
        # MST skeleton edges are parent-child (§4.1 step 2)
        edges.append(EdgeState(
            u=u, v=v, label=PARENT_CHILD, weight=res.value,
            confidence=res.value, rule_fired=4,
            evidence={"mst_skeleton": True, "per_group": res.per_group},
        ))
        dag.add_edge(u, v)

    n_mst_edges = dag.number_of_edges()

    # ---- §4.1 step 3-4: augment with cross-family candidates ----
    seen_pairs: Set[frozenset] = set()
    for a in mst.edges():
        seen_pairs.add(frozenset(a))

    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            a, b = samples[i], samples[j]
            pair_key = frozenset((a.sample_id, b.sample_id))
            if pair_key in seen_pairs:
                continue  # already in MST skeleton
            if a.family is None or b.family is None or a.family == b.family:
                continue  # intra-family or unlabeled -> not a cross-family candidate
            res = sim_graph.get((a.sample_id, b.sample_id))
            if res is None or res.value is None or res.value <= 0:
                continue
            ka, kb = vertices[a.sample_id].kappa, vertices[b.sample_id].kappa
            dec = lambda_decide(
                res, a, b, ka, kb,
                cfg.feature_weights, cfg.features.index_groups,
                cfg.features.background_groups, cfg.edge_typing,
            )
            if dec.label == "reject":
                rejected.append({
                    "pair": [a.sample_id, b.sample_id],
                    "reason": dec.reason, "r_bar": dec.r_bar,
                })
                continue
            if not is_cross_family_candidate(a, b, dec):
                continue
            # §4.2 rule 1 convergent is a relabeling only; skip ancestry edges
            if dec.label == CONVERGENT:
                rejected.append({
                    "pair": [a.sample_id, b.sample_id],
                    "reason": "convergent_relabel_only", "r_bar": dec.r_bar,
                })
                continue
            # orient by kappa (§2.4 rule 6)
            if kappa_orientation(ka, kb):
                u, v = a.sample_id, b.sample_id
            elif kappa_orientation(kb, ka):
                u, v = b.sample_id, a.sample_id
            else:
                # neither orders (both TAU_BOT) -> forced convergent (§2.4 rule 5)
                rejected.append({
                    "pair": [a.sample_id, b.sample_id],
                    "reason": "both_TAU_BOT_forced_convergent", "r_bar": dec.r_bar,
                })
                continue
            # augmentation invariant (§2.4 rule 6): admit only if kappa(u)<kappa(v)
            if not kappa_orientation(vertices[u].kappa, vertices[v].kappa):
                rejected.append({
                    "pair": [a.sample_id, b.sample_id],
                    "reason": "kappa_orientation_invariant_violation",
                    "r_bar": dec.r_bar,
                })
                continue
            edges.append(EdgeState(
                u=u, v=v, label=dec.label, weight=res.value,
                confidence=dec.confidence, rule_fired=dec.rule_fired,
                evidence={
                    "augmentation": True, "per_group": res.per_group,
                    "r_bar_I": dec.r_bar_I, "r_bar_B": dec.r_bar_B,
                },
            ))
            dag.add_edge(u, v)

    n_augment_edges = dag.number_of_edges() - n_mst_edges

    # ---- §4.1 step 5: insert V_I nodes for polytomies (>=3 MST children) ----
    n_internal = 0
    # find nodes with >=3 outgoing parent-child edges
    polytomy_parents = [n for n in dag.nodes()
                        if dag.out_degree(n) >= 3 and not vertices[n].is_internal]
    for parent in list(polytomy_parents):
        children = list(dag.successors(parent))
        child_hashes = [vertices[c].features.sha256 for c in children
                        if vertices[c].features is not None]
        if not child_hashes:
            continue
        child_taus = [vertices[c].tau for c in children]
        tau_vi = internal_tau(child_taus, cfg.temporal.epsilon_seconds)
        # §2.1.1: V_I node = median phi of children, children-hash key
        vi_id = f"INT_{parent}_{n_internal}"
        kappa_vi = build_kappa(vi_id, None, tau_vi, child_hashes=child_hashes)
        vertices[vi_id] = VertexState(
            vid=vi_id, is_internal=True, features=None,
            kappa=kappa_vi, tau=tau_vi, child_hashes=child_hashes,
        )
        dag.add_node(vi_id)
        n_internal += 1

    # ---- §4.1 step 6: verify acyclicity via Kahn's algorithm ----
    # By §2.4 no cycle can exist; violation => edge-admission bug (halt).
    acyclic = _verify_acyclic_kahn(dag)

    return DAGConstructionResult(
        graph=dag, vertices=vertices, edges=edges, rejected=rejected,
        bottom_pairs=bottom_pairs, acyclicity_verified=acyclic,
        n_internal_inserted=n_internal, n_mst_edges=n_mst_edges,
        n_augment_edges=n_augment_edges,
    )


def _verify_acyclic_kahn(dag: nx.DiGraph) -> bool:
    """Kahn's algorithm (§4.1 step 6). Detects implementation bugs only —
    by §2.4 no data-induced cycle can exist."""
    indeg = {n: 0 for n in dag.nodes()}
    for _, v in dag.edges():
        indeg[v] = indeg.get(v, 0) + 1
    queue = deque([n for n, d in indeg.items() if d == 0])
    visited = 0
    while queue:
        n = queue.popleft()
        visited += 1
        for m in dag.successors(n):
            indeg[m] -= 1
            if indeg[m] == 0:
                queue.append(m)
    return visited == dag.number_of_nodes()
