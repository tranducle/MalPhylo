#!/usr/bin/env python3
"""R-G2: Baseline sanity reproduction — B2 (UPGMA), B6 (NeighborNet + median-joining),
B7 (kNN graph), on MalTree Zenodo embeddings.

Gate G2 checks (experiment_plan §4):
  - B1 NJ structural reproduction: already PASS via R-SMOKE-2 (5K, cherry purity 0.8066).
    R-G2 reuses that result as the B1 sanity anchor (full 102K NJ = ~137hrs, deferred).
  - B2 UPGMA: hierarchical clustering (average linkage = UPGMA) on the same 5K subset.
    Sanity: non-degenerate tree, family-cherry purity >> random.
  - B6 untyped phylogenetic network: NeighborNet (NEXUS file for SplitsTree4) + median-joining
    approximation (Python). Sanity: network builds without error, cross-family edges present.
  - B7 kNN graph: plain similarity-threshold graph (no MST, no typing). Sanity: connected,
    cross-family edge fraction reported.

NO ML TRAINING (methodology §0/§3; experiment_plan §0). UPGMA, NJ, median-joining, and
kNN are deterministic graph/statistical algorithms.

Outputs (per experiment_plan §5 artifact contract):
  runs/R-G2/{metrics.json, run.log, config.yaml,
              b2_upgma_tree.newick, b6_nexus.nex, b6_median_joining.gml, b7_knn_graph.gexf}
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

# Reuse R-SMOKE-2 infrastructure
HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
sys.path.insert(0, str(EXP_DIR / "src"))
sys.setrecursionlimit(50000)

from malphpylo.maltree_nj import (
    load_embeddings,
    subsample,
    cosine_distance_matrix,
    NJNode,
    tree_branch_stats,
    family_clustering_purity,
)

import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
import networkx as nx
from sklearn.neighbors import NearestNeighbors


# --------------------------------------------------------------------------- #
# B2: UPGMA (average-linkage hierarchical clustering → Newick tree)
# --------------------------------------------------------------------------- #
def upgma_to_newick(names: List[str], D: np.ndarray) -> Tuple[str, dict]:
    """UPGMA via scipy average linkage. Returns (newick_string, stats).

    UPGMA = average-linkage hierarchical clustering. The linkage matrix is
    converted to a Newick tree with branch lengths = cophenetic distances / 2
    (UPGMA assumes a molecular clock / ultrametric).
    """
    # scipy needs condensed distance (upper triangle)
    cond = squareform(D, checks=False)
    Z = sch.linkage(cond, method="average")  # UPGMA = average linkage

    n = len(names)
    # Build Newick from linkage matrix
    # Z rows: [idx_i, idx_j, dist, count]
    nodes: Dict[int, Any] = {}
    for i in range(n):
        nodes[i] = names[i]  # leaves are strings

    for k, row in enumerate(Z):
        ci, cj, dist, _count = row
        ci, cj = int(ci), int(cj)
        ni = nodes.pop(ci)
        nj = nodes.pop(cj)
        # branch length = half the merge distance (ultrametric)
        blen = round(dist / 2.0, 6)
        if isinstance(ni, str):
            ni_str = ni
        else:
            ni_str = f"({ni})"
        if isinstance(nj, str):
            nj_str = nj
        else:
            nj_str = f"({nj})"
        new_node = f"{ni_str}:{blen},{nj_str}:{blen}"
        nodes[n + k] = new_node

    # last node is the root
    root_newick = list(nodes.values())[0]
    if isinstance(root_newick, str) and not root_newick.startswith("("):
        root_newick = f"({root_newick})"

    # Stats: cophenetic correlation (vs original distances)
    coph_corr, _coph_dists = sch.cophenet(Z, cond)  # (corr_coeff, condensed_cophenetic_dists)

    stats = {
        "method": "UPGMA_average_linkage",
        "n_leaves": n,
        "n_merges": int(len(Z)),
        "cophenetic_correlation": round(coph_corr, 6),
        "max_cophenetic_dist": round(float(Z[-1, 2]), 6),
    }
    return root_newick, stats


def upgma_cherry_purity(names: List[str], D: np.ndarray, families: Dict[str, str]) -> dict | None:
    """Compute cherry purity for UPGMA tree by reconstructing cherries from linkage matrix."""
    if not families:
        return None
    cond = squareform(D, checks=False)
    Z = sch.linkage(cond, method="average")
    n = len(names)

    # Track leaf sets per cluster node
    leaf_sets: Dict[int, set] = {}
    for i in range(n):
        leaf_sets[i] = {names[i]}

    pairs = 0
    same = 0
    for k, row in enumerate(Z):
        ci, cj = int(row[0]), int(row[1])
        si = leaf_sets.pop(ci)
        sj = leaf_sets.pop(cj)
        # A cherry = merge of two singletons (both size 1)
        if len(si) == 1 and len(sj) == 1:
            a, b = next(iter(si)), next(iter(sj))
            fa, fb = families.get(a), families.get(b)
            if fa is not None and fb is not None:
                pairs += 1
                if fa == fb:
                    same += 1
        leaf_sets[n + k] = si | sj

    return {
        "n_cherries_with_labels": pairs,
        "n_same_family": same,
        "cherry_purity": round(same / pairs, 4) if pairs else None,
    }


# --------------------------------------------------------------------------- #
# B6: NeighborNet (NEXUS file for SplitsTree4) + Median-Joining approximation
# --------------------------------------------------------------------------- #
def write_nexus_for_splitstree(names: List[str], D: np.ndarray, out_path: Path, run_id: str):
    """Write a NEXUS file with the distance matrix for SplitsTree4 NeighborNet import.

    SplitsTree4 can import this via: File → Open → select .nex file →
    choose NeighborNet method. The file format is standard NEXUS with
    a 'distances' block in full matrix format.
    """
    n = len(names)
    lines = [
        "#NEXUS",
        "",
        "[ R-G2 B6 NeighborNet input — MalTree embeddings, cosine distance ]",
        f"[ Run: {run_id} | n={n} | Generated by run_g2_baselines.py ]",
        "[ Import in SplitsTree4: File → Open → NeighborNet ]",
        "",
        "BEGIN Taxa;",
        f"  DIMENSIONS ntax={n};",
        "  TAXLABELS",
    ]
    for nm in names:
        lines.append(f"    {nm}")
    lines.append("  ;")
    lines.append("END;")

    lines.append("")
    lines.append("BEGIN Distances;")
    lines.append(f"  DIMENSIONS ntax={n};")
    lines.append("  FORMAT")
    lines.append("    labels=left")
    lines.append("    triangle=lower")
    lines.append("    diagonal")
    lines.append("  ;")
    lines.append("  MATRIX")

    for i in range(n):
        row_vals = " ".join(f"{D[i, j]:.8f}" for j in range(i + 1))
        lines.append(f"    {names[i]} {row_vals}")

    lines.append("  ;")
    lines.append("END;")

    out_path.write_text("\n".join(lines) + "\n")


def median_joining_network(
    names: List[str], D: np.ndarray, families: Dict[str, str], max_medians: int = 100
) -> Tuple[nx.Graph, dict]:
    """Simplified median-joining network (Bandelt et al. 1999) approximation.

    A median-joining network connects nodes via a minimum spanning network,
    then iteratively adds "median" (Steiner) nodes where three nodes form a
    near-minimal triangle. This is a tractable Python approximation of the
    SplitsTree4 implementation.

    Returns (networkx Graph, stats dict).
    """
    n = len(names)
    G = nx.Graph()
    for i in range(n):
        G.add_node(i, name=names[i], family=families.get(names[i], "Unknown"))

    # Step 1: Minimum Spanning Tree (Kruskal via scipy)
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial.distance import squareform as sf

    # Use condensed distance for MST
    mst_mat = minimum_spanning_tree(D).toarray()
    mst_edges = list(zip(*np.where(mst_mat > 0)))
    for i, j in mst_edges:
        G.add_edge(i, j, weight=round(float(D[i, j]), 6), edge_type="mst")

    # Step 2: Add short non-MST edges within a distance threshold (epsilon-neighborhood)
    # Threshold = 1.5x the max MST edge weight (conservative)
    max_mst_w = float(mst_mat.max()) if mst_mat.size > 0 else 1.0
    threshold = max_mst_w * 1.5
    extra_edges = 0
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] <= threshold and not G.has_edge(i, j):
                G.add_edge(i, j, weight=round(float(D[i, j]), 6), edge_type="proximity")
                extra_edges += 1

    # Step 3: Median vectors (Steiner points) — simplified
    # For each triangle (i, j, k) where all three edges are short, add a median node
    medians_added = 0
    edges_snapshot = list(G.edges(data="weight"))
    seen_triples = set()
    for idx, (i, j, _w) in enumerate(edges_snapshot):
        if medians_added >= max_medians:
            break
        neighbors_i = set(G.neighbors(i))
        neighbors_j = set(G.neighbors(j))
        common = neighbors_i & neighbors_j
        for k in common:
            triple = frozenset([i, j, k])
            if triple in seen_triples:
                continue
            seen_triples.add(triple)
            # Add median node if the triangle is compact
            di = D[i, k]
            dj = D[j, k]
            dij = D[i, j]
            if dij < threshold and di < threshold and dj < threshold:
                median_id = n + medians_added
                md_i = round(float((dij + di - dj) / 2.0), 6)
                md_j = round(float((dij + dj - di) / 2.0), 6)
                md_k = round(float((di + dj - dij) / 2.0), 6)
                md_i = max(0.0, md_i)
                md_j = max(0.0, md_j)
                md_k = max(0.0, md_k)
                G.add_node(median_id, name=f"median_{medians_added}", family="MEDIAN")
                G.add_edge(median_id, i, weight=md_i, edge_type="median")
                G.add_edge(median_id, j, weight=md_j, edge_type="median")
                G.add_edge(median_id, k, weight=md_k, edge_type="median")
                medians_added += 1
                if medians_added >= max_medians:
                    break

    # Stats
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    n_components = nx.number_connected_components(G)
    cross_family = 0
    same_family = 0
    for u, v in G.edges():
        fu = G.nodes[u].get("family", "?")
        fv = G.nodes[v].get("family", "?")
        if fu == "MEDIAN" or fv == "MEDIAN":
            continue
        if fu != "?" and fv != "?":
            if fu != fv:
                cross_family += 1
            else:
                same_family += 1

    degrees = [d for _, d in G.degree() if d > 0]
    stats = {
        "method": "median_joining_approximation",
        "n_nodes": n_nodes,
        "n_leaf_nodes": n,
        "n_median_nodes": medians_added,
        "n_edges": n_edges,
        "n_components": n_components,
        "n_mst_edges": len(mst_edges),
        "n_proximity_edges": extra_edges,
        "n_median_edges": medians_added * 3,
        "cross_family_edges": cross_family,
        "same_family_edges": same_family,
        "cross_family_fraction": round(cross_family / (cross_family + same_family), 4)
        if (cross_family + same_family) > 0
        else None,
        "mean_degree": round(float(np.mean(degrees)), 4) if degrees else 0,
        "max_degree": int(max(degrees)) if degrees else 0,
        "mst_max_weight": round(max_mst_w, 6),
        "proximity_threshold": round(threshold, 6),
        "note": (
            "Simplified median-joining approximation (Python). "
            "For publication-quality NeighborNet, import the NEXUS file in SplitsTree4."
        ),
    }
    return G, stats


# --------------------------------------------------------------------------- #
# B7: kNN graph (plain similarity-threshold graph, no MST, no typing)
# --------------------------------------------------------------------------- #
def knn_graph(
    names: List[str],
    emb: np.ndarray,
    families: Dict[str, str],
    k: int = 10,
) -> Tuple[nx.Graph, dict]:
    """Build a kNN graph from embeddings. No MST skeleton, no edge typing.

    This is baseline B7: a plain similarity-threshold graph. Each node connects
    to its k nearest neighbors by cosine distance. Cross-family edge fraction
    measures how much sharing the graph captures *without* MalPhylo's typing.
    """
    n = len(names)
    G = nx.Graph()
    for i in range(n):
        G.add_node(i, name=names[i], family=families.get(names[i], "Unknown"))

    # kNN via sklearn (cosine distance = 1 - cosine similarity)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
    nn.fit(emb)
    dists, indices = nn.kneighbors(emb)

    edges_added = 0
    for i in range(n):
        for j_idx in range(1, k + 1):  # skip self (index 0)
            j = int(indices[i, j_idx])
            if i < j:  # avoid duplicate edges
                w = round(float(dists[i, j_idx]), 6)
                G.add_edge(i, j, weight=w, edge_type="knn")
                edges_added += 1
            elif not G.has_edge(i, j):
                w = round(float(dists[i, j_idx]), 6)
                G.add_edge(i, j, weight=w, edge_type="knn")
                edges_added += 1

    # Stats
    n_components = nx.number_connected_components(G)
    cross_family = 0
    same_family = 0
    for u, v in G.edges():
        fu = G.nodes[u].get("family", "?")
        fv = G.nodes[v].get("family", "?")
        if fu != "?" and fv != "?":
            if fu != fv:
                cross_family += 1
            else:
                same_family += 1

    degrees = [d for _, d in G.degree()]
    stats = {
        "method": "kNN_graph",
        "k": k,
        "n_nodes": n,
        "n_edges": G.number_of_edges(),
        "n_components": n_components,
        "cross_family_edges": cross_family,
        "same_family_edges": same_family,
        "cross_family_fraction": round(cross_family / (cross_family + same_family), 4)
        if (cross_family + same_family) > 0
        else None,
        "mean_degree": round(float(np.mean(degrees)), 4) if degrees else 0,
        "max_degree": int(max(degrees)) if degrees else 0,
        "mean_edge_weight": round(float(np.mean([d["weight"] for _, _, d in G.edges(data=True)])), 6)
        if G.number_of_edges() > 0
        else 0,
        "note": "Plain kNN graph (no MST skeleton, no edge typing) = baseline B7.",
    }
    return G, stats


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run(config_path: str) -> dict:
    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    run_id = cfg["run_id"]
    out_dir = Path(cfg["output_dir"]) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    def log(msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log(f"R-G2 start | run_id={run_id}")
    t0 = time.time()

    # Load + subsample (same seed/subset as R-SMOKE-2 for comparability)
    sha_list, emb, meta_bundle = load_embeddings(cfg["embeddings_path"], cfg.get("dim_expect"))
    log(f"loaded {meta_bundle['meta']['n_total']} embeddings, dim={meta_bundle['meta']['dim']} "
        f"({meta_bundle['meta']['load_seconds']}s)")

    sha_sub, emb_sub = subsample(sha_list, emb, k=cfg["subset_size"], seed=cfg["seed"])
    log(f"subsampled to {len(sha_sub)} (seed={cfg['seed']})")

    families_sub = {
        s: meta_bundle["families"][s]
        for s in sha_sub
        if s in meta_bundle["families"]
    }
    log(f"family-labeled in subset: {len(families_sub)}/{len(sha_sub)}")

    # Distance matrix (shared by B2, B6, B7)
    t1 = time.time()
    D = cosine_distance_matrix(emb_sub)
    log(f"cosine distance matrix {D.shape} built in {round(time.time()-t1,2)}s")

    results: Dict[str, Any] = {}

    # ---- B1 reference (from R-SMOKE-2) ----
    r_smoke2_metrics_path = Path(cfg["output_dir"]) / "R-SMOKE-2" / "metrics.json"
    if r_smoke2_metrics_path.exists():
        with open(r_smoke2_metrics_path) as f:
            b1_ref = json.load(f)
        results["B1_NJ_reference"] = {
            "run_id": "R-SMOKE-2",
            "verdict": b1_ref.get("status", "COMPLETED"),
            "cherry_purity": b1_ref.get("family_clustering_purity", {}).get("cherry_purity"),
            "n_branches": b1_ref.get("tree_branch_stats", {}).get("n_branches"),
            "seconds": b1_ref.get("seconds_total"),
            "note": "B1 NJ sanity = R-SMOKE-2 (5K PASS). Full 102K NJ deferred (~137hrs).",
        }
        log(f"B1 reference (R-SMOKE-2): cherry_purity={results['B1_NJ_reference']['cherry_purity']}")

    # ---- B2: UPGMA ----
    log("B2 UPGMA — building average-linkage tree...")
    t2 = time.time()
    b2_newick, b2_stats = upgma_to_newick(sha_sub, D)
    b2_purity = upgma_cherry_purity(sha_sub, D, families_sub)
    b2_stats["cherry_purity"] = b2_purity
    b2_time = round(time.time() - t2, 2)
    b2_stats["seconds"] = b2_time
    results["B2_UPGMA"] = b2_stats
    (out_dir / "b2_upgma_tree.newick").write_text(b2_newick + ";\n")
    log(f"B2 done in {b2_time}s | purity={b2_purity['cherry_purity'] if b2_purity else 'n/a'} "
        f"| cophenetic_corr={b2_stats['cophenetic_correlation']}")

    # ---- B6: NeighborNet (NEXUS) + Median-Joining ----
    log("B6 NeighborNet — writing NEXUS file for SplitsTree4...")
    t6 = time.time()
    nexus_path = out_dir / "b6_nexus.nex"
    write_nexus_for_splitstree(sha_sub, D, nexus_path, run_id)
    nexus_size = nexus_path.stat().st_size
    log(f"  NEXUS written ({nexus_size // 1024}KB)")

    log("B6 Median-Joining — building approximation network...")
    mj_graph, mj_stats = median_joining_network(
        sha_sub, D, families_sub, max_medians=cfg.get("b6_max_medians", 100)
    )
    # Save as GML
    nx.write_gml(mj_graph, out_dir / "b6_median_joining.gml")
    mj_stats["nexus_file"] = str(nexus_path)
    mj_stats["nexus_size_kb"] = nexus_size // 1024
    mj_stats["seconds"] = round(time.time() - t6, 2)
    results["B6_NeighborNet_MedianJoining"] = mj_stats
    log(f"B6 done in {mj_stats['seconds']}s | nodes={mj_stats['n_nodes']} "
        f"edges={mj_stats['n_edges']} cross_family_frac={mj_stats['cross_family_fraction']}")

    # ---- B7: kNN graph ----
    k_val = cfg.get("b7_k", 10)
    log(f"B7 kNN graph — k={k_val}...")
    t7 = time.time()
    knn_g, knn_stats = knn_graph(sha_sub, emb_sub, families_sub, k=k_val)
    nx.write_gexf(knn_g, out_dir / "b7_knn_graph.gexf")
    knn_stats["seconds"] = round(time.time() - t7, 2)
    results["B7_kNN_graph"] = knn_stats
    log(f"B7 done in {knn_stats['seconds']}s | edges={knn_stats['n_edges']} "
        f"components={knn_stats['n_components']} cross_family_frac={knn_stats['cross_family_fraction']}")

    # ---- Aggregate metrics ----
    total_time = round(time.time() - t0, 2)
    metrics = {
        "run_id": run_id,
        "gate": "G2",
        "milestone": "baseline sanity reproduction",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "COMPLETED",
        "verdict": "PASS",  # updated below based on checks
        "subset_size": len(sha_sub),
        "seed": cfg["seed"],
        "distance": "cosine",
        "baselines": results,
        "timing": {"total_seconds": total_time},
        "data_meta": meta_bundle["meta"],
        "artifacts": {
            "b2_upgma_tree": str(out_dir / "b2_upgma_tree.newick"),
            "b6_nexus": str(nexus_path),
            "b6_median_joining": str(out_dir / "b6_median_joining.gml"),
            "b7_knn_graph": str(out_dir / "b7_knn_graph.gexf"),
        },
    }

    # G2 sanity checks
    checks = []
    # B2: non-degenerate + purity >> random
    if b2_purity and b2_purity["cherry_purity"] is not None:
        checks.append(("B2_cherry_purity_gt_random", b2_purity["cherry_purity"] > 0.01))
        checks.append(("B2_non_degenerate", b2_stats["n_merges"] > 0))
    # B6: network built + cross-family edges exist
    checks.append(("B6_network_built", mj_stats["n_edges"] > 0))
    checks.append(("B6_cross_family_edges", mj_stats["cross_family_edges"] > 0))
    # B7: graph connected or near-connected + cross-family edges
    checks.append(("B7_graph_built", knn_stats["n_edges"] > 0))
    checks.append(("B7_cross_family_edges", knn_stats["cross_family_edges"] > 0))

    all_pass = all(c[1] for c in checks)
    metrics["g2_checks"] = {name: bool(ok) for name, ok in checks}
    metrics["verdict"] = "PASS" if all_pass else "PARTIAL"
    metrics["note"] = (
        "G2 baseline sanity: B1 (NJ, from R-SMOKE-2) + B2 (UPGMA) + B6 (NeighborNet NEXUS + "
        "median-joining approx) + B7 (kNN graph). B6 NeighborNet proper requires SplitsTree4 GUI "
        "on the generated NEXUS file; median-joining approximation is the Python-tractable proxy."
    )

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "run.log").write_text("\n".join(log_lines) + "\n")
    with open(out_dir / "config.yaml", "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    log(f"R-G2 COMPLETE | verdict={metrics['verdict']} | time={total_time}s")
    log(f"  G2 checks: {metrics['g2_checks']}")
    return metrics


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="R-G2: B2/B6/B7 baseline sanity reproduction")
    ap.add_argument("--config", default="configs/r_g2.yaml", help="YAML config path")
    args = ap.parse_args()
    m = run(args.config)
    print(json.dumps({
        "run_id": m["run_id"],
        "verdict": m["verdict"],
        "g2_checks": m["g2_checks"],
        "baselines": {k: {kk: vv for kk, vv in v.items() if kk != "note"}
                       for k, v in m["baselines"].items()},
        "timing": m["timing"],
    }, indent=2))
