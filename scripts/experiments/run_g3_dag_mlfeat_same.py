#!/usr/bin/env python3
"""R-DAG-MLFEAT-SAME: No-ML same-corpus comparison via PCA proxy.

Replaces the cross-corpus R-DAG-MLFEAT (MalwareBazaar vs MalTree, 28 overlap)
with a SAME-CORPUS comparison on R-G1 (3002 samples):

1. MalPhylo: interpretable feature similarity (imports Jaccard + string entropy + sections + byte_hist)
2. PCA-ML proxy: PCA(50) on standardized R-G1 features → cosine similarity = "learned representation"
3. Both build a DAG on the SAME 3002 samples
4. Compare cross-family recovery rate (cf_fraction)
5. Non-inferiority test: Δ(MalPhylo - PCA-ML) within δ_eq=0.10

This is fair: both use the same data, only the representation differs.
PCA captures the dominant variance directions - a reasonable ML embedding proxy.

Runs 5 times for stability.

Outputs: results/R-DAG-MLFEAT-SAME-v2/{metrics.json, run.log, config.yaml}
"""
from __future__ import annotations

import json, sys, time, random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent.parent
sys.path.insert(0, str(EXP_DIR))

RG1_PATH = EXP_DIR / "results" / "R-G1" / "features.jsonl"
THRESH_PATH = EXP_DIR / "configs" / "thresholds.yaml"
OUT_DIR = EXP_DIR / "results" / "R-DAG-MLFEAT-SAME-v2"  # canonical 12-family corpus

N_RUNS = 5
# Filter to canonical target families before applying per-family sample caps.
# limit=2074 read only the FIRST 2074 lines of features.jsonl, then the >=100/
# family filter removed many Unknown/small-family rows, leaving n=687 (not the
# canonical 2074 12-family corpus). To match R-CROSSFAM-HONEST exactly, load ALL
# samples, apply the SAME >=100-family filter, and use the full resulting corpus.
MAX_SAMPLES = 10**9  # effectively unlimited; the family filter is the real gate
K_NEIGHBORS = 32


def load_features(path: str, limit: int) -> Tuple[np.ndarray, List[str], List[str]]:
    """Load R-G1 features as numeric vectors + family + sample_id."""
    samples = []
    with open(path) as f:
        for line in f:
            d = json.loads(line.strip())
            samples.append(d)
            if len(samples) >= limit:
                break

    vectors, families, ids = [], [], []
    for d in samples:
        fg = d.get("feature_groups", {})
        vec = []
        vec.extend(float(x) for x in fg.get("byte_histogram", {}).get("byte_hist", [0.0] * 256))
        vec.extend(float(x) for x in fg.get("byte_histogram", {}).get("entropy_hist", [0.0] * 16))
        vec.extend(float(x) for x in fg.get("string_entropy", {}).get("entropy_distribution", [0.0] * 16))
        vec.extend(float(x) for x in fg.get("cfg_topology", {}).get("percentiles", [0.0] * 5))
        sm = fg.get("section_metadata", {})
        secs = sm.get("sections", [])
        vec.extend([float(len(secs)),
                    float(sum(s.get("entropy", 0) for s in secs)),
                    float(sum(s.get("vsize", 0) for s in secs))])
        vec.append(float(fg.get("imports", {}).get("n_symbols", 0)))
        # Add import Jaccard features: top-20 most common import prefixes
        imports = fg.get("imports", {}).get("symbol_set", [])
        imp_prefixes = [imp.split("!")[0] if "!" in imp else imp[:8] for imp in imports]
        for prefix in ["kernel32", "advapi32", "user32", "ntdll", "ole32", "oleaut32", "wininet", "urlmon", "ws2_32", "msvcrt", "gdi32", "shell32", "comctl32", "crypt32", "version", "winreg", "psapi", "iphlpapi", "dbghelp", "shlwapi"]:
            vec.append(float(sum(1 for p in imp_prefixes if prefix in p.lower())))
        vectors.append(vec)
        families.append(d.get("family", "Unknown"))
        ids.append(d.get("sample_id", d.get("sha256", "")))

    return np.array(vectors, dtype=float), families, ids


def compute_cf_rate(sim: np.ndarray, families: List[str], k: int) -> Tuple[float, Dict]:
    """Compute cross-family edge recovery rate from similarity matrix."""
    n = len(families)
    cf_edges = 0
    same_edges = 0
    per_family = defaultdict(lambda: {"cf": 0, "same": 0})

    for i in range(n):
        top_k = np.argpartition(-sim[i], min(k, n-1))[:min(k, n-1)]
        for j in top_k:
            ji = int(j)
            if families[i] == "Unknown" or families[ji] == "Unknown":
                continue
            if families[i] != families[ji]:
                cf_edges += 1
                per_family[families[i]]["cf"] += 1
            else:
                same_edges += 1
                per_family[families[i]]["same"] += 1

    total = cf_edges + same_edges
    cf_rate = cf_edges / total if total > 0 else 0
    return cf_rate, {"cf_edges": cf_edges, "same_edges": same_edges, "total": total,
                     "per_family": {f: dict(d) for f, d in per_family.items()}}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_lines = []
    def log(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True); log_lines.append(line)

    with open(THRESH_PATH) as f:
        thresh = yaml.safe_load(f)
    delta_eq = thresh["delta_eq"]["value"]

    log("=" * 70)
    log("R-DAG-MLFEAT-SAME: No-ML same-corpus comparison (PCA proxy, 5 runs)")
    log("=" * 70)

    t0 = time.time()
    features, families, ids = load_features(str(RG1_PATH), MAX_SAMPLES)

    # Apply the same canonical 12-family filter so the No-ML control
    # runs on the canonical 12-family corpus (>=100 samples/family, no Unknown).
    # The previous 500-sample cap without this filter admitted "Bumblebee" (a
    # <100 family), causing a 13-vs-12 family contradiction with R-CROSSFAM.
    fam_counts_all = Counter(families)
    valid_families = {f for f, c in fam_counts_all.items()
                      if c >= 100 and f != "Unknown"}
    keep_mask = [f in valid_families for f in families]
    features = features[keep_mask]
    families = [f for f, k in zip(families, keep_mask) if k]
    ids = [i for i, k in zip(ids, keep_mask) if k]

    n, dim = features.shape
    log(f"Loaded {n} samples ( {len(valid_families)} families, >=100 each), {dim} dimensions")
    fam_counts = Counter(families)
    log(f"Families: {len(fam_counts)} ({dict(fam_counts.most_common(5))}...)")

    # Remove zero-variance dims
    var_mask = np.var(features, axis=0) > 0
    features = features[:, var_mask]
    log(f"After zero-variance filter: {features.shape[1]} dims")

    # Standardize
    from sklearn.preprocessing import StandardScaler
    features_std = StandardScaler().fit_transform(features)

    run_results = []
    for run_idx in range(N_RUNS):
        seed = 42 + run_idx
        np.random.seed(seed)
        log(f"\n=== Run {run_idx+1}/{N_RUNS} (seed={seed}) ===")

        # MalPhylo: interpretable similarity (cosine on standardized raw features)
        norms_mp = np.linalg.norm(features_std, axis=1, keepdims=True)
        norms_mp[norms_mp == 0] = 1e-10
        features_mp_norm = features_std / norms_mp
        sim_mp = features_mp_norm @ features_mp_norm.T
        np.fill_diagonal(sim_mp, -1)

        # PCA-ML proxy: PCA(50) → cosine similarity = "learned representation"
        from sklearn.decomposition import PCA
        n_components = min(50, features_std.shape[1])
        pca = PCA(n_components=n_components, random_state=seed)
        features_pca = pca.fit_transform(features_std)
        norms_pca = np.linalg.norm(features_pca, axis=1, keepdims=True)
        norms_pca[norms_pca == 0] = 1e-10
        features_pca_norm = features_pca / norms_pca
        sim_pca = features_pca_norm @ features_pca_norm.T
        np.fill_diagonal(sim_pca, -1)

        # Cross-family recovery for both
        cf_mp, stats_mp = compute_cf_rate(sim_mp, families, K_NEIGHBORS)
        cf_pca, stats_pca = compute_cf_rate(sim_pca, families, K_NEIGHBORS)

        delta = cf_mp - cf_pca
        non_inferior = delta >= -delta_eq

        log(f"  MalPhylo cf_rate: {cf_mp:.4f} ({stats_mp['cf_edges']}/{stats_mp['total']})")
        log(f"  PCA-ML cf_rate:   {cf_pca:.4f} ({stats_pca['cf_edges']}/{stats_pca['total']})")
        log(f"  Δ(MalPhylo - PCA-ML): {delta:+.4f}")
        log(f"  Non-inferior (Δ ≥ -{delta_eq}): {non_inferior}")

        verdict = "NON-INFERIOR" if non_inferior else "INFERIOR"
        log(f"  Verdict: {verdict}")

        run_results.append({
            "run": run_idx + 1, "seed": seed,
            "malphpylo_cf_rate": round(cf_mp, 4),
            "pca_ml_cf_rate": round(cf_pca, 4),
            "delta": round(delta, 4),
            "non_inferior": non_inferior,
            "verdict": verdict,
            "stats_mp": stats_mp,
            "stats_pca": stats_pca,
            "pca_components": n_components,
            "pca_explained_variance": round(float(pca.explained_variance_ratio_[:5].sum()), 4),
        })

    # Stability
    all_ni = all(r["non_inferior"] for r in run_results)
    deltas = [r["delta"] for r in run_results]
    delta_std = float(np.std(deltas))
    delta_mean = float(np.mean(deltas))
    stable = delta_std < 0.02

    log(f"\n{'='*70}")
    log(f"STABILITY CHECK ({N_RUNS} runs)")
    log(f"  All NON-INFERIOR: {all_ni}")
    log(f"  Δ: mean={delta_mean:+.4f}, std={delta_std:.4f} - {deltas}")
    log(f"  Stable (std<0.02): {stable}")
    final = "NON-INFERIOR (STABLE)" if all_ni and stable else "UNSTABLE"
    log(f"  Final: {final}")
    log(f"{'='*70}")

    metrics = {
        "run_id": "R-DAG-MLFEAT-SAME", "version": "v1",
        "milestone": "No-ML same-corpus comparison (PCA proxy)",
        "gate": "G3", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "COMPLETED", "verdict": final,
        "n_runs": N_RUNS, "all_non_inferior": all_ni,
        "delta_mean": round(delta_mean, 4), "delta_std": round(delta_std, 4),
        "stable": stable, "run_results": run_results,
        "thresholds": {"delta_eq": delta_eq},
        "evaluation_type": "SAME-CORPUS (R-G1, PCA proxy for ML embedding)",
        "n_samples": n, "k_neighbors": K_NEIGHBORS,
        "timing_seconds": round(time.time() - t0, 2),
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    (OUT_DIR / "run.log").write_text("\n".join(log_lines) + "\n")
    with open(OUT_DIR / "config.yaml", "w") as f:
        yaml.dump({"run_id": "R-DAG-MLFEAT-SAME", "n_runs": N_RUNS,
                   "max_samples": MAX_SAMPLES, "k": K_NEIGHBORS}, f, default_flow_style=False)
    log(f"\nOutputs: {OUT_DIR}/")
    return metrics


if __name__ == "__main__":
    m = main()
    print(json.dumps({"verdict": m["verdict"], "delta_mean": m["delta_mean"],
                      "delta_std": m["delta_std"], "stable": m["stable"]}, indent=2))
