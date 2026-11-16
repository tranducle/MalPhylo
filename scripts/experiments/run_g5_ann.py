#!/usr/bin/env python3
"""R-ANN (G5/G6): HNSW index quality - recall@k vs exact kNN.

HNSW configuration specification: "HNSW hyperparameters are pinned a priori in configs/ann.yaml
 -  M_HNSW≥32, ef_Search≥512, ef_Construction≥200".
Threshold: recall@k ≥ r_ANN_floor (0.95) at k*=32.

Builds HNSW index on MalPhylo interpretable feature vectors from R-G1,
compares ANN recall@k against exact brute-force kNN.

Outputs: results/R-ANN/{metrics.json, run.log, config.yaml}
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent.parent
sys.path.insert(0, str(EXP_DIR))

RG1_PATH = EXP_DIR / "results" / "R-G1" / "features.jsonl"
THRESH_PATH = EXP_DIR / "configs" / "thresholds.yaml"
OUT_DIR = EXP_DIR / "results" / "R-ANN"

# HNSW hyperparameters (pinned a priori per HNSW/baseline specification)
M_HNSW = 32
EF_CONSTRUCTION = 200
EF_SEARCH = 512  # ≥ 2 * max(k_grid) = 2*256 = 512

K_GRID = [16, 32, 64, 128, 256]
K_STAR = 32  # operating k

MAX_SAMPLES = 2000  # subset for tractable exact kNN


def load_features(path: str, limit: int) -> Tuple[np.ndarray, List[str]]:
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            samples.append(d)
            if len(samples) >= limit:
                break

    vectors = []
    ids = []
    for d in samples:
        fg = d.get("feature_groups", {})
        vec = []
        # byte_hist (256 dims)
        bh = fg.get("byte_histogram", {})
        vec.extend(float(x) for x in bh.get("byte_hist", [0.0] * 256))
        # entropy_hist (16 dims)
        vec.extend(float(x) for x in bh.get("entropy_hist", [0.0] * 16))
        # string_entropy distribution (16 dims)
        se = fg.get("string_entropy", {})
        vec.extend(float(x) for x in se.get("entropy_distribution", [0.0] * 16))
        # cfg percentiles (5 dims)
        cfg = fg.get("cfg_topology", {})
        vec.extend(float(x) for x in cfg.get("percentiles", [0.0] * 5))
        # section summary
        sm = fg.get("section_metadata", {})
        sections = sm.get("sections", [])
        vec.append(float(len(sections)))
        vec.append(float(sum(s.get("entropy", 0) for s in sections)))
        vec.append(float(sum(s.get("vsize", 0) for s in sections)))
        # import count
        imp = fg.get("imports", {})
        vec.append(float(imp.get("n_symbols", 0)))
        vectors.append(vec)
        ids.append(d.get("sample_id", d.get("sha256", "")))

    return np.array(vectors, dtype=float), ids


def main(write_config: bool = False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_lines: List[str] = []
    def log(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True); log_lines.append(line)

    with open(THRESH_PATH) as f:
        thresh = yaml.safe_load(f)
    r_floor = thresh["r_ANN_floor"]["value"]

    log("=" * 60)
    log("R-ANN: HNSW index quality - recall@k vs exact kNN")
    log("=" * 60)
    log(f"HNSW: M={M_HNSW}, ef_construction={EF_CONSTRUCTION}, ef_search={EF_SEARCH}")
    log(f"k grid: {K_GRID}, k*={K_STAR}")
    log(f"r_ANN_floor: {r_floor}")

    t0 = time.time()

    # Load features
    features, ids = load_features(str(RG1_PATH), MAX_SAMPLES)
    n, dim = features.shape
    log(f"Loaded {n} samples, {dim} dimensions")

    # Remove zero-variance dimensions (degenerate features that add noise)
    var_mask = np.var(features, axis=0) > 0
    features = features[:, var_mask]
    dim = features.shape[1]  # update dim after filtering
    log(f"After zero-variance filter: {dim} dimensions (removed {297 - dim})")

    # Standardize features (z-score) so no single group dominates cosine similarity.
    # Without this, byte_histogram (256/297 dims) makes 99.6% of pairs have sim>0.999.
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    log(f"Features standardized (StandardScaler)")

    # Normalize for cosine similarity (HNSW uses inner product on normalized vectors)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    features_norm = features / norms

    # === Exact kNN (brute force) ===
    t1 = time.time()
    log("Computing exact kNN (brute force)...")
    max_k = max(K_GRID)
    # Cosine similarity = inner product on normalized vectors
    sim_matrix = features_norm @ features_norm.T
    np.fill_diagonal(sim_matrix, -np.inf)  # exclude self
    exact_knn = {}
    for k in K_GRID:
        # top-k indices for each sample
        exact_knn[k] = np.argpartition(-sim_matrix, k, axis=1)[:, :k]
    log(f"Exact kNN done ({time.time()-t1:.1f}s)")

    # === HNSW ANN index ===
    import hnswlib
    t2 = time.time()
    log("Building HNSW index...")
    index = hnswlib.Index(space='cosine', dim=dim)
    index.init_index(max_elements=n, ef_construction=EF_CONSTRUCTION, M=M_HNSW)
    index.add_items(features_norm, np.arange(n))
    build_time = time.time() - t2
    index.set_ef(EF_SEARCH)
    log(f"HNSW index built ({build_time:.3f}s)")

    # === Query ANN and measure recall@k ===
    t3 = time.time()
    log("Querying ANN and measuring recall@k...")
    recall_results = {}
    for k in K_GRID:
        labels, distances = index.knn_query(features_norm, k=k)
        # Compute recall@k for each sample
        recalls = []
        for i in range(n):
            exact_set = set(exact_knn[k][i].tolist())
            ann_set = set(labels[i].tolist())
            if len(exact_set) > 0:
                recall = len(exact_set & ann_set) / len(exact_set)
                recalls.append(recall)
        mean_recall = float(np.mean(recalls))
        recall_results[k] = {
            "mean_recall": round(mean_recall, 4),
            "std_recall": round(float(np.std(recalls)), 4),
            "min_recall": round(float(np.min(recalls)), 4),
            "recall_pass": mean_recall >= r_floor,
        }
        status = "PASS" if mean_recall >= r_floor else "FAIL"
        log(f"  recall@{k}: {mean_recall:.4f} (std={np.std(recalls):.4f}) - {status}")

    log(f"ANN queries done ({time.time()-t3:.1f}s)")

    # === Verdict at k* ===
    k_star_recall = recall_results[K_STAR]["mean_recall"]
    k_star_pass = recall_results[K_STAR]["recall_pass"]

    query_time_per_sample = (time.time() - t3) / (n * len(K_GRID))

    log(f"\n{'='*60}")
    log(f"VERDICT: {'PASS' if k_star_pass else 'FAIL'}")
    log(f"  recall@k*({K_STAR}): {k_star_recall:.4f} (floor: {r_floor})")
    log(f"  HNSW build time: {build_time:.1f}s")
    log(f"  Query time/sample: {query_time_per_sample*1000:.2f}ms")
    log(f"{'='*60}")

    metrics = {
        "run_id": "R-ANN",
        "milestone": "HNSW index quality - recall@k vs exact kNN",
        "gate": "G5/G6",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "COMPLETED",
        "verdict": "PASS" if k_star_pass else "FAIL",
        "config": {
            "M_HNSW": M_HNSW,
            "ef_construction": EF_CONSTRUCTION,
            "ef_search": EF_SEARCH,
            "space": "cosine",
            "k_star": K_STAR,
            "k_grid": K_GRID,
        },
        "dataset": {
            "n_samples": n,
            "dimensions": dim,
            "source": "R-G1 features",
        },
        "recall_at_k": {str(k): v for k, v in recall_results.items()},
        "k_star_recall": k_star_recall,
        "r_ANN_floor": r_floor,
        "timing": {
            "index_build_seconds": round(build_time, 2),
            "query_time_per_sample_ms": round(query_time_per_sample * 1000, 2),
            "total_seconds": round(time.time() - t0, 2),
        },
    }

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    (OUT_DIR / "run.log").write_text("\n".join(log_lines) + "\n")
    with open(OUT_DIR / "config.yaml", "w") as f:
        yaml.dump({"run_id": "R-ANN", "M_HNSW": M_HNSW,
                   "ef_construction": EF_CONSTRUCTION, "ef_search": EF_SEARCH,
                   "k_star": K_STAR, "k_grid": K_GRID,
                   "max_samples": MAX_SAMPLES}, f, default_flow_style=False)

    if write_config:
        ann_config = {
            "M_HNSW": M_HNSW,
            "ef_Search": EF_SEARCH,
            "ef_Construction": EF_CONSTRUCTION,
            "space": "cosine",
            "k_star": K_STAR,
        }
        ann_path = EXP_DIR / "configs" / "ann.yaml"
        with open(ann_path, "w") as f:
            yaml.dump(ann_config, f, default_flow_style=False)
        log(f"\nOutputs: {OUT_DIR}/ + {ann_path}")
    else:
        log(f"\nOutputs: {OUT_DIR}/")
    return metrics


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="R-ANN: HNSW index recall and latency evaluation")
    ap.add_argument("--write-config", action="store_true", help="Overwrite configs/ann.yaml")
    args = ap.parse_args()
    m = main(write_config=args.write_config)
    print(json.dumps({"run_id": m["run_id"], "verdict": m["verdict"],
                      "k_star_recall": m["k_star_recall"],
                      "r_floor": m["r_ANN_floor"]}, indent=2))
