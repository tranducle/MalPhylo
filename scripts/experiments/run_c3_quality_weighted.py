#!/usr/bin/env python3
"""C3 quality-weighted fusion validation.

Validates the Round-2 auto-improve C3 estimate by computing REAL quality-
weighted median fusion on the actual temporal signal data, then evaluating
within-family triplet inconsistency.

Round-2 estimate: near-pure string_markers (10:0.1:0.5:0) → 0.165
This script computes the REAL number.

REAL DATA ONLY — uses R-G1 features.jsonl + MalwareBazaar manifest.
NO FAKE DATA, NO ESTIMATES.
"""
from __future__ import annotations
import json, sys, time, random, re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
sys.path.insert(0, str(EXP_DIR / "src"))

# Reuse the temporal ablation infrastructure
from run_g4_ablate_temp_real import (
    load_data, compute_features_norm, compute_within_family_triplet_inconsistency,
    TEMPORAL_SIGNALS, TS_MIN, TS_MAX
)

OUT_DIR = EXP_DIR / "runs" / "R-C3-QUALITY-WEIGHTED"
SEEDS = [42, 43, 44, 45, 46]


def weighted_median_fusion(signals: dict, samples: list,
                            weights: Dict[str, float]) -> list:
    """Compute quality-weighted temporal fusion per sample.

    For each sample, take the weighted median of available signals.
    If only one signal is available, use it directly.
    If no signals, return None.
    """
    n = len(samples)
    temporal_vals = [None] * n

    # Build per-sample signal values
    sha_to_idx = {s["sha256"]: i for i, s in enumerate(samples)}

    for sig_name in TEMPORAL_SIGNALS:
        sig_vals = signals.get(sig_name, {})
        w = weights.get(sig_name, 0.0)

    # For each sample, collect (value, weight) pairs from available signals
    for i, s in enumerate(samples):
        sha = s["sha256"]
        pairs = []  # (value, weight)
        for sig_name in TEMPORAL_SIGNALS:
            val = signals.get(sig_name, {}).get(sha)
            w = weights.get(sig_name, 0.0)
            if val is not None and w > 0:
                pairs.append((val, w))

        if not pairs:
            temporal_vals[i] = None
        elif len(pairs) == 1:
            temporal_vals[i] = pairs[0][0]
        else:
            # Weighted median: sort by value, find where cumulative weight crosses 50%
            pairs.sort(key=lambda x: x[0])
            total_w = sum(w for _, w in pairs)
            cum = 0
            for val, w in pairs:
                cum += w
                if cum >= total_w / 2:
                    temporal_vals[i] = val
                    break
            if temporal_vals[i] is None:
                temporal_vals[i] = pairs[-1][0]  # fallback

    return temporal_vals


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_lines = []
    def log(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True); log_lines.append(line)

    log("=" * 70)
    log("C3 QUALITY-WEIGHTED FUSION VALIDATION (REAL data)")
    log("=" * 70)
    t0 = time.time()

    samples, signals = load_data()
    log(f"Loaded {len(samples)} samples")
    for sig in TEMPORAL_SIGNALS:
        log(f"  {sig}: {len(signals[sig])} samples with signal")

    features_norm = compute_features_norm(samples)
    log(f"Feature norms: {features_norm.shape}")

    # Weight configurations to test
    weight_configs = [
        ("uniform", {"pe_timestamp": 1.0, "first_seen": 1.0, "string_markers": 1.0, "code_reuse_order": 1.0}),
        ("quality_3:1:1:0.5", {"pe_timestamp": 1.0, "first_seen": 0.5, "string_markers": 3.0, "code_reuse_order": 1.0}),
        ("aggressive_5:0.5:1:0.2", {"pe_timestamp": 0.5, "first_seen": 0.2, "string_markers": 5.0, "code_reuse_order": 1.0}),
        ("near_pure_string_10:0.1:0.5:0", {"pe_timestamp": 0.1, "first_seen": 0.0, "string_markers": 10.0, "code_reuse_order": 0.5}),
        ("string_only", {"pe_timestamp": 0.0, "first_seen": 0.0, "string_markers": 1.0, "code_reuse_order": 0.0}),
        ("balanced_4:1:2:0.3", {"pe_timestamp": 1.0, "first_seen": 0.3, "string_markers": 4.0, "code_reuse_order": 2.0}),
    ]

    results = {}
    best_inconsist = 1.0
    best_config = None

    for name, weights in weight_configs:
        log(f"\n--- {name}: weights={weights} ---")
        temporal_vals = weighted_median_fusion(signals, samples, weights)

        n_valid = sum(1 for v in temporal_vals if v is not None)
        log(f"  Samples with fused timestamp: {n_valid}/{len(samples)}")

        # Evaluate over 5 seeds
        inconsists = []
        for seed in SEEDS:
            inc = compute_within_family_triplet_inconsistency(
                samples, features_norm, temporal_vals, seed=seed, max_triplets=5000)
            inconsists.append(inc)

        mean_inc = float(np.mean(inconsists))
        std_inc = float(np.std(inconsists))
        results[name] = {
            "weights": weights,
            "n_valid": n_valid,
            "inconsistency_mean": round(mean_inc, 4),
            "inconsistency_std": round(std_inc, 4),
            "per_seed": [round(x, 4) for x in inconsists],
        }
        log(f"  Triplet inconsistency: {mean_inc:.4f} ± {std_inc:.4f}")
        log(f"  Beats uniform (0.41): {mean_inc < 0.41}")
        log(f"  Beats best single (0.21): {mean_inc < 0.2114}")

        if mean_inc < best_inconsist:
            best_inconsist = mean_inc
            best_config = name
            log(f"  ✅ NEW BEST!")

    # Also evaluate individual signals for reference
    log(f"\n--- Individual signal references ---")
    for sig_name in TEMPORAL_SIGNALS:
        temporal_vals = [None] * len(samples)
        sig_vals = signals.get(sig_name, {})
        for i, s in enumerate(samples):
            v = sig_vals.get(s["sha256"])
            if v is not None:
                temporal_vals[i] = v
        n_valid = sum(1 for v in temporal_vals if v is not None)
        inconsists = []
        for seed in SEEDS:
            inc = compute_within_family_triplet_inconsistency(
                samples, features_norm, temporal_vals, seed=seed, max_triplets=5000)
            inconsists.append(inc)
        mean_inc = float(np.mean(inconsists))
        results[f"single_{sig_name}"] = {
            "n_valid": n_valid,
            "inconsistency_mean": round(mean_inc, 4),
            "inconsistency_std": round(float(np.std(inconsists)), 4),
        }
        log(f"  {sig_name}: {mean_inc:.4f} (n={n_valid})")

    log(f"\n{'='*70}")
    log(f"C3 QUALITY-WEIGHTED FUSION VALIDATION COMPLETE")
    log(f"  Best config: {best_config}")
    log(f"  Best inconsistency: {best_inconsist:.4f}")
    log(f"  Uniform baseline: {results['uniform']['inconsistency_mean']:.4f}")
    log(f"  Best single (string_markers): {results['single_string_markers']['inconsistency_mean']:.4f}")
    log(f"  Beats uniform: {best_inconsist < results['uniform']['inconsistency_mean']}")
    log(f"  Beats best single: {best_inconsist < results['single_string_markers']['inconsistency_mean']}")
    log(f"{'='*70}")

    metrics = {
        "run_id": "R-C3-QUALITY-WEIGHTED",
        "gate": "G4",
        "milestone": "C3 quality-weighted fusion validation (REAL data)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "COMPLETED",
        "results": results,
        "best_config": best_config,
        "best_inconsistency": round(best_inconsist, 4),
        "beats_uniform": best_inconsist < results["uniform"]["inconsistency_mean"],
        "beats_best_single": best_inconsist < results["single_string_markers"]["inconsistency_mean"],
        "timing_seconds": round(time.time() - t0, 2),
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (OUT_DIR / "run.log").write_text("\n".join(log_lines) + "\n")
    return metrics


if __name__ == "__main__":
    m = main()
    print(json.dumps({
        "best_config": m["best_config"],
        "best_inconsistency": m["best_inconsistency"],
        "beats_uniform": m["beats_uniform"],
        "beats_best_single": m["beats_best_single"],
    }, indent=2))
