#!/usr/bin/env python3
"""R-ABLATE-TEMP-LOO-REAL v2: REAL temporal LOO with within-family triplet inconsistency.

Key fix: pe_timestamp is no longer used as both signal and ground truth.
Instead, uses WITHIN-FAMILY triplet inconsistency as the metric:
  For triplets (i,j,k) within the same family where sim(i,j) > sim(i,k),
  check if temporal ordering agrees (j closer in time to i than k).
  Lower inconsistency = better temporal alignment.

Signals (4 REAL):
  1. pe_timestamp: PE compilation time
  2. first_seen: MalwareBazaar submission time  
  3. string_markers: date strings from binary
  4. code_reuse_order: import subset ordering proxy

Outputs: results/R-ABLATE-TEMP-LOO-REAL/{metrics.json, run.log, config.yaml}
"""
from __future__ import annotations

import json, sys, time, re, random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent.parent
sys.path.insert(0, str(EXP_DIR))

RG1_PATH = EXP_DIR / "results" / "R-G1" / "features.jsonl"
MANIFEST_PATH = EXP_DIR / "data" / "malwarebazaar" / "manifest.jsonl"
THRESH_PATH = EXP_DIR / "configs" / "thresholds.yaml"
OUT_DIR = EXP_DIR / "results" / "R-ABLATE-TEMP-LOO-REAL"

N_RUNS = 5
TS_MIN = 946684800
TS_MAX = 1767225600
TEMPORAL_SIGNALS = ["pe_timestamp", "first_seen", "string_markers", "code_reuse_order"]


def parse_date(s):
    if not s: return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try: return datetime.strptime(str(s).strip(), fmt).timestamp()
        except ValueError: pass
    return None


def extract_string_date(rare):
    if not rare: return None
    text = " ".join(rare)
    years = re.findall(r"(20[12]\d)", text)
    if years:
        return datetime(min(int(y) for y in years), 1, 1).timestamp()
    return None


def extract_code_reuse(sample, all_samples):
    imports = set(sample.get("feature_groups", {}).get("imports", {}).get("symbol_set", []))
    if not imports: return None
    fam = sample.get("family", "Unknown")
    if fam == "Unknown": return None
    peers = [s for s in all_samples if s.get("family") == fam and s.get("sha256") != sample.get("sha256")]
    if len(peers) < 2: return None
    n_supersets = sum(1 for p in peers 
                      if imports.issubset(set(p.get("feature_groups",{}).get("imports",{}).get("symbol_set",[])))
                      and len(set(p.get("feature_groups",{}).get("imports",{}).get("symbol_set",[]))) > len(imports))
    pe = sample.get("temporal_signals", {}).get("pe_timestamp")
    if pe and TS_MIN < float(pe) < TS_MAX:
        return float(pe) - n_supersets * 86400 * 7
    return None


def load_data():
    rg1 = {}
    with open(RG1_PATH) as f:
        for line in f:
            d = json.loads(line.strip())
            rg1[d["sha256"]] = d
    manifest = {}
    with open(MANIFEST_PATH) as f:
        for line in f:
            d = json.loads(line.strip())
            sha = d.get("sha256", "")
            if sha in rg1: manifest[sha] = d
    
    samples = list(rg1.values())
    signals = {sig: {} for sig in TEMPORAL_SIGNALS}
    
    for s in samples:
        sha = s["sha256"]
        pe = s.get("temporal_signals", {}).get("pe_timestamp")
        if pe and TS_MIN < float(pe) < TS_MAX:
            signals["pe_timestamp"][sha] = float(pe)
        if sha in manifest:
            fs = parse_date(manifest[sha].get("first_seen", ""))
            if fs: signals["first_seen"][sha] = fs
        rare = s.get("feature_groups", {}).get("string_entropy", {}).get("rare_strings", [])
        sm = extract_string_date(rare)
        if sm: signals["string_markers"][sha] = sm
        cro = extract_code_reuse(s, samples)
        if cro and TS_MIN < cro < TS_MAX:
            signals["code_reuse_order"][sha] = cro
    
    return samples, signals


def compute_features_norm(samples):
    from sklearn.preprocessing import StandardScaler
    features = []
    for d in samples:
        fg = d.get("feature_groups", {})
        vec = list(fg.get("byte_histogram", {}).get("byte_hist", [0.0]*256))
        vec.extend(fg.get("byte_histogram", {}).get("entropy_hist", [0.0]*16))
        vec.extend(fg.get("string_entropy", {}).get("entropy_distribution", [0.0]*16))
        vec.extend(fg.get("cfg_topology", {}).get("percentiles", [0.0]*5))
        sm = fg.get("section_metadata", {}); secs = sm.get("sections", [])
        vec.extend([len(secs), sum(s.get("entropy",0) for s in secs), sum(s.get("vsize",0) for s in secs)])
        vec.append(fg.get("imports", {}).get("n_symbols", 0))
        features.append([float(x) for x in vec])
    X = np.array(features)
    mask = np.var(X, axis=0) > 0; X = X[:, mask]
    X = StandardScaler().fit_transform(X)
    norms = np.linalg.norm(X, axis=1, keepdims=True); norms[norms==0] = 1e-10
    return X / norms


def compute_within_family_triplet_inconsistency(samples, features_norm, temporal_vals, seed=42, max_triplets=5000):
    """Within-family triplet inconsistency: for triplets in same family where
    sim(i,j) > sim(i,k), check if temporal ordering agrees."""
    rng = random.Random(seed)
    n = len(samples)
    
    # Group by family
    by_family = defaultdict(list)
    for i, s in enumerate(samples):
        f = s.get("family", "Unknown")
        if f != "Unknown":
            by_family[f].append(i)
    
    # Only families with ≥3 samples and valid temporal vals
    valid_families = {}
    for fam, indices in by_family.items():
        valid = [i for i in indices if temporal_vals[i] is not None]
        if len(valid) >= 3:
            valid_families[fam] = valid
    
    if not valid_families:
        return 0.5  # no data, random
    
    inconsistent = 0
    checked = 0
    attempts = 0
    
    all_valid = []
    for fam, indices in valid_families.items():
        all_valid.append((fam, indices))
    
    while checked < max_triplets and attempts < max_triplets * 5:
        attempts += 1
        fam, indices = rng.choice(all_valid)
        if len(indices) < 3: continue
        i, j, k = rng.sample(indices, 3)
        
        ti, tj, tk = temporal_vals[i], temporal_vals[j], temporal_vals[k]
        if ti is None or tj is None or tk is None: continue
        
        sim_ij = float(features_norm[i] @ features_norm[j])
        sim_ik = float(features_norm[i] @ features_norm[k])
        if abs(sim_ij - sim_ik) < 0.01: continue
        
        checked += 1
        if sim_ij > sim_ik:
            if abs(ti - tj) > abs(ti - tk): inconsistent += 1
        else:
            if abs(ti - tk) > abs(ti - tj): inconsistent += 1
    
    return inconsistent / checked if checked > 0 else 0.5


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_lines = []
    def log(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True); log_lines.append(line)

    with open(THRESH_PATH) as f:
        thresh = yaml.safe_load(f)
    delta_sig_floor = thresh["Delta_sig_floor"]["value"]
    delta_pct_thresh = thresh["Delta_pct"]["value"]

    log("=" * 70)
    log("A9 REAL v2: Within-family triplet inconsistency (4 real signals, 5 runs)")
    log("=" * 70)

    t0 = time.time()
    samples, signals = load_data()
    for sig in TEMPORAL_SIGNALS:
        log(f"  {sig}: {len(signals[sig])} samples")

    # Use samples with ≥2 signals AND known family
    sig_counts = defaultdict(int)
    for sha in set().union(*[set(signals[s].keys()) for s in TEMPORAL_SIGNALS]):
        for sig in TEMPORAL_SIGNALS:
            if sha in signals[sig]: sig_counts[sha] += 1
    
    eval_shas = {sha for sha, c in sig_counts.items() if c >= 2}
    eval_samples = [s for s in samples if s["sha256"] in eval_shas and s.get("family", "Unknown") != "Unknown"]
    log(f"\n  Evaluation set: {len(eval_samples)} samples (≥2 signals, known family)")

    features_norm = compute_features_norm(eval_samples)

    # Build temporal arrays
    sig_arrays = {}
    for sig in TEMPORAL_SIGNALS:
        arr = [signals[sig].get(s["sha256"], None) for s in eval_samples]
        sig_arrays[sig] = arr

    run_results = []
    for run_idx in range(N_RUNS):
        seed = 42 + run_idx
        log(f"\n=== Run {run_idx+1}/{N_RUNS} (seed={seed}) ===")

        # Full multi-signal: median of available signals per sample
        full_temporal = []
        for i, s in enumerate(eval_samples):
            vals = [sig_arrays[sig][i] for sig in TEMPORAL_SIGNALS if sig_arrays[sig][i] is not None]
            full_temporal.append(np.median(vals) if vals else None)

        full_inc = compute_within_family_triplet_inconsistency(eval_samples, features_norm, full_temporal, seed)
        log(f"  Full multi-signal inconsistency: {full_inc:.4f}")

        # Single-signal baselines
        single_inc = {}
        for sig in TEMPORAL_SIGNALS:
            vals = sig_arrays[sig]
            inc = compute_within_family_triplet_inconsistency(eval_samples, features_norm, vals, seed)
            single_inc[sig] = inc
            log(f"  {sig}-only inconsistency: {inc:.4f}")

        # LOO
        loo_results = {}
        for drop_sig in TEMPORAL_SIGNALS:
            remaining = [s for s in TEMPORAL_SIGNALS if s != drop_sig]
            temporal_loo = []
            for i, s in enumerate(eval_samples):
                vals = [sig_arrays[sig][i] for sig in remaining if sig_arrays[sig][i] is not None]
                temporal_loo.append(np.median(vals) if vals else None)
            
            inc_without = compute_within_family_triplet_inconsistency(eval_samples, features_norm, temporal_loo, seed)
            delta = (inc_without - full_inc) * 100  # positive = signal was helping
            loo_results[drop_sig] = {
                "inconsistency_without": round(inc_without, 4),
                "delta_pct": round(delta, 2),
                "survives": delta >= delta_sig_floor,
            }
            status = "SURVIVES" if delta >= delta_sig_floor else "Below floor"
            log(f"  Drop {drop_sig}: inc={inc_without:.4f}, Δ={delta:+.2f}% - {status}")

        n_surviving = sum(1 for v in loo_results.values() if v["survives"])
        
        # C3 delta: multi-signal vs first_seen-only (VT equivalent)
        c3_delta = (single_inc["first_seen"] - full_inc) * 100

        log(f"\n  Surviving: {n_surviving}/4 (≥2 required)")
        log(f"  C3 Δ (first_seen-only → multi): {c3_delta:+.2f}% (floor: {delta_pct_thresh}%)")

        run_results.append({
            "run": run_idx + 1, "seed": seed, "n_eval": len(eval_samples),
            "full_inconsistency": round(full_inc, 4),
            "single_inconsistency": {k: round(v, 4) for k, v in single_inc.items()},
            "loo_results": loo_results, "n_surviving": n_surviving,
            "c3_delta_pct": round(c3_delta, 2),
        })

    # Stability
    all_pass = all(r["n_surviving"] >= 2 and r["c3_delta_pct"] >= delta_pct_thresh for r in run_results)
    c3_deltas = [r["c3_delta_pct"] for r in run_results]
    surviving = [r["n_surviving"] for r in run_results]
    c3_std = float(np.std(c3_deltas))
    stable = c3_std < 2.0 and len(set(surviving)) == 1

    log(f"\n{'='*70}")
    log(f"STABILITY CHECK ({N_RUNS} runs)")
    log(f"  All PASS: {all_pass}")
    log(f"  C3 Δ: mean={np.mean(c3_deltas):.2f}%, std={c3_std:.2f} - {c3_deltas}")
    log(f"  Surviving: {surviving}")
    log(f"  Stable: {stable}")
    final = "PASS (STABLE)" if all_pass and stable else ("STABLE but FAIL" if stable else "UNSTABLE")
    log(f"  Final: {final}")
    log(f"{'='*70}")

    metrics = {
        "run_id": "R-ABLATE-TEMP-LOO-REAL", "version": "v2",
        "milestone": "A9 REAL within-family triplet inconsistency",
        "gate": "G4", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "COMPLETED", "verdict": final,
        "n_runs": N_RUNS, "all_pass": all_pass, "stable": stable,
        "temporal_signals": TEMPORAL_SIGNALS,
        "signal_availability": {sig: len(signals[sig]) for sig in TEMPORAL_SIGNALS},
        "n_eval_samples": len(eval_samples),
        "run_results": run_results,
        "thresholds": {"Delta_sig_floor": delta_sig_floor, "Delta_pct": delta_pct_thresh},
        "metric": "within_family_triplet_inconsistency",
        "evaluation_type": "REAL temporal signals",
        "timing_seconds": round(time.time() - t0, 2),
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    (OUT_DIR / "run.log").write_text("\n".join(log_lines) + "\n")
    with open(OUT_DIR / "config.yaml", "w") as f:
        yaml.dump({"run_id": "R-ABLATE-TEMP-LOO-REAL", "version": "v2", "n_runs": N_RUNS}, f, default_flow_style=False)
    log(f"\nOutputs: {OUT_DIR}/")
    return metrics


if __name__ == "__main__":
    m = main()
    print(json.dumps({"verdict": m["verdict"], "stable": m["stable"],
                      "c3_delta_mean": round(float(np.mean([r["c3_delta_pct"] for r in m["run_results"]])), 2),
                      "surviving": m["run_results"][0]["n_surviving"]}, indent=2))
