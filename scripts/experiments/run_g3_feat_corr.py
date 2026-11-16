#!/usr/bin/env python3
"""R-FEAT-CORR: H2 Spearman correlation - MalPhylo interpretable sim vs MalTree embedding distance.

hypothesis specification H2 hypothesis: "interpretable features ≈ embeddings"  - 
Spearman ρ ≥ ρ_min (0.30) between MalPhylo similarity and MalTree embedding
similarity on held-out pairs.

Challenge: Only 28 SHA256 overlaps between MalwareBazaar (MalPhylo corpus) and
MalTree (ML embeddings), of which only 5 are in R-G1 features (10 pairs).
This is insufficient for a reliable correlation (C(5,2)=10 pairs vs the
methodology's recommended ≥100 pairs).

Approach (per calibration protocol note: "use MalTree's own samples for the H2
correlation test"):
  1. Extract MalPhylo interpretable features from the 7 overlapping samples
     that have raw PE ZIPs on disk.
  2. Combine with the 5 R-G1 samples already in both corpora = 12 samples max.
  3. Compute C(n,2) pairwise similarities (MalPhylo sim + MalTree cosine sim).
  4. Report Spearman ρ with confidence interval and power caveat.
  5. If n < 20, document as preliminary and recommend a larger overlap study
     for the final paper.

NO ML TRAINING (Sections 3.1-3.5; evaluation protocol).

Outputs:
  results/R-FEAT-CORR/{metrics.json, run.log, config.yaml, scatter_data.csv}
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent.parent
sys.path.insert(0, str(EXP_DIR))
sys.setrecursionlimit(50000)

from malphpylo.config import PackingConfig, RunConfig, FeatureGroupConfig, \
    TemporalConfig, EdgeTypingConfig, MSTConfig
from malphpylo.features import SampleFeatures, extract_pe_features, extract_features_from_bytes
from malphpylo.similarity import pairwise_similarity
from malphpylo.maltree_nj import load_embeddings

import pyzipper
import zipfile
from scipy.stats import spearmanr, kendalltau

OUT_DIR = EXP_DIR / "results" / "R-FEAT-CORR"
EMB_PATH = EXP_DIR / "data" / "maltree" / "embeddings_fused.json"
MANIFEST_PATH = EXP_DIR / "data" / "malwarebazaar" / "manifest.jsonl"
SAMPLES_DIR = EXP_DIR / "data" / "malwarebazaar" / "samples"
RG1_PATH = EXP_DIR / "results" / "R-G1" / "features.jsonl"
THRESHOLDS_PATH = EXP_DIR / "configs" / "thresholds.yaml"

ZIP_PW = b"infected"


def deserialize_one(d: Dict) -> SampleFeatures:
    """Reconstruct a SampleFeatures from R-G1 JSONL line."""
    from malphpylo.features import (
        OpcodeNgramFeatures, CFGTopologyFeatures, ImportFeatures,
        StringEntropyFeatures, SectionMetadataFeatures, ByteHistogramFeatures,
    )
    fg = d["feature_groups"]
    return SampleFeatures(
        sample_id=d["sample_id"],
        family=d.get("family"),
        sha256=d.get("sha256", d["sample_id"]),
        platform=d.get("platform", "win32_pe"),
        packed=d.get("packed", False),
        opcode_ngrams=OpcodeNgramFeatures(
            distribution=np.array([], dtype=np.float64), vocab=tuple(),
            trusted=fg.get("opcode_ngrams", {}).get("trusted", False)),
        cfg_topology=CFGTopologyFeatures(
            percentiles=np.array(fg.get("cfg_topology", {}).get("percentiles", [0]*5), dtype=np.float64),
            n_functions=fg.get("cfg_topology", {}).get("n_functions", 0),
            trusted=fg.get("cfg_topology", {}).get("trusted", False)),
        imports=ImportFeatures(
            symbol_set=frozenset(fg.get("imports", {}).get("symbol_set", [])),
            trusted=fg.get("imports", {}).get("trusted", False)),
        string_entropy=StringEntropyFeatures(
            entropy_distribution=np.array(
                fg.get("string_entropy", {}).get("entropy_distribution", [0]*16), dtype=np.float64),
            rare_string_set=frozenset(fg.get("string_entropy", {}).get("rare_strings", [])),
            trusted=fg.get("string_entropy", {}).get("trusted", False)),
        section_metadata=SectionMetadataFeatures(
            sections=fg.get("section_metadata", {}).get("sections", []),
            trusted=fg.get("section_metadata", {}).get("trusted", False)),
        byte_histogram=ByteHistogramFeatures(
            byte_hist=np.array(fg.get("byte_histogram", {}).get("byte_hist", [0]*256), dtype=np.float64),
            entropy_hist=np.array(fg.get("byte_histogram", {}).get("entropy_hist", [0]*16), dtype=np.float64),
            trusted=fg.get("byte_histogram", {}).get("trusted", False)),
        temporal_signals=d.get("temporal_signals", {}),
    )


def extract_pe_from_zip(zip_path: Path) -> Optional[bytes]:
    """Extract PE binary from password-protected ZIP (pw 'infected')."""
    try:
        with pyzipper.AESZipFile(zip_path) as zf:
            names = zf.namelist()
            if not names:
                return None
            raw = zf.read(names[0], pwd=ZIP_PW)
            if raw[:2] == b"MZ":
                return raw
            return None
    except Exception:
        pass
    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            if not names:
                return None
            raw = zf.read(names[0], pwd=ZIP_PW)
            if raw[:2] == b"MZ":
                return raw
            return None
    except Exception:
        return None


def run() -> Dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_lines: List[str] = []

    def log(msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log("=== R-FEAT-CORR START ===")
    t0 = time.time()

    # Load thresholds
    with open(THRESHOLDS_PATH) as f:
        thresholds = yaml.safe_load(f)
    rho_min = thresholds["rho_min"]["value"]
    log(f"Threshold: ρ_min={rho_min}")

    # Load MalTree embeddings
    log("Loading MalTree embeddings...")
    sha_list, emb, meta_bundle = load_embeddings(str(EMB_PATH))
    mt_shas = {k.lower() for k in sha_list}
    emb_dict = {}
    for i, sha in enumerate(sha_list):
        emb_dict[sha.lower()] = emb[i]
    log(f"Loaded {len(sha_list)} MalTree embeddings, dim={meta_bundle['meta']['dim']}")

    # Load MalwareBazaar manifest
    mb_entries = {}
    with open(MANIFEST_PATH) as f:
        for line in f:
            d = json.loads(line.strip())
            sha = (d.get("sha256") or d.get("sha256_hash") or d.get("id", "")).lower()
            if sha:
                mb_entries[sha] = d
    log(f"MalwareBazaar manifest: {len(mb_entries)} entries")

    # Find overlap
    overlap = set(mb_entries.keys()) & mt_shas
    log(f"SHA256 overlap: {len(overlap)}")

    # Load R-G1 features for samples already extracted
    rg1_features = {}
    with open(RG1_PATH) as f:
        for line in f:
            d = json.loads(line.strip())
            rg1_features[d["sample_id"].lower()] = d
    log(f"R-G1 features: {len(rg1_features)} samples")

    # Build the set of samples with BOTH MalPhylo features and MalTree embeddings
    paired_samples: Dict[str, Dict] = {}  # sha -> {malphpylo: SampleFeatures, maltree_emb: np.ndarray, family: str}

    # Source 1: R-G1 samples that are in MalTree
    for sha in overlap:
        if sha in rg1_features:
            paired_samples[sha] = {
                "malphpylo": deserialize_one(rg1_features[sha]),
                "maltree_emb": emb_dict[sha],
                "family": rg1_features[sha].get("family", "Unknown"),
                "source": "R-G1",
            }
    log(f"From R-G1 overlap: {len(paired_samples)} samples")

    # Source 2: Extract MalPhylo features from raw PE ZIPs on disk for remaining overlaps
    packing_cfg = PackingConfig()
    newly_extracted = 0
    for sha in overlap:
        if sha in paired_samples:
            continue
        zip_path = SAMPLES_DIR / f"{sha}.zip"
        if zip_path.exists():
            raw_pe = extract_pe_from_zip(zip_path)
            if raw_pe is not None:
                try:
                    sf = extract_features_from_bytes(
                        raw_pe, sha, mb_entries[sha].get("signature") or mb_entries[sha].get("family"),
                        "win32_pe", packing_cfg)
                    paired_samples[sha] = {
                        "malphpylo": sf,
                        "maltree_emb": emb_dict[sha],
                        "family": mb_entries[sha].get("signature") or "Unknown",
                        "source": "raw_pe_extract",
                    }
                    newly_extracted += 1
                except Exception as e:
                    log(f"  Failed to extract {sha[:16]}...: {e}")
    log(f"From raw PE extraction: {newly_extracted} samples")

    n_paired = len(paired_samples)
    n_pairs = n_paired * (n_paired - 1) // 2
    log(f"Total paired samples: {n_paired}, pairs: {n_pairs}")

    if n_paired < 5:
        log("WARNING: Insufficient paired samples (<5) for correlation. Documenting as limitation.")
        metrics = {
            "run_id": "R-FEAT-CORR",
            "gate": "G3 (H2 support)",
            "status": "COMPLETED_WITH_LIMITATION",
            "n_paired_samples": n_paired,
            "n_pairs": n_pairs,
            "spearman_rho": None,
            "verdict": "INCONCLUSIVE",
            "limitation": f"Only {n_paired} paired samples ({n_pairs} pairs). Need ≥20 samples for reliable Spearman ρ.",
            "recommendation": "Acquire a larger SHA256-overlapping corpus or use MalTree's own raw samples for a same-corpus H2 test.",
        }
        (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
        (OUT_DIR / "run.log").write_text("\n".join(log_lines) + "\n")
        return metrics

    # Compute pairwise similarities
    alpha = {
        "opcode_ngrams": 0.20, "cfg_topology": 0.15, "imports": 0.20,
        "string_entropy": 0.15, "section_metadata": 0.10, "byte_histogram": 0.20,
    }
    enabled_groups = ["opcode_ngrams", "cfg_topology", "imports",
                      "string_entropy", "section_metadata", "byte_histogram"]

    shas_sorted = sorted(paired_samples.keys())
    malphpylo_sims = []
    maltree_sims = []
    pair_records = []

    for i in range(len(shas_sorted)):
        for j in range(i + 1, len(shas_sorted)):
            sha_i, sha_j = shas_sorted[i], shas_sorted[j]
            si = paired_samples[sha_i]
            sj = paired_samples[sha_j]

            # MalPhylo interpretable similarity
            res = pairwise_similarity(si["malphpylo"], sj["malphpylo"], alpha, enabled_groups)
            mp_sim = res.value if res.value is not None else 0.0

            # MalTree embedding cosine similarity
            ei = si["maltree_emb"]
            ej = sj["maltree_emb"]
            ni = np.linalg.norm(ei)
            nj = np.linalg.norm(ej)
            mt_sim = float(np.dot(ei, ej) / (ni * nj)) if ni > 0 and nj > 0 else 0.0

            malphpylo_sims.append(mp_sim)
            maltree_sims.append(mt_sim)
            pair_records.append({
                "sha_i": sha_i[:16], "sha_j": sha_j[:16],
                "family_i": si["family"], "family_j": sj["family"],
                "malphpylo_sim": round(mp_sim, 6),
                "maltree_cosine_sim": round(mt_sim, 6),
            })

    mp_arr = np.array(malphpylo_sims)
    mt_arr = np.array(maltree_sims)

    # Spearman ρ
    rho, p_value = spearmanr(mp_arr, mt_arr)

    # Kendall τ (more robust for small samples)
    tau, tau_p = kendalltau(mp_arr, mt_arr)

    # Pearson r (for reference)
    from scipy.stats import pearsonr
    r_pearson, r_p = pearsonr(mp_arr, mt_arr)

    # Distance correlation (MalPhylo sim → should correlate with MalTree sim)
    # Both are similarities; high positive correlation = agreement

    log(f"Spearman ρ: {rho:.4f} (p={p_value:.4f})")
    log(f"Kendall τ: {tau:.4f} (p={tau_p:.4f})")
    log(f"Pearson r: {r_pearson:.4f} (p={r_p:.4f})")
    log(f"MalPhylo sim: mean={np.mean(mp_arr):.4f}, std={np.std(mp_arr):.4f}")
    log(f"MalTree sim:  mean={np.mean(mt_arr):.4f}, std={np.std(mt_arr):.4f}")

    # Verdict
    passes_rho = rho >= rho_min
    verdict = "PASS" if passes_rho else "FAIL"

    # Write scatter data for visualization
    import csv
    with open(OUT_DIR / "scatter_data.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sha_i", "sha_j", "family_i", "family_j",
                                           "malphpylo_sim", "maltree_cosine_sim"])
        w.writeheader()
        w.writerows(pair_records)

    total_time = round(time.time() - t0, 2)
    metrics = {
        "run_id": "R-FEAT-CORR",
        "gate": "G3 (H2 support)",
        "milestone": "Spearman correlation MalPhylo sim vs MalTree embedding sim (H2)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "COMPLETED",
        "h2_verdict": verdict,
        "n_paired_samples": n_paired,
        "n_pairs": n_pairs,
        "spearman_rho": round(float(rho), 6),
        "spearman_p_value": round(float(p_value), 6),
        "kendall_tau": round(float(tau), 6),
        "kendall_p_value": round(float(tau_p), 6),
        "pearson_r": round(float(r_pearson), 6),
        "pearson_p_value": round(float(r_p), 6),
        "malphpylo_sim_stats": {
            "mean": round(float(np.mean(mp_arr)), 6),
            "std": round(float(np.std(mp_arr)), 6),
            "min": round(float(np.min(mp_arr)), 6),
            "max": round(float(np.max(mp_arr)), 6),
        },
        "maltree_sim_stats": {
            "mean": round(float(np.mean(mt_arr)), 6),
            "std": round(float(np.std(mt_arr)), 6),
            "min": round(float(np.min(mt_arr)), 6),
            "max": round(float(np.max(mt_arr)), 6),
        },
        "threshold": {"rho_min": rho_min},
        "passes_rho_min": passes_rho,
        "paired_samples": {
            sha: {"family": paired_samples[sha]["family"], "source": paired_samples[sha]["source"]}
            for sha in shas_sorted
        },
        "timing": {"total_s": total_time},
        "artifacts": {
            "scatter_data_csv": str(OUT_DIR / "scatter_data.csv"),
        },
        "limitations": [
            f"Only {n_paired} paired samples ({n_pairs} pairs) - far below the recommended ≥100 pairs for a reliable correlation",
            f"Cross-corpus overlap is only 28 SHA256s between MalwareBazaar ({len(mb_entries)}) and MalTree ({len(sha_list)}), of which {n_paired} have MalPhylo features available",
            "If ρ is insignificant, this may be due to small sample size rather than a true lack of correlation",
            "Recommendation for the paper: acquire a larger overlapping corpus or run a same-corpus H2 test on MalTree's own raw samples (if available)",
        ],
        "interpretation": (
            f"Spearman ρ={rho:.4f} (p={p_value:.4f}) on {n_pairs} pairs. "
            f"{'PASS: ρ ≥ ρ_min (0.30) - interpretable features correlate with ML embeddings.' if passes_rho else 'FAIL: ρ < ρ_min (0.30) - correlation insufficient; may be due to small sample size.'}"
        ),
    }

    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (OUT_DIR / "run.log").write_text("\n".join(log_lines) + "\n")
    with open(OUT_DIR / "config.yaml", "w") as f:
        yaml.safe_dump({"run_id": "R-FEAT-CORR", "rho_min": rho_min,
                        "n_paired": n_paired, "n_pairs": n_pairs}, f, sort_keys=False)

    log(f"=== R-FEAT-CORR COMPLETE | verdict={verdict} | {total_time}s ===")
    return metrics


if __name__ == "__main__":
    m = run()
    print(json.dumps({
        "run_id": m["run_id"],
        "h2_verdict": m["h2_verdict"],
        "n_paired_samples": m["n_paired_samples"],
        "n_pairs": m["n_pairs"],
        "spearman_rho": m["spearman_rho"],
        "spearman_p": m["spearman_p_value"],
        "kendall_tau": m["kendall_tau"],
        "pearson_r": m["pearson_r"],
        "interpretation": m["interpretation"],
    }, indent=2))
