#!/usr/bin/env python3
"""R-G1: Feature extraction (φ) on real Win32 PE binaries from MalwareBazaar.

This is the FIRST real MalPhylo-on-binary run. Extracts interpretable features
from raw PE samples stored as password-protected ZIPs (password "infected").

Gate G1 checks (evaluation design):
  - coverage %: fraction of manifest samples with raw PE on disk that parse
  - packing log: which samples are detected as packed (Section 3.3)
  - feature groups extracted per sample (6 groups per Section 3.3)

Outputs (per artifact layout specification):
  results/R-G1/metrics.json, run.log, features.jsonl, packing_log.jsonl, config.yaml

Safety: static analysis only. No malware execution. PE binaries
are extracted in-memory from password-protected ZIPs, parsed by pefile, and
never written to disk as raw executables.

Usage:
  python3 scripts/features/run_g1_features.py [--limit N]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

# pyzipper supports AES-encrypted ZIPs (MalwareBazaar API uses AES-128, pw "infected")
try:
    import pyzipper
    HAS_PYZIPPER = True
except ImportError:
    HAS_PYZIPPER = False

# Add repo root to path
HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent.parent
sys.path.insert(0, str(EXP_DIR))

from malphpylo.config import PackingConfig
from malphpylo.features import (
    SampleFeatures,
    extract_pe_features_from_bytes,
)

ZIP_PW = b"infected"
SAMPLES_DIR = EXP_DIR / "data" / "malwarebazaar" / "samples"
MANIFEST_PATH = EXP_DIR / "data" / "malwarebazaar" / "manifest.jsonl"
OUT_DIR = EXP_DIR / "results" / "R-G1"


# ===========================================================================
# Serialization: SampleFeatures → JSON-safe dict
# ===========================================================================
def serialize_features(sf: SampleFeatures) -> Dict[str, Any]:
    """Convert SampleFeatures dataclass to a JSON-serializable dict."""
    return {
        "sample_id": sf.sample_id,
        "family": sf.family,
        "sha256": sf.sha256,
        "platform": sf.platform,
        "packed": sf.packed,
        "trusted_groups": sorted(sf.trusted_groups()),
        "temporal_signals": {k: v for k, v in sf.temporal_signals.items()},
        "feature_groups": {
            "opcode_ngrams": {
                "trusted": sf.opcode_ngrams.trusted,
                "dim": len(sf.opcode_ngrams.distribution),
            },
            "cfg_topology": {
                "trusted": sf.cfg_topology.trusted,
                "n_functions": sf.cfg_topology.n_functions,
                "percentiles": (
                    sf.cfg_topology.percentiles.tolist()
                    if len(sf.cfg_topology.percentiles) > 0
                    else []
                ),
            },
            "imports": {
                "trusted": sf.imports.trusted,
                "n_symbols": len(sf.imports.symbol_set),
                "symbol_set": sorted(sf.imports.symbol_set)[:200],  # cap for storage
            },
            "string_entropy": {
                "trusted": sf.string_entropy.trusted,
                "entropy_distribution": sf.string_entropy.entropy_distribution.tolist(),
                "n_rare_strings": len(sf.string_entropy.rare_string_set),
                "rare_strings": sorted(sf.string_entropy.rare_string_set)[:100],  # cap
            },
            "section_metadata": {
                "trusted": sf.section_metadata.trusted,
                "n_sections": len(sf.section_metadata.sections),
                "sections": sf.section_metadata.sections,
            },
            "byte_histogram": {
                "trusted": sf.byte_histogram.trusted,
                "byte_hist": sf.byte_histogram.byte_hist.tolist(),
                "entropy_hist": sf.byte_histogram.entropy_hist.tolist(),
            },
        },
    }


# ===========================================================================
# PE extraction from password-protected ZIP
# ===========================================================================
def extract_pe_from_zip(zip_path: Path) -> Optional[bytes]:
    """Extract the PE binary from a password-protected ZIP (pw 'infected').
    Handles both standard ZipCrypto and AES-128 encrypted ZIPs.
    Returns raw PE bytes or None on failure."""
    pw = b"infected"

    # Try pyzipper first (supports AES-encrypted ZIPs from MalwareBazaar API)
    if HAS_PYZIPPER:
        try:
            with pyzipper.AESZipFile(zip_path) as zf:
                names = zf.namelist()
                if not names:
                    return None
                raw = zf.read(names[0], pwd=pw)
                if raw[:2] == b"MZ":
                    return raw
                return None
        except Exception:
            pass  # fall through to standard zipfile

    # Fallback: standard zipfile (works for unencrypted or ZipCrypto-encrypted ZIPs)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            if not names:
                return None
            raw = zf.read(names[0], pwd=pw)
            if raw[:2] == b"MZ":
                return raw
            return None
    except Exception:
        return None


# ===========================================================================
# Main runner
# ===========================================================================
def run(limit: Optional[int] = None, config_path: Optional[str] = None) -> Dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log_lines: List[str] = []
    def log(msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    # Load packing config
    packing_cfg = PackingConfig()
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            cfg_raw = yaml.safe_load(f)
        if "packing" in cfg_raw:
            packing_cfg = PackingConfig(
                section_entropy_threshold=cfg_raw["packing"].get("section_entropy_threshold", 7.2),
                min_code_entropy=cfg_raw["packing"].get("min_code_entropy", 6.5),
            )

    log(f"R-G1 start | packing_cfg: entropy_thresh={packing_cfg.section_entropy_threshold}, "
        f"min_code_entropy={packing_cfg.min_code_entropy}")

    # Load manifest → map sha256 → family
    sha_to_family: Dict[str, str] = {}
    sha_to_meta: Dict[str, Dict] = {}
    with open(MANIFEST_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            sha = d.get("sha256", "")
            fam = d.get("family", "Unknown")
            sha_to_family[sha] = fam
            sha_to_meta[sha] = d

    # Find raw PE ZIPs on disk
    sample_zips = sorted(SAMPLES_DIR.glob("*.zip"))
    if limit:
        sample_zips = sample_zips[:limit]

    log(f"Manifest entries: {len(sha_to_family)}")
    log(f"Raw PE ZIPs on disk: {len(list(SAMPLES_DIR.glob('*.zip')))}")
    log(f"Processing: {len(sample_zips)} samples" + (f" (limit={limit})" if limit else ""))

    # Extract features
    t0 = time.time()
    features_list: List[Dict] = []
    packing_log: List[Dict] = []
    parse_ok = 0
    parse_fail = 0
    packed_count = 0
    fallback_count = 0
    error_log: List[Dict] = []
    family_counter: Counter = Counter()

    features_path = OUT_DIR / "features.jsonl"
    packing_log_path = OUT_DIR / "packing_log.jsonl"

    with open(features_path, "w") as feat_f, open(packing_log_path, "w") as pack_f:
        for i, zp in enumerate(sample_zips):
            sha = zp.stem
            fam = sha_to_family.get(sha, "Unknown")
            family_counter[fam] += 1

            if (i + 1) % 50 == 0 or i == 0:
                log(f"[{i+1}/{len(sample_zips)}] {sha[:12]}... family={fam}")

            # Extract PE from ZIP
            pe_data = extract_pe_from_zip(zp)
            if pe_data is None:
                parse_fail += 1
                error_log.append({"sha256": sha, "error": "zip_extract_failed"})
                continue

            # Parse PE bytes directly in memory (zero raw executables written to disk)
            try:
                sf = extract_pe_features_from_bytes(pe_data, sha, fam, packing_cfg)

                if sf.packed:
                    packed_count += 1
                    packing_log.append({
                        "sha256": sha,
                        "family": fam,
                        "packed": True,
                        "trusted_groups": sorted(sf.trusted_groups()),
                    })
                    pack_f.write(json.dumps(packing_log[-1]) + "\n")

                # Check if fell back to byte-path (no PE parse)
                # Heuristic: if imports symbol_set is empty and sections is empty → fallback
                if len(sf.imports.symbol_set) == 0 and len(sf.section_metadata.sections) == 0:
                    fallback_count += 1

                feat_dict = serialize_features(sf)
                feat_f.write(json.dumps(feat_dict) + "\n")
                features_list.append(feat_dict)
                parse_ok += 1

            except Exception as e:
                parse_fail += 1
                error_log.append({"sha256": sha, "error": str(e)[:200]})

    elapsed = time.time() - t0

    # Compute metrics
    total_processed = parse_ok + parse_fail
    coverage_pct = round(parse_ok / total_processed * 100, 2) if total_processed > 0 else 0.0

    # Feature group trust stats
    group_trust: Dict[str, int] = {}
    group_untrusted: Dict[str, int] = {}
    for feat in features_list:
        for gname, gdata in feat["feature_groups"].items():
            if gdata["trusted"]:
                group_trust[gname] = group_trust.get(gname, 0) + 1
            else:
                group_untrusted[gname] = group_untrusted.get(gname, 0) + 1

    metrics = {
        "run_id": "R-G1",
        "milestone": "feature integrity",
        "gate": "G1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "COMPLETED",
        "verdict": "PASS" if coverage_pct >= 80.0 else ("PARTIAL" if coverage_pct >= 50.0 else "FAIL"),
        "summary": {
            "total_samples_on_disk": len(list(SAMPLES_DIR.glob("*.zip"))),
            "samples_processed": total_processed,
            "parse_ok": parse_ok,
            "parse_fail": parse_fail,
            "coverage_pct": coverage_pct,
            "packed_count": packed_count,
            "packed_pct": round(packed_count / parse_ok * 100, 2) if parse_ok > 0 else 0.0,
            "fallback_to_byte_path": fallback_count,
            "fallback_pct": round(fallback_count / parse_ok * 100, 2) if parse_ok > 0 else 0.0,
            "families_seen": dict(sorted(family_counter.items(), key=lambda x: -x[1])[:20]),
            "n_families": len(family_counter),
        },
        "feature_group_trust": {
            g: {
                "trusted": group_trust.get(g, 0),
                "untrusted": group_untrusted.get(g, 0),
            }
            for g in ["opcode_ngrams", "cfg_topology", "imports", "string_entropy",
                       "section_metadata", "byte_histogram"]
        },
        "timing": {
            "total_seconds": round(elapsed, 2),
            "per_sample_seconds": round(elapsed / total_processed, 4) if total_processed > 0 else 0,
        },
        "errors": error_log[:50],
        "n_errors_total": len(error_log),
        "artifacts": {
            "features": str(features_path),
            "packing_log": str(packing_log_path),
        },
    }

    metrics_path = OUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    (OUT_DIR / "run.log").write_text("\n".join(log_lines) + "\n")

    log(f"R-G1 COMPLETE | coverage={coverage_pct}% ({parse_ok}/{total_processed})")
    log(f"  Packed: {packed_count} ({metrics['summary']['packed_pct']}%)")
    log(f"  Fallback to byte-path: {fallback_count} ({metrics['summary']['fallback_pct']}%)")
    log(f"  Families: {len(family_counter)}")
    log(f"  Time: {elapsed:.1f}s ({metrics['timing']['per_sample_seconds']}s/sample)")
    log(f"  Verdict: {metrics['verdict']}")
    log(f"  Written: {features_path}, {packing_log_path}, {metrics_path}")

    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="R-G1: φ feature extraction on real PE")
    ap.add_argument("--limit", type=int, default=None, help="limit N samples (for smoke)")
    ap.add_argument("--config", default=None, help="optional YAML config path")
    args = ap.parse_args()
    m = run(limit=args.limit, config_path=args.config)
    print(json.dumps({k: m[k] for k in ("run_id", "status", "verdict", "summary", "timing")},
                     indent=2, default=str))
