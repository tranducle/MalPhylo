"""R-SMOKE-1 pipeline runner + synthetic data generator.

Per experiment_plan.md §4: R-SMOKE-1 = sanity run, pipeline end-to-end on 1K
samples. Goal: completes, no crash. Labeled SMOKE — precedes G2.

This module provides:
  - run_pipeline(cfg): end-to-end run from config
  - generate_synthetic_corpus(n, seed): deterministic synthetic malware-like
    samples so the smoke test can execute without real corpora
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from .config import RunConfig, load_config
from .dag import DAGConstructionResult, construct_dag
from .features import SampleFeatures, extract
from .logging_utils import EvidenceLedger, dump_metrics, setup_logger


# ---------------------------------------------------------------------------
# Synthetic corpus — deterministic, no real malware needed for smoke test
# ---------------------------------------------------------------------------

_FAMILIES = [
    "mirai", "emotet", "trickbot", "wannacry", "agenttesla",
    "qakbot", "dridex", "remcos", "njrat", "nanocore",
    "redline", "agenttesla2",
]


def _family_blob(family: str, variant: int, seed: int) -> bytes:
    """Deterministic per-family byte blob simulating family code-sharing.

    Design: large low-entropy family signature (ASCII repetition ~1.5KB) so
    intra-family byte-histogram similarity is high AND section entropy stays
    below the packing threshold (§3.2: 7.2 bits). Small variant noise (256B)
    distinguishes variants without overwhelming the signature.
    """
    seed_key = abs(hash((family, variant, seed))) % (2 ** 32)
    rng = random.Random(seed_key)
    # family signature: repeated ASCII pattern (low entropy, high intra-family sim)
    fam_sig = (family.encode() + b"\x00") * 200  # ~1.6-2.0 KB
    # variant-specific deterministic noise (small fraction of total)
    noise = bytes(rng.randint(0, 255) for _ in range(256))
    # builder/kit string marker (toolkit-share signal, §4.2 rule 2)
    builder = f"BUILDER_{family.upper()}_{variant % 3}".encode()
    blob = fam_sig + noise + builder
    return blob


# Shared cross-family builder kit (substantial, so byte-histogram + string
# overlap create toolkit-share candidates across families, §4.2 rule 2)
_SHARED_KIT = (
    b"MIRAI_BUILDER_CORE_v2.3\x00" * 40 +
    b"telnet_scan_module\x00" * 20 +
    b"creds_list_default\x00" * 20 +
    b"SELF_REPLICATE_FUNC\x00" * 20
)  # ~2 KB shared kit payload


def generate_synthetic_corpus(n: int,
                              seed: int = 42,
                              platform: str = "win32_pe",
                              n_families: int = 12) -> List[SampleFeatures]:
    """Generate n deterministic synthetic samples for the smoke test.

    Design (exercises all §4.1/§4.2 code paths):
      - intra-family: large shared signature -> high similarity (MST clusters)
      - cross-family: ~15% of samples across 2+ families embed _SHARED_KIT
        -> toolkit-share candidates (§4.2 rule 2) for augmentation path
      - temporal: deterministic PE timestamps spread across 2020-2025
    """
    rng = random.Random(seed)
    families = _FAMILIES[:n_families]
    samples: List[SampleFeatures] = []
    base_ts = 1_600_000_000  # 2020-ish epoch

    for i in range(n):
        fam = families[i % n_families]
        variant = i // n_families
        blob = _family_blob(fam, variant, seed)
        # inject cross-family builder-kit sharing for ~15% of samples
        # (2+ families share the kit -> toolkit-share augmentation candidates)
        if i % 7 == 0 and variant > 0:
            blob = blob + _SHARED_KIT
        # temporal signal: spread deterministically
        ts = base_ts + (variant * 86400 * 30) + (i * 3600)
        sid = f"syn_{i:05d}_{fam}"
        feat = extract(blob, sid, fam, platform,
                       __import__("malphpylo.config", fromlist=["PackingConfig"]).PackingConfig())
        feat.temporal_signals = {"pe_timestamp": float(ts)}
        samples.append(feat)
    return samples


# ---------------------------------------------------------------------------
# End-to-end pipeline (§4.1 + §4.2 + §4.3 + evidence ledger)
# ---------------------------------------------------------------------------

def run_pipeline(cfg: RunConfig) -> Dict:
    """Execute the full MalPhylo pipeline for one run config.

    Stages (mirrors §4.1 + scientific gate ladder):
      G1: feature extraction (R-G1)
      similarity computation
      G2: DAG construction (MST skeleton + augmentation + V_I + acyclicity)
      emit metrics.json + evidence ledger + GraphML
    """
    run_dir = Path(cfg.output_dir) / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir, cfg.run_id)
    logger.info(f"=== START run {cfg.run_id} (type={cfg.run_type}) ===")

    t0 = time.time()
    metrics: Dict = {"run_id": cfg.run_id, "run_type": cfg.run_type, "seed": cfg.seed}

    with EvidenceLedger(run_dir / "evidence.jsonl") as ledger:
        # ---- G1: feature extraction (or load synthetic) ----
        logger.info("Stage G1: feature extraction")
        t_g1 = time.time()
        if cfg.dataset_path == "synthetic":
            n = cfg.sample_limit or 1000
            samples = generate_synthetic_corpus(n, seed=cfg.seed, platform=cfg.platform)
        else:
            # real corpus: scan directory for PE/APK files
            dpath = Path(cfg.dataset_path)
            exts = {".exe", ".dll"} if cfg.platform == "win32_pe" else {".apk"}
            files = [p for p in dpath.rglob("*") if p.suffix.lower() in exts]
            if cfg.sample_limit:
                files = files[:cfg.sample_limit]
            samples = []
            for p in files:
                try:
                    fam = p.parent.name or None
                    s = extract(p, p.stem, fam, cfg.platform, cfg.packing)
                    samples.append(s)
                except Exception as e:
                    ledger.record("feature_extraction_error", {"path": str(p), "err": str(e)})
        t_g1_dur = time.time() - t_g1
        metrics["n_samples"] = len(samples)
        metrics["g1_feature_extraction_seconds"] = round(t_g1_dur, 3)
        metrics["n_packed"] = sum(1 for s in samples if s.packed)
        families = set(s.family for s in samples if s.family)
        metrics["n_families"] = len(families)
        ledger.record("g1_complete", {
            "n_samples": len(samples), "n_families": len(families),
            "n_packed": metrics["n_packed"], "duration_s": t_g1_dur,
        })
        logger.info(f"G1 done: {len(samples)} samples, {metrics['n_packed']} packed, "
                    f"{len(families)} families, {t_g1_dur:.2f}s")

        # ---- DAG construction (§4.1: similarity + MST + augment + V_I + Kahn) ----
        logger.info("Stage G2: DAG construction (§4.1)")
        t_g2 = time.time()
        result: DAGConstructionResult = construct_dag(samples, cfg)
        t_g2_dur = time.time() - t_g2
        metrics["g2_dag_construction_seconds"] = round(t_g2_dur, 3)
        metrics["n_vertices"] = result.graph.number_of_nodes()
        metrics["n_internal_VI"] = result.n_internal_inserted
        metrics["n_edges_total"] = result.graph.number_of_edges()
        metrics["n_mst_edges"] = result.n_mst_edges
        metrics["n_augment_edges"] = result.n_augment_edges
        metrics["n_bottom_pairs"] = result.bottom_pairs
        metrics["n_rejected_candidates"] = len(result.rejected)
        metrics["acyclicity_verified"] = result.acyclicity_verified
        # edge-type distribution
        label_counts: Dict[str, int] = {}
        for e in result.edges:
            label_counts[e.label] = label_counts.get(e.label, 0) + 1
        metrics["edge_type_distribution"] = label_counts
        ledger.record("g2_complete", {
            "n_vertices": metrics["n_vertices"], "n_edges": metrics["n_edges_total"],
            "n_mst": result.n_mst_edges, "n_augment": result.n_augment_edges,
            "acyclic": result.acyclicity_verified, "duration_s": t_g2_dur,
            "edge_types": label_counts,
        })
        logger.info(f"G2 done: {metrics['n_vertices']} vertices, "
                    f"{metrics['n_edges_total']} edges "
                    f"(MST={result.n_mst_edges}, augment={result.n_augment_edges}), "
                    f"acyclic={result.acyclicity_verified}, {t_g2_dur:.2f}s")

        # ---- write GraphML (per §5 output artifacts) ----
        nx.write_graphml(result.graph, run_dir / "dag.graphml")

        # ---- write rejected candidates ledger ----
        with open(run_dir / "rejected_candidates.json", "w") as f:
            json.dump(result.rejected, f, indent=2, default=str)

    total = time.time() - t0
    metrics["total_seconds"] = round(total, 3)
    metrics["status"] = "COMPLETE" if result.acyclicity_verified else "ACYCLICITY_FAILED"
    dump_metrics(run_dir, metrics)
    logger.info(f"=== END run {cfg.run_id}: {metrics['status']} in {total:.2f}s ===")
    return metrics


def run_from_config_path(config_path: str | Path) -> Dict:
    """Convenience: load config + run pipeline."""
    cfg = load_config(config_path)
    return run_pipeline(cfg)
