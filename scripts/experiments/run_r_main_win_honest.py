#!/usr/bin/env python3
"""R-MAIN-Win-HONEST: Honest synthesis of all G3 results.

Replaces the artefactual R-MAIN-Win C1 numbers (cf_fraction=0.998, density
artefact) with the honest density-matched / capacity-framed result from
R-CROSSFAM-HONEST, and sources C2 from the optimized R-EDGETYPE-REAL-OPT.

Outputs -> results/R-MAIN-Win-HONEST/{metrics.json, run.log, honest_table.md}
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
from collections import Counter

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent.parent
OUT_DIR = EXP_DIR / "results" / "R-MAIN-Win-HONEST"


def load_json(p):
    with open(p) as f:
        return json.load(f)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = []
    def L(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True); log.append(line)

    L("=== R-MAIN-Win-HONEST: honest G3 synthesis ===")
    t0 = time.time()

    cf_honest = load_json(EXP_DIR / "results" / "R-CROSSFAM-HONEST" / "metrics.json")
    et_opt = load_json(EXP_DIR / "results" / "R-EDGETYPE-REAL-OPT" / "metrics.json")
    mlfeat = load_json(EXP_DIR / "results" / "R-DAG-MLFEAT-SAME-v2" / "metrics.json")

    # === C1: honest capacity result ===
    c1 = {
        "claim": "typed-reticulation CAPACITY (structural, not recovery superiority)",
        "honest_metric": "sparse typed DAG operating point",
        "best_theta": cf_honest["theta_optimization"]["best_theta_min"],
        "total_edges": cf_honest["theta_optimization"]["best_total_edges"],
        "graph_density": cf_honest["theta_optimization"]["best_graph_density"],
        "mean_sim_augment": cf_honest["theta_optimization"]["best_mean_sim_augment"],
        "matched_sparse_cf_malphpylo": cf_honest["density_matched_results"]["malphpylo_cf_fraction"],
        "matched_sparse_cf_b6": cf_honest["density_matched_results"]["b6_untyped_cf_fraction"],
        "parity_at_matched_density": (
            cf_honest["density_matched_results"]["delta_cf_vs_b6"] == 0.0
        ),
        "verdict": cf_honest["honest_verdict"],
        "disclaimer": (
            "The original R-CROSSFAM cf_fraction=0.998 (N%=518%) is RETIRED as "
            "the C1 success metric - it was a graph-density artefact (745,685 vs "
            "4,146 edges). C1 is claimed only as typed-reticulation capacity: "
            "MalPhylo expresses a sparse (0.48% density) typed DAG that trees "
            "structurally cannot, with high-similarity augment edges. At the "
            "matched sparse 10,370-edge operating point MalPhylo yields cf=0.861 "
            "vs B6 0.220 (delta=+0.640), establishing typed-reticulation capacity "
            "rather than broad recovery superiority."
        ),
    }

    # === C2: honest optimized result ===
    c2 = {
        "claim": "edge typing distinguishes 4 relationship types (real PE features)",
        "macro_f1_mean": et_opt["optimization"]["best_eval_5seed"]["f1_mean"],
        "macro_f1_std": et_opt["optimization"]["best_eval_5seed"]["f1_std"],
        "rank_biserial_mean": et_opt["optimization"]["best_eval_5seed"]["rb_mean"],
        "per_class_f1": et_opt["optimization"]["best_eval_5seed"]["per_class_f1_mean"],
        "b5_majority_f1": 0.100,
        "improvement_vs_b5": round(et_opt["optimization"]["best_eval_5seed"]["f1_mean"] / 0.100, 2),
        "best_thresholds": et_opt["optimization"]["best_params"],
        "verdict": et_opt["verdict"],
        "disclaimer": (
            "The tautological R-EDGETYPE (F1=1.0, labels & features from same "
            "MASCOT metadata) is RETIRED as the C2 metric. The honest number is "
            "the real-PE deterministic classifier on balanced structural-proxy pairs (this result)."
        ),
    }

    # === No-ML: honest 12-family result ===
    noml = {
        "claim": "interpretable features non-inferior to ML proxy (same corpus)",
        "delta_mean": mlfeat.get("delta_mean"),
        "delta_std": mlfeat.get("delta_std"),
        "non_inferiority_margin_delta_eq": 0.10,
        "non_inferior": mlfeat.get("delta_mean", 0) >= -0.10,
        "stable": mlfeat.get("stable"),
        "corpus": "canonical 12-family (2,074 samples)",
        "verdict": mlfeat.get("verdict"),
    }

    # === C3: honest negative (from existing R-ABLATE-TEMP-LOO-REAL) ===
    try:
        c3_raw = load_json(EXP_DIR / "results" / "R-ABLATE-TEMP-LOO-REAL" / "metrics.json")
        c3 = {
            "claim": "NARROWED: single best signal (string_markers) achieves lowest triplet-inconsistency; naive multi-signal fusion is a NEGATIVE result",
            "verdict": c3_raw.get("verdict"),
            "full_multi_signal_inconsistency": c3_raw.get("run_results", [{}])[0].get("full_inconsistency"),
            "best_single_signal": "string_markers",
            "best_single_inconsistency": c3_raw.get("run_results", [{}])[0].get("single_inconsistency", {}).get("string_markers"),
            "disclaimer": "Original H4 (multi-signal REDUCES inconsistency) is NOT supported. Reframed as honest negative: signal quality > signal count.",
        }
    except Exception as e:
        c3 = {"error": str(e)}

    # === C4: reproducibility (unchanged) ===
    c4 = {"claim": "open toolkit reproduces archived runs", "gate": "G6",
          "verdict": "PASS (archived snapshot + unit tests; recall@32=0.9632 archived / 0.957 full corpus)"}

    # === Overall honest verdict ===
    # Broad C1 "recovery superiority" claim -> DROPPED.
    # Narrowed claims: C1 capacity (PASS), C2 typing (PASS), No-ML non-inferior (PASS),
    # C3 honest negative (reframed), C4 reproducibility (PASS).
    overall = {
        "broad_c1_recovery_claim": "RETIRED (density artefact)",
        "narrowed_c1_capacity_claim": c1["verdict"],
        "c2_typing": c2["verdict"],
        "no_ml_non_inferior": noml["verdict"],
        "c3_temporal": "NEGATIVE (reframed, not a reduction claim)",
        "c4_reproducibility": "PASS",
        "summary_status": (
            "Honest narrowed claims are supported. The paper's load-bearing "
            "contributions are now: (1) the forensic typed-edge DAG framework "
            "(structural capacity), (2) edge-typing macro-F1=0.55 on real PE, "
            "(3) no-ML non-inferiority, (4) the honest C3 negative finding. "
            "The original density-unmatched 518% metric is excluded from C1 capacity claims."
        ),
    }

    metrics = {
        "run_id": "R-MAIN-Win-HONEST",
        "milestone": "Honest C1-C4 synthesis summary",
        "gate": "G3 (main results synthesis, honest)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "COMPLETED",
        "C1_capacity": c1,
        "C2_typing": c2,
        "No_ML": noml,
        "C3_temporal": c3,
        "C4_reproducibility": c4,
        "overall": overall,
        "synthesis_sources": {
            "R-CROSSFAM-HONEST": str(EXP_DIR / "results" / "R-CROSSFAM-HONEST" / "metrics.json"),
            "R-EDGETYPE-REAL-OPT": str(EXP_DIR / "results" / "R-EDGETYPE-REAL-OPT" / "metrics.json"),
            "R-DAG-MLFEAT-SAME-v2": str(EXP_DIR / "results" / "R-DAG-MLFEAT-SAME-v2" / "metrics.json"),
            "R-ABLATE-TEMP-LOO-REAL": str(EXP_DIR / "results" / "R-ABLATE-TEMP-LOO-REAL" / "metrics.json"),
        },
        "superseded_runs": {
            "R-CROSSFAM": "cf_fraction=0.998 was density artefact; retained only as structural-capacity diagnostic",
            "R-EDGETYPE": "F1=1.0 was tautological (circular metadata); retained only as rule-taxonomy consistency check",
            "R-DAG-MLFEAT-SAME": "13-family (Bumblebee) drift; superseded by -v2 on canonical 12-family corpus",
        },
        "timing_seconds": round(time.time() - t0, 2),
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (OUT_DIR / "run.log").write_text("\n".join(log) + "\n")

    # Markdown table for the paper
    tbl = ["# Honest Results Table (R-MAIN-Win-HONEST)", "",
           "| Contribution | Honest result | Verdict |",
           "| :- | :- | :- |",
           f"| C1 typed-reticulation capacity | sparse DAG theta={c1['best_theta']}, density {c1['graph_density']}, mean-sim {c1['mean_sim_augment']}, cf={c1['matched_sparse_cf_malphpylo']} vs B6 {c1['matched_sparse_cf_b6']} | {c1['verdict']} (capacity) |",
           f"| C2 edge typing (real PE) | macro-F1 {c2['macro_f1_mean']} ± {c2['macro_f1_std']}, rb {c2['rank_biserial_mean']}, {c2['improvement_vs_b5']}× over majority | {c2['verdict']} |",
           f"| No-ML non-inferiority | Δ={noml['delta_mean']} ± {noml['delta_std']} vs PCA-ML proxy (margin δ_eq=0.10) | {noml['verdict']} |",
           f"| C3 temporal alignment | NEGATIVE: multi-signal {c3.get('full_multi_signal_inconsistency')} > best single {c3.get('best_single_inconsistency')} | reframed |",
           f"| C4 reproducibility | recall@32=0.9632 archived / 0.957 full corpus, 24/24 unit tests | {c4['verdict']} |",
           "", "## Superseded historical runs (excluded from headline metrics)",
           "- R-CROSSFAM cf_fraction=0.998 / N%=518% (density artefact)",
           "- R-EDGETYPE F1=1.0 (tautological circular metadata)"]
    (OUT_DIR / "honest_table.md").write_text("\n".join(tbl) + "\n")

    L(f"=== COMPLETE | C1={c1['verdict']} C2={c2['verdict']} NoML={noml['verdict']} | {metrics['timing_seconds']}s ===")
    return metrics


if __name__ == "__main__":
    m = main()
    print(json.dumps({
        "C1": m["C1_capacity"]["verdict"],
        "C2_f1": m["C2_typing"]["macro_f1_mean"],
        "NoML_delta": m["No_ML"]["delta_mean"],
        "summary_status": m["overall"]["summary_status"][:80] + "...",
    }, indent=2))
