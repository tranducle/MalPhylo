# Reproducibility Guide

MalPhylo is a **CPU-only, training-free** static analysis and graph construction framework. No GPU, gradient-based optimization, or learned neural parameters are used.

## Environment Setup

Python 3.11+ (`environment.yml` pins `python=3.11`; also verified under Python 3.12) with the dependencies in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key dependencies: `networkx`, `scipy`, `numpy`, `scikit-learn` (evaluation metrics and linear PCA control baseline only), `pyyaml`, `hnswlib`, `pefile`, `pyzipper`, `openpyxl`, `androguard`, and `pytest`.

## Offline Verification (No External Data Required)

1. Formal-property and core-invariant unit tests (`24/24` tests in `tests/test_formal_properties.py`):

```bash
python -m pytest tests/ -v
```

2. End-to-end synthetic smoke pipeline (`configs/runs/r_smoke_1.yaml`):

```bash
python -m malphpylo run configs/runs/r_smoke_1.yaml
```

This writes synthetic run outputs to `results/R-SMOKE-1/` and verifies DAG acyclicity via Kahn's topological sort and `V_I` polytomy junction connectivity.

## Corpus-Dependent Pipeline Stages

After acquiring datasets per [`DATA.md`](DATA.md), individual stages execute from the repository root:

| Stage | Script / Command | Archived Run Directory |
| :- | :- | :- |
| Feature extraction (`G1`) | `python scripts/features/run_g1_features.py` | `results/R-G1/` |
| Pre-registered threshold calibration | `python scripts/calibration/calibrate_thresholds.py` | `results/R-CALIBRATE/` |
| MalTree `B1` NJ smoke check | `python -m malphpylo.maltree_nj configs/runs/r_smoke_2.yaml` | Standalone CLI (`B1`) |
| MalTree baseline sanity (`B2`, `B6`, `B7`) | `python scripts/baselines/run_g2_baselines.py` | `results/R-G2/` |
| MST skeleton baseline (`B4`) | `python scripts/baselines/run_g4_mst_skel.py` | `results/R-MST-SKEL/` |
| MASCOT Tier-G gold ledger construction | `python scripts/gold_set/build_gold_set.py` | `results/R-EDGETYPE-GOLD/` |
| Toolkit-share Tier-G enrichment | `python scripts/gold_set/fill_toolkit_share_edges.py` | `results/R-EDGETYPE-GOLD/` |
| MASCOT genealogy comparison (`B3`) | `python scripts/experiments/run_g4_mascot_cmp.py` | `results/R-MASCOT-CMP/` |
| Feature correlation vs MalTree (`H2`) | `python scripts/experiments/run_g3_feat_corr.py` | `results/R-FEAT-CORR/` |
| Same-corpus PCA non-inferiority (`B8`) | `python scripts/experiments/run_g3_dag_mlfeat_same.py` | `results/R-DAG-MLFEAT-SAME-v2/` |
| `k`-sensitivity and temporal LOO (`A8`, `A9`) | `python scripts/ablations/run_g4_ablate_k_a9.py` | `results/R-ABLATE-K-A9/` |
| Real-corpus temporal LOO ablation (`C3`) | `python scripts/ablations/run_g4_ablate_temp_real.py` | `results/R-ABLATE-TEMP-LOO-REAL/` |
| Quality-weighted temporal fusion (`C3`) | `python scripts/experiments/run_c3_quality_weighted.py` | `results/R-C3-QUALITY-WEIGHTED/` |
| HNSW retrieval scalability (`C4`) | `python scripts/experiments/run_g5_ann.py` | `results/R-ANN/` |
| Honest synthesis table generator | `python scripts/experiments/run_r_main_win_honest.py` | `results/R-MAIN-Win-HONEST/` |

Notes on configuration and scripts:
- `configs/runs/r_smoke_1.yaml` is a `RunConfig` for `python -m malphpylo run`; `configs/runs/r_smoke_2.yaml` and `configs/runs/r_g2.yaml` are standalone baseline configs for `malphpylo.maltree_nj` and `run_g2_baselines.py`.
- Global threshold configuration lives in `configs/thresholds.yaml` and HNSW hyperparameters (`M_HNSW=32`, `ef_Construction=200`, `ef_Search=512`, `k_star=32`) live in `configs/ann.yaml`. Archived `results/*/` directories store `metrics.json` and `run.log` outputs.
- `malphpylo/features.py` parses Win32 PE headers, sections, imports, strings, and byte/entropy histograms in memory via `pefile`; disassembly-dependent `opcode_ngrams` and `cfg_topology` groups remain unextracted on raw PEs and are marked `trusted=False` so missing modalities are excluded from similarity computation via trust renormalization.
- `scripts/experiments/run_g3_dag_mlfeat_same.py` (`B8`) compares top-`32` cosine cross-family neighbor rate over standardized `R-G1` static features against a 50-component linear PCA proxy on the canonical `2,074`-sample 12-family subset.
- `results/R-MST-SKEL/metrics.json` records a June 2026 snapshot captured before `run_g4_mst_skel.py` was updated to read the nested `feature_groups` schema and reference `results/R-CROSSFAM-HONEST/metrics.json`.

## Archived Snapshots vs Extended Manuscript Runs

To maintain strict transparency between archived repository outputs and extended manuscript evaluations:

- **Archived measurement provenance:** Existing JSON and log files under `results/` preserve the June 2026 run snapshots (`R-CROSSFAM-HONEST`: `cf = 0.8606` vs `0.2203` at `10,370` edges; `R-EDGETYPE-REAL-OPT`: threshold optimization `0.5516 +/- 0.0151` over `0.5445 +/- 0.003` baseline; `R-DAG-MLFEAT-SAME-v2`: five-run `Delta = -0.0122 +/- 0.0002` against same-corpus PCA; `R-ABLATE-TEMP-LOO-REAL` and `R-C3-QUALITY-WEIGHTED`: `string_markers` inconsistency `0.2114` / `0.2101` vs uniform median `0.3914`; `R-ANN`: `recall@32 = 0.9632` on `2,000` samples).
- **Unbundled helper dependencies in this snapshot:** `scripts/experiments/run_g3_crossfam_honest.py` imports `run_g3_crossfam`, and `scripts/experiments/run_g4_edgetype_real_opt.py` imports `run_g4_edgetype_real`. Those two upstream baseline helper files are not included in this repository snapshot (both scripts fail explicitly with an informative `ModuleNotFoundError`).
- **External extended manuscript evaluations:** Extended analyses reported in the manuscript, namely the held-out headline macro-F1 (`0.556 +/- 0.005`, `rb = 0.361`), post-hoc rule-refinement upper estimate (`0.647 +/- 0.015`, `rb = 0.413`), leave-index-out masking ablation (`A-N1`: no-mask `0.557`, mask imports `0.391`, mask strings `0.423`, mask sections `0.757`), the `41,003`-pair cross-family near-duplicate audit, the `503`-sample common-availability `C3` subset (`0.196 +/- 0.003` string markers), and the `3,002`-sample HNSW timing run (`recall@32 = 0.957`, `0.06 ms/query`, `0.13 s` build), were produced outside this standalone snapshot and are not shipped as separate runnable scripts here.
