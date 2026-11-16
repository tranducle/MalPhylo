# MalPhylo

**A Typed-Edge DAG Framework for Malware Phylogeny from Interpretable Binary Features**

> **Anonymous research artifact** prepared for double-blind peer review. Author names, affiliations, and personal identifiers are omitted from repository files. No raw malware binaries or third-party datasets are redistributed.

MalPhylo models malware evolution as a typed-edge directed acyclic graph $G=(V,E,\tau,w,\lambda)$ built from static Win32 PE binary features across interpretable feature groups (`imports`, `string_entropy`, `section_metadata`, `byte_histogram`, plus packing-masked disassembly modalities). Every cross-family candidate edge receives one of four forensic relationship labels (**parent-child**, **code-borrow**, **toolkit-share**, or **convergent**) from a deterministic, rule-based $\lambda$-decide classifier without gradient-based training or learned neural embeddings.

## Headline Evaluation Summary

Values below summarize the current manuscript evaluation alongside the archived baseline/ablation runs under `results/`. See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the exact correspondence between archived JSON artifacts and external extended runs.

| Claim | Metric | Manuscript Result | Baseline / Reference |
| :- | :- | :- | :- |
| **C1** Typed reticulation capacity | Cross-family fraction $cf$ at `10,370` edges (`2,074` PE samples) | **`0.861`** (mean augment sim `0.997`) | `0.220` (untyped `B6` at matched budget) |
| **C2** Edge-type classification | Held-out 4-class macro-F1 (`5` seeds, `300` PE structural-proxy edges/seed) | **`0.556 +/- 0.005`** (`rb = 0.361`) | `0.100` (majority class `B5`) |
| **C2** Refinement upper estimate | Post-hoc rule refinement (non-headline upper estimate) | `0.647 +/- 0.015` (`rb = 0.413`) | Threshold-opt snapshot: `0.5516` (`results/R-EDGETYPE-REAL-OPT/`) |
| **Training-free (`B8`)** | Top-`32` cross-family neighbor rate vs same-corpus PCA proxy (`2,074` PE samples) | `0.455` vs `0.468`, **`Delta = -0.012`** | Non-inferior within `delta_eq = 0.10` (`results/R-DAG-MLFEAT-SAME-v2/`) |
| **C3** Temporal alignment | Within-family triplet inconsistency (`3,002` PE corpus) | `0.21` (`string_markers`) vs `0.30` quality-weighted (honest negative) | `0.39` uniform median; `0.43` `first_seen` |
| **C4** Scalability & tests | HNSW `recall@32` / latency (`3,002` PE) + regression suite | **`0.957 @ 0.06 ms/query`** (`0.13 s` build); **`24` unit tests** (vs `14`-test paper baseline) | `>= 0.95` floor (`2,000`-sample snapshot `0.9632` in `results/R-ANN/`) |

**Key scope boundaries reported in the paper:**

- **C1 structural capacity:** Broad cross-family recovery superiority is not claimed; `C1` establishes sparse typed-reticulation capacity (`2,073` MST skeleton + `8,297` augment edges, density `0.48%`). A cross-family near-duplicate audit of `41,003` pairs with similarity `>= 0.99` found `0` exact SHA-256 duplicates, `435` (`1.1%`) near-duplicates, and `31` suspicious pairs.
- **C2 ground-truth separation:** Held-out PE classification is evaluated on balanced real-PE structural-proxy pairs (`75` per class per seed). The MASCOT source-code Tier-G ledger (`3,139` provenance edges + `43,150` Tier-S candidates across `6,031` processed specimens in `results/R-EDGETYPE-GOLD/`) is reported separately.
- **C3 negative finding:** Naive unweighted multi-signal temporal fusion (`0.39` inconsistency) underperforms the sparse `string_markers` signal (`0.21` overall across `1,036/3,002` samples; `0.196 +/- 0.003` on the `503`-sample common-availability subset). Hierarchical fallback (`kappa`) prefers `string_markers` and falls back to `first_seen` when string markers are absent.

## Repository Structure

```
MalPhylo/
├── malphpylo/                  # Core training-free package (features, similarity, kappa, DAG, lambda-decide)
├── scripts/
│   ├── data_acquisition/       # MalwareBazaar API and daily archive fetch scripts
│   ├── features/               # In-memory static Win32 PE feature extraction (G1)
│   ├── baselines/              # Neighbor joining, UPGMA, NeighborNet/median-joining, kNN, MST skeleton
│   ├── ablations/              # k-sensitivity (A8) and leave-one-out temporal ablations (A9, C3)
│   ├── calibration/            # Pre-registered threshold calibration (20% split)
│   ├── experiments/            # C1-C4, B3 MASCOT, B8 PCA control, HNSW retrieval, synthesis scripts
│   └── gold_set/               # MASCOT Tier-G provenance ledger construction
├── configs/                    # ann.yaml, thresholds.yaml, and configs/runs/ YAMLs
├── tests/                      # Formal-property and core-invariant regression unit tests
├── results/                    # Archived metrics.json and run.log snapshots per RUN_ID
├── data/                       # Documentation only (see DATA.md for corpus setup)
├── DATA.md                     # Upstream datasets, credentials setup, and corpus documentation
├── REPRODUCIBILITY.md          # Environment, offline checks, and snapshot artifact mapping
├── RELEASE_SHA256.txt          # SHA-256 checksum manifest of tracked release files
├── requirements.txt
├── environment.yml
└── .env.example                # MalwareBazaar key template (real .env is gitignored)
```

## Quick Start (Offline Checks)

```bash
pip install -r requirements.txt

# 1. Run formal-property and invariant unit tests (24 tests)
python -m pytest tests/ -v

# 2. Run the synthetic end-to-end smoke pipeline
python -m malphpylo run configs/runs/r_smoke_1.yaml
```

No GPU is required. All stages run on CPU only. Baseline MalTree configs (`configs/runs/r_smoke_2.yaml` and `configs/runs/r_g2.yaml`) are invoked via `python -m malphpylo.maltree_nj configs/runs/r_smoke_2.yaml` and `python scripts/baselines/run_g2_baselines.py` (defaults to `configs/runs/r_g2.yaml`). See [`DATA.md`](DATA.md) for dataset acquisition and [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for full experimental reproduction details.

## Methodology Mapping (Code to Manuscript)

| Module | Manuscript Section |
| :- | :- |
| `malphpylo/temporal.py` (`kappa` order and acyclicity) | Section 3.2 Temporal Order and Acyclicity |
| `malphpylo/features.py` (`phi`, static groups, packing detection) | Section 3.3 Interpretable Feature Vector |
| `malphpylo/similarity.py` (trust renormalization) | Section 3.3 and Appendix A.2 Trust Renormalization |
| `malphpylo/dag.py` (MST + cross-family augment + `V_I` + Kahn check) | Section 3.4 DAG Construction |
| `malphpylo/edge_typing.py` (`lambda`-decide rule classifier) | Section 3.5 Edge Typing via lambda-Decide |
| `tests/test_formal_properties.py` | Appendix A Formal Notes on Acyclicity and Similarity |

## Security and Privacy Notes

- **No secrets or credentials** are included. Set `MALWAREBAZAAR_AUTH_KEY` only inside a local `.env` file (gitignored).
- **No raw malware binaries** are committed or redistributed. Samples processed during `G1` are extracted in memory from ZIP archives and parsed statically via `pefile` (`extract_pe_features_from_bytes`) without writing raw executables to disk.
- All scripts operate purely via static analysis and deterministic graph/statistical routines.

## License and Citation

Released under the **MIT License** (see [`LICENSE`](LICENSE)).

For double-blind review, cite the anonymous artifact referenced in the submission:

```bibtex
@misc{malphylo2026anonymous,
  title  = {MalPhylo: A Typed-Edge DAG Framework for Malware Phylogeny
            from Interpretable Binary Features},
  author = {Anonymous},
  year   = {2026},
  note   = {Anonymous artifact, submitted for double-blind review}
}
```
