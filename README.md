# MalPhylo

**A Typed-Edge DAG Framework for Malware Phylogeny from Interpretable Binary Features**

> 📌 **Anonymous artifact** — prepared for double-blind review. No author, affiliation, or
> personal-identifying information is included in this repository. For the anonymized
> submission mirror, see the `anonymous.4open.science` link in the paper.

MalPhylo reconstructs malware phylogeny as a typed-edge directed acyclic graph (DAG)
$G=(V,E,\tau,w,\lambda)$ over interpretable static binary features. Each edge carries one
of four forensic relationship labels — **parent-child**, **code-borrow**, **toolkit-share**,
or **convergent** — assigned by a deterministic, training-free $\lambda$-decide classifier.

**Key property:** no gradient-based training, no learned embeddings. Auditability is a
first-class design objective.

---

## 📊 Headline Results

| Claim | Metric | Value | Baseline |
|---|---|---|---|
| **C1** Typed reticulation capacity | cross-family fraction $cf$ @ 10,370 edges | **0.861** | 0.220 (untyped B6) |
| **C2** Edge-type classification | macro-F1 (Tier-G, 5 seeds) | **0.647 ± 0.015** | 0.100 (majority) |
| **No-ML** Interpretable vs ML proxy | $\Delta$ cross-family rate | **−0.012** (within $\delta_{eq}=0.10$) | PCA proxy |
| **C3** Temporal alignment | triplet inconsistency (string markers) | **0.21** (best single signal) | 0.30 (best fusion) |
| **C4** Scalability | HNSW recall@32 / query time | **0.957 @ 0.06 ms** | ≥ 0.95 floor |

Full metrics for every run are under [`results/`](results/).

---

## 📁 Repository Structure

```
MalPhylo/
├── malphpylo/                  # Core package (training-free)
│   ├── config.py               # YAML config schema + validation
│   ├── features.py             # φ extractor: 6 feature groups + packing detection
│   ├── similarity.py           # Trust-renormalized weighted similarity
│   ├── temporal.py             # κ total order + multi-signal τ
│   ├── edge_typing.py          # λ-decide 5-rule deterministic classifier
│   ├── dag.py                  # DAG construction: MST + augment + V_I + Kahn
│   ├── maltree_nj.py           # Neighbor-joining / UPGMA baselines
│   ├── pipeline.py             # End-to-end runner
│   ├── logging_utils.py        # Structured JSON logs + evidence ledger
│   ├── __main__.py             # CLI entry point
│   └── __init__.py
├── scripts/                    # Experiment runners, organized by stage
│   ├── data_acquisition/       # Fetch MalwareBazaar / temporal archives
│   ├── features/               # G1 feature extraction
│   ├── baselines/              # B1/B2/B4/B6/B7 trees, MST, networks, kNN
│   ├── experiments/            # Official C1–C4 + No-ML runners (paper versions)
│   ├── ablations/              # k-sensitivity, temporal-LOO
│   ├── calibration/            # λ-decide threshold calibration (20% split)
│   └── gold_set/               # Tier-G provenance-anchored label construction
├── configs/                    # ann.yaml, thresholds.yaml, run configs
├── tests/                      # Formal-property unit tests
├── results/                    # metrics.json + run.log per run (no raw outputs)
├── data/                       # Dataset download guide (data not vendored)
├── requirements.txt
├── environment.yml
└── .env.example                # API-key template (real keys NEVER committed)
```

---

## 📥 Datasets

MalPhylo consumes **raw Win32 PE binaries**; pre-extracted feature datasets cannot run the
$\phi$ extractor. Three public sources are used:

| Dataset | Role | Source |
|---|---|---|
| **MalwareBazaar** (abuse.ch) | Primary raw Win32 PE corpus (2,074 main + 928 background = 3,002) | https://bazaar.abuse.ch/ — API: https://mb-api.abuse.ch/api/v1/ |
| **MalTree embeddings** | Tree baselines (B1/B2) + No-ML PCA proxy (B8) | https://zenodo.org/records/20261117 |
| **MASCOT** | Source-code genealogy → Tier-G provenance anchor + B3 reference | https://huggingface.co/datasets/Bojing94/MASCOT |

See [`data/README.md`](data/README.md) for exact download steps, licensing, and the
"infected" ZIP password convention.

---

## ⚙️ Installation

```bash
git clone <this-repo-url> MalPhylo
cd MalPhylo

# Option A: pip
pip install -r requirements.txt

# Option B: conda (recommended — pins Python 3.11)
conda env create -f environment.yml
conda activate malphpylo
```

**No GPU required.** All experiments run CPU-only. The reported HNSW query time
(0.06 ms/query) was measured on a workstation with an AMD Ryzen 7 5800X, 128 GB RAM, and
an NVIDIA RTX 3090 (the GPU was present but unused by MalPhylo's training-free pipeline).

---

## 🚀 Reproducing the Paper's Results

```bash
# 1. Acquire data (requires a free MalwareBazaar Auth-Key — see data/README.md)
cp .env.example .env   # then edit .env and paste your key
PYTHONPATH=. python scripts/data_acquisition/fetch_targeted_families.py

# 2. Extract features (G1)
PYTHONPATH=. python scripts/features/run_g1_features.py

# 3. Calibrate thresholds on a disjoint 20% split (600 / 3,002 samples)
PYTHONPATH=. python scripts/calibration/calibrate_thresholds.py

# 4. Run the four claim experiments
PYTHONPATH=. python scripts/experiments/run_g3_crossfam_honest.py    # C1
PYTHONPATH=. python scripts/experiments/run_g4_edgetype_real_opt.py  # C2
PYTHONPATH=. python scripts/experiments/run_c3_quality_weighted.py   # C3
PYTHONPATH=. python scripts/experiments/run_g5_ann.py                # C4

# 5. No-ML control (C1 structural metric vs PCA proxy)
PYTHONPATH=. python scripts/experiments/run_g3_dag_mlfeat_same.py

# 6. Verify formal properties (acyclicity, similarity symmetry/bounds/no-imputation)
PYTHONPATH=. pytest tests/ -v
```

Every reported number in the paper maps to a `results/<RUN_ID>/metrics.json` artifact;
`results/<RUN_ID>/run.log` records the exact command, seed, and corpus snapshot.

---

## 🔬 Methodology Mapping (code → paper)

| Module | Paper section |
|---|---|
| `malphpylo/features.py` (φ, 6 groups, packing) | §3.3 Interpretable Features + Trust Renormalization |
| `malphpylo/similarity.py` | §3.3, Proposition 1 (symmetry / bounds / no-imputation) |
| `malphpylo/temporal.py` (κ) | §3.2 Temporal Order and Acyclicity |
| `malphpylo/dag.py` (MST + augment + Kahn) | §3.4 DAG Construction |
| `malphpylo/edge_typing.py` (λ-decide) | §3.5 Edge Typing via λ-Decide |
| `tests/test_formal_properties.py` | Appendix, Formal Notes on Acyclicity |

---

## 🔒 Privacy & Security Notes

- **No API keys, credentials, or personal data** are committed. The MalwareBazaar Auth-Key
  is read from a local `.env` file (gitignored); copy `.env.example` to `.env` and supply
  your own key.
- **No raw malware binaries** are vendored. Samples are fetched on demand from MalwareBazaar
  as password-protected ZIPs (`"infected"` is the standard abuse.ch password, not a secret).
- Sample SHA-256 hashes appearing in `results/*.json` are public MalwareBazaar identifiers.

---

## 📜 License

Released under the **MIT License** — see [`LICENSE`](LICENSE). Academic use encouraged.

## 📖 Citation

For double-blind review, cite the anonymous version referenced in the submitted paper.
A non-anonymous `CITATION.bib` will be provided upon acceptance.

```bibtex
@misc{malphylo2026anonymous,
  title  = {MalPhylo: A Typed-Edge DAG Framework for Malware Phylogeny
            from Interpretable Binary Features},
  author = {Anonymous},
  year   = {2026},
  note   = {Anonymous artifact, submitted for double-blind review}
}
```
