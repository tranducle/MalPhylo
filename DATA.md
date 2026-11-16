# Data Acquisition and Corpus Guide

This repository contains **no raw malware binaries and no redistributed third-party datasets**. Documentation and acquisition scripts are included so users can obtain source corpora directly under upstream terms.

## Primary Sources

### 1. MalwareBazaar (Primary Win32 PE Corpus)

- **Portal:** https://bazaar.abuse.ch/
- **API Endpoint:** https://mb-api.abuse.ch/api/v1/
- **Daily Archives:** https://datalake.abuse.ch/malware-bazaar/daily/
- **Authentication:** Free Auth-Key from https://auth.abuse.ch/register, stored locally in `.env` (gitignored) as `MALWAREBAZAAR_AUTH_KEY` (see `.env.example`).
- **Archive Format:** Upstream sample downloads use password-protected ZIPs with the public abuse.ch convention password `"infected"` (handled by acquisition scripts; not a secret). Local daily-archive extracts store PE payloads inside standard ZIP containers (`scripts/features/run_g1_features.py` handles both encrypted and standard ZIPs in memory).
- **Study Corpus (`results/R-G1/metrics.json`):** `3,002` Win32 PE binaries across `15` family labels (`14` named families plus `Unknown`; `2,181` packed samples at `72.65%` and `821` unpacked), comprising the `2,074`-sample main `12`-family targeted subset (`>= 100` samples each: `Formbook` 414, `AgentTesla` 339, `AsyncRAT` 246, `NjRAT` 120, `PrivateLoader` 120, `RedLineStealer` 120, `IcedID` 120, `CobaltStrike` 120, `RaccoonStealer` 119, `PlugX` 119, `Amadey` 119, `NanoCore` 118) plus `77` `Bumblebee`, `1` `RedLine`, and `850` `Unknown` temporal background samples. Family labels derive from MalwareBazaar signature queries (`get_siginfo`), with daily-archive pulls tagged `Unknown`.

Acquisition workflow (must initialize `manifest.jsonl` before targeted family top-ups):

```bash
cp .env.example .env
# Edit .env and set MALWAREBAZAAR_AUTH_KEY

# Step 1a: initial family manifest + sample ZIPs
python scripts/data_acquisition/fetch_malwarebazaar.py

# Step 1b: multi-year daily archive temporal backfill (optional)
python scripts/data_acquisition/fetch_temporal_archives.py

# Step 1c: backfill families with >=100 manifest records toward 120 samples on disk
python scripts/data_acquisition/fetch_targeted_families.py
```

### 2. MalTree Embeddings (Tree Baselines and Correlation Check)

- **Record:** https://zenodo.org/records/20261117 (`embeddings_fused.json`, license: CC-BY)
- **Role in Evaluation:** Used for neighbor-joining (`B1`, run via `python -m malphpylo.maltree_nj configs/runs/r_smoke_2.yaml`) and UPGMA/NeighborNet/kNN (`B2`, `B6`, `B7`, run via `python scripts/baselines/run_g2_baselines.py`, archived in `results/R-G2/`), plus the `R-FEAT-CORR` similarity correlation check (`7` overlapping SHA-256 samples). Because only `1` of `2,074` samples in the main MalwareBazaar subset matched MalTree by SHA-256 (`results/R-FEAT-CORR/metrics.json`), the training-free `B8` non-inferiority comparison uses a **same-corpus linear PCA control baseline** (`results/R-DAG-MLFEAT-SAME-v2/`).

### 3. MASCOT (Source-Code Provenance Ledger and B3 Reference)

- **Record:** https://huggingface.co/datasets/Bojing94/MASCOT
- **Role in Evaluation:** Anchors the independent source-code provenance ledger (`results/R-EDGETYPE-GOLD/`: `3,139` Tier-G provenance edges plus `43,150` Tier-S candidate pairs across `6,031` processed source specimens, using AVClass2 metadata tags present in MASCOT) and the `B3` source-code genealogy reference comparison (`results/R-MASCOT-CMP/`). This source-code Tier-G ledger is distinct from the balanced four-class real-PE structural-proxy edge set used for `C2`.

## Excluded Sources

- Pre-extracted feature datasets (such as EMBER feature vectors) are excluded because MalPhylo computes static features directly from raw Win32 PE binaries rather than pre-featurized vectors.

## Local Artifacts and Provenance

- Local downloads under `data/` and large intermediate outputs (`results/R-G1/features.jsonl`, similarity matrices, graph exports) are excluded by `.gitignore`.
- Archived `results/R-G1/metrics.json` records feature extraction coverage across `3,002` Win32 PE samples (`0` parse failures, `2,181` packed samples at `72.65%`).
- Sample SHA-256 identifiers appearing in `results/*/metrics.json` are public MalwareBazaar hashes.
