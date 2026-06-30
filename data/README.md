# Data Acquisition Guide

MalPhylo operates on **raw binaries** (Win32 PE in this phase). Pre-extracted feature
datasets cannot run the φ extractor and are therefore not used as primary inputs.
This directory contains **only documentation** — no data is vendored. Run the fetch
scripts to populate it locally.

## Sources

### 1. MalwareBazaar — Primary Raw Win32 PE Corpus  ✅ required
- **URL:** https://bazaar.abuse.ch/
- **API:** https://mb-api.abuse.ch/api/v1/
- **Auth:** Free Auth-Key from https://auth.abuse.ch/register → put in `.env` as
  `MALWAREBAZAAR_AUTH_KEY` (see `.env.example`).
- **Format:** Samples arrive as password-protected ZIPs. The password `"infected"` is the
  standard abuse.ch convention (public, **not a secret**) and is hardcoded in the fetch
  scripts accordingly.
- **Corpus used in the paper:** 2,074 samples across 12 target families for C1/C2, plus
  928 non-target-family / `Unknown` samples as distractors for C3/C4 (3,002 total).

```bash
cp .env.example .env             # paste your Auth-Key
PYTHONPATH=. python scripts/data_acquisition/fetch_targeted_families.py
```

### 2. MalTree Embeddings — Tree Baselines + No-ML Proxy  ✅ required
- **URL:** https://zenodo.org/records/20261117
- **File:** `embeddings_fused.json` (~1.9 GB)
- **Role:** B1/B2 tree baselines (neighbor joining / UPGMA over embeddings) and B8 PCA
  machine-learning proxy for the no-ML non-inferiority test.
- **License:** CC-BY (Zenodo record).

### 3. MASCOT — Source-Code Genealogy (Tier-G Anchor + B3 Reference)  ✅ required
- **URL:** https://huggingface.co/datasets/Bojing94/MASCOT
- **Mirror:** https://github.com/BojingLi24/MASOT
- **Role:** Provenance evidence anchoring the Tier-G edge-type gold set (C2 truth) and the
  B3 source-code-genealogy reference baseline.
- **Format:** 6,032 Windows malware source specimens.

## Deferred to Phase 2 (not used in this paper)
- **EMBER** (https://github.com/elastic/ember) — ships features, not raw PE; not consumable
  by φ. Dropped after the corpus audit.
- **AndroZoo** (https://androzoo.unibe.ch/) — raw APK for the Android cross-platform arm;
  deferred until phase 2.

## Licensing & Terms
- MalwareBazaar: abuse.ch fair-use terms; Auth-Key required.
- MalTree (Zenodo): CC-BY.
- MASCOT: research-use notice (verify HuggingFace/GitHub terms before redistribution).

## Provenance
Every sample SHA-256 used in the paper is recorded in `results/R-G0/provenance_manifest.jsonl`
and `results/R-G1/features.jsonl`. Datasets are **not** redistributed with this artifact.
