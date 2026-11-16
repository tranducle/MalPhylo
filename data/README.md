# Data Directory Notes

This directory holds **documentation only**. No raw malware binaries, third-party feature dumps, or source corpora are vendored in the repository.

See [`DATA.md`](../DATA.md) at the repository root for complete upstream links, licensing, corpus breakdown (`3,002` Win32 PE binaries across `15` family labels = `14` named families + `Unknown`, `2,181` packed / `821` unpacked, containing the `2,074`-sample `12`-family targeted subset; MalTree embeddings for `B1`/`B2`/`B6`/`B7`; `6,031` MASCOT source specimens for the independent Tier-G provenance ledger), and `.env` setup.

Quick setup order for MalwareBazaar fetches (run `fetch_malwarebazaar.py` before `fetch_targeted_families.py` so `data/malwarebazaar/manifest.jsonl` exists):

```bash
cp .env.example .env
# Paste MALWAREBAZAAR_AUTH_KEY in .env
python scripts/data_acquisition/fetch_malwarebazaar.py
python scripts/data_acquisition/fetch_targeted_families.py
```

MalwareBazaar upstream sample archives use the public `"infected"` ZIP password by abuse.ch convention; `pyzipper` in `requirements.txt` handles AES-encrypted ZIPs during in-memory static feature extraction (`scripts/features/run_g1_features.py`) without writing raw executables to disk.
