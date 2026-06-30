#!/usr/bin/env python3
"""Targeted download: fetch raw PE for families with <100 samples on disk.

Resolves StatisticalAnalyst BLOCK 2: family coverage gap (3 → 12+ families).

Strategy: for each family in the manifest with ≥100 entries but <100 raw PE
ZIPs on disk, download up to `per_family` samples (default 120, to allow for
some parse failures). This ensures ≥12 families with ≥100 processable samples.

Uses the MalwareBazaar API (Auth-Key from .env). Rate-limited at 0.3s/request.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
DATA_DIR = EXP_DIR / "data" / "malwarebazaar"
SAMPLES_DIR = DATA_DIR / "samples"
MANIFEST = DATA_DIR / "manifest.jsonl"
API = "https://mb-api.abuse.ch/api/v1/"


def load_env() -> str:
    env_path = EXP_DIR / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    key = os.environ.get("MALWAREBAZAAR_AUTH_KEY")
    if not key:
        sys.exit("ERROR: MALWAREBAZAAR_AUTH_KEY missing in .env")
    return key


def main():
    per_family = 120  # download 120 per family to ensure ≥100 after parse failures
    target_families = 13  # need 13 more to reach 16 total (≥12 threshold)

    key = load_env()
    print(f"Auth-Key loaded (len={len(key)})")

    # Build family → list of sha256 from manifest
    manifest_by_fam: dict[str, list[str]] = defaultdict(list)
    with open(MANIFEST) as f:
        for line in f:
            d = json.loads(line)
            fam = d.get("family", "Unknown")
            sha = d.get("sha256", "")
            if sha:
                manifest_by_fam[fam].append(sha)

    # Count on-disk per family
    disk_shas = set()
    for fn in os.listdir(SAMPLES_DIR):
        if fn.endswith(".zip"):
            disk_shas.add(fn[:-4])

    disk_by_fam: dict[str, int] = {}
    for fam, shas in manifest_by_fam.items():
        disk_by_fam[fam] = sum(1 for s in shas if s in disk_shas)

    # Find families needing download: manifest ≥100, on-disk <100
    needs_download = []
    for fam, count in sorted(manifest_by_fam.items(), key=lambda x: -len(x[1])):
        manifest_count = len(manifest_by_fam[fam])
        on_disk = disk_by_fam.get(fam, 0)
        if manifest_count >= 100 and on_disk < 100:
            needed = min(per_family - on_disk, manifest_count - on_disk)
            needs_download.append((fam, manifest_count, on_disk, needed))

    print(f"\nFamilies needing download ({len(needs_download)}):")
    for fam, mc, od, nd in needs_download:
        print(f"  {fam:<20} manifest={mc:>5} on_disk={od:>5} download={nd:>5}")

    total_to_download = sum(nd for _, _, _, nd in needs_download)
    print(f"\nTotal samples to download: {total_to_download}")
    print(f"Estimated time: {total_to_download * 0.3 / 60:.1f} minutes")

    if total_to_download == 0:
        print("Nothing to download — all families already have ≥100 samples.")
        return

    # Download
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    failed = 0
    t0 = time.time()

    for fam, mc, od, nd in needs_download:
        # Get SHAs for this family that are not on disk
        fam_shas = [s for s in manifest_by_fam[fam] if s not in disk_shas]
        fam_shas = fam_shas[:nd]
        fam_downloaded = 0

        for sha in fam_shas:
            out = SAMPLES_DIR / f"{sha}.zip"
            if out.exists():
                disk_shas.add(sha)
                fam_downloaded += 1
                continue
            try:
                raw = urllib.parse.urlencode({"query": "get_file", "sha256_hash": sha}).encode()
                req = urllib.request.Request(API, data=raw, headers={"Auth-Key": key})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    payload = resp.read()
                if payload[:2] == b"PK":
                    out.write_bytes(payload)
                    disk_shas.add(sha)
                    fam_downloaded += 1
                    downloaded += 1
                else:
                    try:
                        err = json.loads(payload.decode())
                        if err.get("query_status") != "file_not_found":
                            print(f"  [warn] {sha[:12]}: {err.get('query_status')}")
                    except Exception:
                        pass
                    failed += 1
            except Exception as e:
                failed += 1
                if failed % 20 == 0:
                    print(f"  [warn] {sha[:12]}: {e}")

            time.sleep(0.3)  # fair-use rate limit

            if downloaded % 100 == 0 and downloaded > 0:
                elapsed = time.time() - t0
                rate = downloaded / elapsed
                eta = (total_to_download - downloaded) / rate / 60
                print(f"  [{downloaded}/{total_to_download}] {fam}: {fam_downloaded}/{nd} "
                      f"({rate:.1f}/s, ETA {eta:.1f}min)")

        print(f"  {fam}: downloaded {fam_downloaded}/{nd} (total on disk: {od + fam_downloaded})")

    elapsed = time.time() - t0
    print(f"\nDownload complete: {downloaded} downloaded, {failed} failed, {elapsed:.1f}s")

    # Verify final family coverage
    final_disk = Counter()
    for fn in os.listdir(SAMPLES_DIR):
        if fn.endswith(".zip"):
            sha = fn[:-4]
            for fam, shas in manifest_by_fam.items():
                if sha in shas:
                    final_disk[fam] += 1
                    break

    families_ge100 = {f: c for f, c in final_disk.items() if c >= 100 and f != "Unknown"}
    print(f"\nFinal family coverage: {len(families_ge100)} families with ≥100 samples")
    for f, c in sorted(families_ge100.items(), key=lambda x: -x[1]):
        print(f"  {f}: {c}")


if __name__ == "__main__":
    main()
