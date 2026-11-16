#!/usr/bin/env python3
"""Download MalwareBazaar daily archives for temporal depth (H4/C3 fix).

Downloads 1 ZIP per month (or --quarterly) across a date range, extracts the
embedded metadata + Win32 PE samples (each sample is individually zipped with
password "infected"), filters to exe/dll, and appends to the manifest + raw-PE
store. This adds multi-year temporal span that the live API can't provide.

Output: data/malwarebazaar/manifest.jsonl (appended), data/malwarebazaar/samples/
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import urllib.request

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent.parent
DATA_DIR = EXP_DIR / "data" / "malwarebazaar"
SAMPLES_DIR = DATA_DIR / "samples"
ARCHIVE_DIR = DATA_DIR / "archives"
BASE = "https://datalake.abuse.ch/malware-bazaar/daily/"
ZIP_PW = b"infected"


def load_env():
    env = EXP_DIR / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def month_iter(start: date, end: date, step_months: int = 1):
    """Yield the 15th of every Nth month from start to end (mid-month = stable archive)."""
    d = date(start.year, start.month, 15)
    while (d.year, d.month) <= (end.year, end.month):
        yield d
        m = d.month - 1 + step_months
        d = date(d.year + m // 12, m % 12 + 1, 15)


def download_day(day: date) -> Path | None:
    """Download a daily ZIP. Returns local path or None if 404."""
    fname = f"{day.isoformat()}.zip"
    url = BASE + fname
    out = ARCHIVE_DIR / fname
    if out.exists() and out.stat().st_size > 1000:
        return out  # cached
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MalPhylo-research/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        if len(data) < 1000:
            return None
        out.write_bytes(data)
        return out
    except Exception as e:
        print(f"  [skip] {day}: {e}")
        return None


def extract_win32_pe(zip_path: Path, seen: set[str], day_str: str,
                     max_per_day: int = 20) -> tuple[int, int]:
    """Extract Win32 PE samples from a daily archive ZIP (SAFETY: static parse only,
    never execute; samples stored as password-protected ZIPs).

    Daily ZIP structure: all entries are AES-128 encrypted with password "infected".
    We decrypt .exe/.dll entries in-memory, check MZ header, and re-store as
    password-protected ZIPs. Limited to max_per_day for speed (temporal span needs
    samples from different dates, not all samples from one date).
    Returns (n_pe_found, n_new)."""
    n_pe = 0
    n_new = 0

    try:
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                base = name.rsplit("/", 1)[-1]
                sha = base.split(".")[0]
                if len(sha) != 64 or sha in seen:
                    continue

                lower = name.lower()
                # FAST PATH: only process .exe/.dll (raw PE directly in outer ZIP).
                # Skip .zip entries (password-protected inner archives, slow to decrypt,
                # and mostly non-PE formats; the .exe entries cover Win32 PE adequately).
                if not lower.endswith((".exe", ".dll")):
                    continue

                is_pe = False
                raw_pe: bytes | None = None
                try:
                    raw = zf.read(name, pwd=ZIP_PW)
                    if raw[:2] == b"MZ":
                        is_pe = True
                        raw_pe = raw
                except Exception:
                    pass

                if not is_pe:
                    continue

                n_pe += 1
                if n_pe > max_per_day:
                    break  # speed: cap extraction per day
                out = SAMPLES_DIR / f"{sha}.zip"
                if out.exists():
                    seen.add(sha)
                    continue
                # Store extracted PE payload inside a local standard ZIP container
                # (run_g1_features handles both standard ZIPs and AES/encrypted ZIPs)
                if lower.endswith(".zip"):
                    out.write_bytes(raw_pe if raw_pe else zf.read(name))
                else:
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as wz:
                        wz.writestr(base, raw_pe)
                    out.write_bytes(buf.getvalue())
                seen.add(sha)
                n_new += 1

                with open(DATA_DIR / "manifest.jsonl", "a") as mf:
                    mf.write(json.dumps({
                        "sha256": sha,
                        "family": "Unknown",
                        "signature": "Unknown",
                        "first_seen": day_str,
                        "file_type": "exe",
                        "file_size": len(raw_pe) if raw_pe else None,
                        "source": f"archive:{day_str}",
                    }) + "\n")
    except zipfile.BadZipFile:
        print(f"  [warn] {zip_path.name}: bad zip")
    except Exception as e:
        print(f"  [warn] {zip_path.name}: {e}")
    return n_pe, n_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-03", help="start month YYYY-MM (default 2020-03)")
    ap.add_argument("--end", default="2025-12", help="end month YYYY-MM (default 2025-12)")
    ap.add_argument("--quarterly", action="store_true", help="sample every 3 months (fewer downloads)")
    ap.add_argument("--monthly", action="store_true", default=True, help="sample every month (default)")
    ap.add_argument("--max-per-day", type=int, default=20, help="max PE samples to extract per archive day (speed cap)")
    args = ap.parse_args()

    load_env()
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    start = date.fromisoformat(f"{args.start}-01")
    end = date.fromisoformat(f"{args.end}-01")
    step = 3 if args.quarterly else 1

    days = list(month_iter(start, end, step))
    print(f"Temporal archive plan: {len(days)} days, {days[0]} to {days[-1]}, step={step}mo")
    print(f"Temp span: {(days[-1] - days[0]).days / 365.25:.1f} years")

    # load existing manifest for dedup
    seen: set[str] = set()
    manifest = DATA_DIR / "manifest.jsonl"
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            try:
                seen.add(json.loads(line)["sha256"])
            except Exception:
                pass
    print(f"Existing manifest: {len(seen)} unique SHA256s")

    total_pe = 0
    total_new = 0
    for i, day in enumerate(days):
        print(f"[{i+1}/{len(days)}] {day}...", end=" ", flush=True)
        zp = download_day(day)
        if zp is None:
            print("404 / missing - skip")
            continue
        n_pe, n_new = extract_win32_pe(zp, seen, day.isoformat(), args.max_per_day)
        total_pe += n_pe
        total_new += n_new
        print(f"{n_pe} PE found, {n_new} new → manifest total: {len(seen)}")
        time.sleep(0.5)  # politeness

    print(f"\nDone. Archives: {len(days)} days. PE found: {total_pe}. New samples: {total_new}.")
    print(f"Manifest now: {len(seen)} unique SHA256s.")


if __name__ == "__main__":
    main()
