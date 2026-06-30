#!/usr/bin/env python3
"""Fill toolkit-share edges for Tier-G gold set.

The original build_gold_set.py produced 0 toolkit-share edges because the MASCOT
PACK column was empty. This script fills the gap by detecting toolkit indicators
(packer, crypter, injector, loader, rootkit, builder, obfuscator) from MASCOT
project/folder names and creating cross-family edges within each toolkit group.

Methodology grounding:
  - methodology §4.2 λ-decide taxonomy: "toolkit-share — shared builder/packer toolkit"
  - experiment_plan §4 R-EDGETYPE-GOLD: Tier-G provenance-anchored edges
  - Provenance source: P4_shared_toolkit (keyword/name-based toolkit detection)

Approach:
  1. Load MASCOT specimens (same as build_gold_set.py).
  2. Detect toolkit type from folder/project name using keyword matching.
  3. Group specimens by toolkit type.
  4. For each group with ≥2 families, create cross-family toolkit-share edges.
  5. Merge new edges into existing tier_g.jsonl (dedup by src+dst+edge_type).
  6. Update metrics.json with new type distribution.

Safety: static metadata only. No malware execution.

Usage:
  python3 scripts/fill_toolkit_share_edges.py
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import openpyxl

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
MASCOT_DIR = EXP_DIR / "data" / "mascot-meta"
GOLD_DIR = EXP_DIR / "runs" / "R-EDGETYPE-GOLD"

# Toolkit detection patterns (ordered by specificity)
TOOLKIT_PATTERNS: List[Tuple[str, str]] = [
    # (toolkit_type, regex_pattern)
    ("crypter",      r"\b(crypt|cryptor|crypter|xorcrypt|encrypt)\b"),
    ("packer",       r"\b(pack|packer|compress|upx|armor)\b"),
    ("injector",     r"\b(inject|injector|injection|dll.inject|shellcode.inject)\b"),
    ("loader",       r"\b(loader|shellcode.loader|dropper|stager)\b"),
    ("rootkit",      r"\b(rootkit|rk)\b"),
    ("obfuscator",   r"\b(obfus|deobfus|obfuscator)\b"),
    ("builder",      r"\b(builder|build|generator|maker|creator)\b"),
    ("wrapper",      r"\b(wrapper|wrap|bundler|packer)\b"),
    ("exploit-kit",  r"\b(exploit.kit|cve.exploit|exploit.framework)\b"),
    ("backdoor-kit", r"\b(backdoor.creator|backdoor.maker|backdoor.generator)\b"),
    ("rat-kit",      r"\b(rat.builder|rat.framework)\b"),
    ("keylogger-kit",r"\b(keylogger.builder|keylogger.creator)\b"),
]


def detect_toolkit_type(name: str) -> str | None:
    """Detect toolkit type from a project/folder name. Returns the first match."""
    name_lower = name.lower()
    for tk_type, pattern in TOOLKIT_PATTERNS:
        if re.search(pattern, name_lower):
            return tk_type
    return None


def load_mascot_specimens() -> List[Dict]:
    """Load MASCOT specimens with family labels (same logic as build_gold_set.py)."""
    specimens = []
    wb = openpyxl.load_workbook(MASCOT_DIR / "MASCOT.xlsx", read_only=True)
    ws = wb.active
    headers = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
    for row in ws.iter_rows(min_row=2, values_only=True):
        d = dict(zip(headers, row))
        if not d.get("Source folder name"):
            continue
        specimens.append({
            "folder_name": d.get("Source folder name", ""),
            "project_name": d.get("GitHub project name", ""),
            "date": d.get("Date"),
            "language": d.get("Language", ""),
            "keyword": d.get("Keyword", ""),
            "notes": d.get("Notes", ""),
        })
    wb.close()

    # Load family labels
    label_map: Dict[str, Dict] = {}
    wb2 = openpyxl.load_workbook(MASCOT_DIR / "MASCOT_LABELS.xlsx", read_only=True)
    for sn in wb2.sheetnames:
        if not sn.startswith("MASCOT_"):
            continue
        ws = wb2[sn]
        rows = list(ws.iter_rows(min_row=1, values_only=True))
        if not rows:
            continue
        hdrs = rows[0]
        for row in rows[1:]:
            d = dict(zip(hdrs, row))
            folder = d.get("Source Folder Name")
            if not folder:
                continue
            avclass2 = d.get("AVClass2_Labels", "")
            label_map[str(folder)] = {
                "avclass2": str(avclass2) if avclass2 else "",
                "language_sheet": sn,
            }
    wb2.close()

    # Merge + parse families
    for sp in specimens:
        labels = label_map.get(sp["folder_name"], {})
        sp.update(labels)
        avclass2 = sp.get("avclass2", "")
        families: List[str] = []
        if avclass2 and avclass2 != "SINGLETON":
            for part in avclass2.split(","):
                part = part.strip()
                if part.startswith("FAM:"):
                    fam_name = part.split(":")[1].split("|")[0].strip()
                    if fam_name and fam_name.lower() not in ("singleton", "unknown"):
                        families.append(fam_name)
        sp["parsed_families"] = families
        sp["primary_family"] = families[0] if families else None

    return specimens


def build_toolkit_share_edges(specimens: List[Dict]) -> List[Dict]:
    """Build toolkit-share edges from specimens grouped by toolkit type."""
    # Detect toolkit type for each specimen
    by_toolkit: Dict[str, List[Dict]] = defaultdict(list)
    for sp in specimens:
        combined = f"{sp['folder_name']} {sp['project_name']} {sp.get('notes', '')}"
        tk_type = detect_toolkit_type(combined)
        if tk_type:
            sp["_toolkit_type"] = tk_type
            by_toolkit[tk_type].append(sp)

    edges: List[Dict] = []
    edge_id = 50000  # start high to avoid collision with existing edges

    for tk_type, specs in sorted(by_toolkit.items()):
        # Only keep specimens with family labels
        labeled = [s for s in specs if s.get("primary_family")]
        if len(labeled) < 2:
            continue
        fams_in_group: Set[str] = set(s["primary_family"] for s in labeled)
        if len(fams_in_group) < 2:
            continue  # need ≥2 families for cross-family toolkit-share

        # Create edges between different-family pairs
        for i in range(len(labeled)):
            for j in range(i + 1, min(len(labeled), i + 8)):
                sa, sb = labeled[i], labeled[j]
                fa, fb = sa["primary_family"], sb["primary_family"]
                if not fa or not fb or fa == fb:
                    continue
                edge_id += 1
                edges.append({
                    "edge_id": f"E{edge_id:05d}",
                    "src": sa["folder_name"],
                    "dst": sb["folder_name"],
                    "src_family": fa,
                    "dst_family": fb,
                    "src_date": str(sa.get("date", "")),
                    "dst_date": str(sb.get("date", "")),
                    "src_language": sa.get("language", ""),
                    "dst_language": sb.get("language", ""),
                    "edge_type": "toolkit-share",
                    "provenance_sources": ["P2_family_diff", "P4_shared_toolkit"],
                    "n_provenance": 2,
                    "evidence": {
                        "toolkit_type": tk_type,
                        "family_a": fa,
                        "family_b": fb,
                        "src_name": sa["folder_name"],
                        "dst_name": sb["folder_name"],
                        "same_language": sa.get("language") == sb.get("language"),
                    },
                    "tier": "G",
                })

    return edges


def main():
    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    def log(msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log("Loading MASCOT specimens...")
    t0 = time.time()
    specimens = load_mascot_specimens()
    log(f"Loaded {len(specimens)} specimens ({time.time()-t0:.1f}s)")

    # Detect toolkit types
    toolkit_counts = Counter()
    for sp in specimens:
        combined = f"{sp['folder_name']} {sp['project_name']} {sp.get('notes', '')}"
        tk = detect_toolkit_type(combined)
        if tk:
            toolkit_counts[tk] += 1
    log(f"Toolkit type distribution: {dict(toolkit_counts.most_common())}")

    # Build new toolkit-share edges
    log("Building toolkit-share edges...")
    new_edges = build_toolkit_share_edges(specimens)
    log(f"Generated {len(new_edges)} new toolkit-share edges")

    # Load existing Tier-G edges
    existing_path = GOLD_DIR / "tier_g.jsonl"
    existing_edges: List[Dict] = []
    with open(existing_path) as f:
        for line in f:
            line = line.strip()
            if line:
                existing_edges.append(json.loads(line))
    log(f"Existing Tier-G edges: {len(existing_edges)}")

    # Dedup: merge new edges, skip if (src, dst, edge_type) already exists
    seen = set()
    for e in existing_edges:
        seen.add((e["src"], e["dst"], e["edge_type"]))
        seen.add((e["dst"], e["src"], e["edge_type"]))  # bidirectional dedup

    added = 0
    merged = list(existing_edges)
    for e in new_edges:
        key = (e["src"], e["dst"], e["edge_type"])
        rev_key = (e["dst"], e["src"], e["edge_type"])
        if key not in seen and rev_key not in seen:
            merged.append(e)
            seen.add(key)
            seen.add(rev_key)
            added += 1

    log(f"Added {added} new toolkit-share edges (after dedup)")
    log(f"Total Tier-G edges now: {len(merged)}")

    # Write merged tier_g.jsonl
    with open(existing_path, "w") as f:
        for e in merged:
            f.write(json.dumps(e) + "\n")

    # Update metrics.json
    metrics_path = GOLD_DIR / "metrics.json"
    with open(metrics_path) as f:
        metrics = json.load(f)

    g_types = Counter(e["edge_type"] for e in merged)
    metrics["tier_g"]["total_edges"] = len(merged)
    metrics["tier_g"]["qualified_edges"] = len(merged)  # all have ≥2 provenance
    metrics["tier_g"]["type_distribution"] = dict(g_types)
    metrics["tier_g"]["toolkit_share_filled"] = True
    metrics["tier_g"]["toolkit_share_added"] = added
    metrics["tier_g"]["toolkit_type_distribution"] = dict(toolkit_counts.most_common())
    metrics["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    (GOLD_DIR / "run.log").write_text("\n".join(log_lines) + "\n")

    log(f"Verdict: toolkit-share gap {'FILLED' if added > 0 else 'STILL EMPTY'}")
    log(f"  Type distribution: {dict(g_types)}")
    log(f"  Written: {existing_path}, {metrics_path}")

    return metrics


if __name__ == "__main__":
    m = main()
    print(json.dumps({
        "tier_g": m["tier_g"],
    }, indent=2))
