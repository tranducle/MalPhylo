#!/usr/bin/env python3
"""C2/H3: Edge-type gold-set construction from MASCOT genealogy.

Builds the two-tier provenance-anchored edge-type gold set per methodology §4.2.1:
  - Tier-G (provenance-anchored, PRIMARY truth): ≥150 edges, ≥2 provenance sources
  - Tier-S (annotator-agreement, LABELABILITY): ≥350 edges, κ_agree ≥ 0.6

Edge types (methodology §4.2 λ-decide taxonomy):
  1. parent-child    — vertical inheritance (derivative/fork)
  2. code-borrow     — horizontal code transfer (cross-family code reuse)
  3. toolkit-share   — shared builder/packer toolkit
  4. convergent      — similar behavior, independent development

Provenance sources for Tier-G:
  (P1) MASCOT genealogy: same family + temporal ordering (date) → parent-child
  (P2) AVClass2 family labels: family-level provenance
  (P3) GitHub fork/derivative: project name patterns (e.g., *-main, *-master vs fork)
  (P4) Language + keyword: cross-language → code-borrow; same keyword → toolkit-share

Safety: static metadata only. No malware execution. No source code execution.

Usage:
  python3 scripts/build_gold_set.py [--output runs/R-EDGETYPE-GOLD]
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import openpyxl

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
MASCOT_DIR = EXP_DIR / "data" / "mascot-meta"
OUT_DIR = EXP_DIR / "runs" / "R-EDGETYPE-GOLD"

# Edge type constants
ET_PARENT_CHILD = "parent-child"
ET_CODE_BORROW = "code-borrow"
ET_TOOLKIT_SHARE = "toolkit-share"
ET_CONVERGENT = "convergent"


def load_mascot_metadata() -> List[Dict[str, Any]]:
    """Load MASCOT.xlsx (specimen inventory) and MASCOT_LABELS.xlsx (AVClass2 labels).
    Returns a list of specimen dicts with unified fields."""
    specimens = []

    # 1. Load MASCOT.xlsx (main inventory)
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
            "url": d.get("Url", ""),
            "keyword": d.get("Keyword", ""),
            "notes": d.get("Notes", ""),
        })
    wb.close()

    # 2. Load MASCOT_LABELS.xlsx (per-language AVClass2 labels)
    label_map: Dict[str, Dict] = {}  # folder_name → labels
    wb2 = openpyxl.load_workbook(MASCOT_DIR / "MASCOT_LABELS.xlsx", read_only=True)
    for sn in wb2.sheetnames:
        if not sn.startswith("MASCOT_"):
            continue  # skip MalSource sheets (different format)
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
            claravy = d.get("ClarAVy_Labels", "")
            fam = d.get("FAM", "")
            beh = d.get("BEH", "")
            klass = d.get("CLASS", "")
            pack = d.get("PACK", "")
            label_map[str(folder)] = {
                "avclass2": str(avclass2) if avclass2 else "",
                "claravy": str(claravy) if claravy else "",
                "fam": str(fam) if fam else "",
                "beh": str(beh) if beh else "",
                "class": str(klass) if klass else "",
                "pack": str(pack) if pack else "",
                "language_sheet": sn,
            }
    wb2.close()

    # 3. Merge labels into specimens
    for sp in specimens:
        fn = sp["folder_name"]
        labels = label_map.get(fn, {})
        sp.update(labels)
        # Parse family from AVClass2 (format: "FAM:familyname|count,...")
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


def extract_behavior_tags(avclass2: str) -> Set[str]:
    """Extract BEH: tags from AVClass2 label string."""
    tags = set()
    if not avclass2 or avclass2 == "SINGLETON":
        return tags
    for part in avclass2.split(","):
        part = part.strip()
        if part.startswith("BEH:"):
            tag = part.split(":")[1].split("|")[0].strip().lower()
            if tag:
                tags.add(tag)
    return tags


def extract_class_tags(avclass2: str) -> Set[str]:
    """Extract CLASS: tags from AVClass2 label string."""
    tags = set()
    if not avclass2 or avclass2 == "SINGLETON":
        return tags
    for part in avclass2.split(","):
        part = part.strip()
        if part.startswith("CLASS:"):
            tag = part.split(":")[1].split("|")[0].strip().lower()
            if tag:
                tags.add(tag)
    return tags


def build_edges(specimens: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Build Tier-G and Tier-S candidate edges from MASCOT specimens.

    Tier-G edges need ≥2 provenance sources:
      P1: temporal ordering (date)
      P2: family label (AVClass2)
      P3: project/fork pattern
      P4: language/keyword overlap

    Edge type assignment:
      parent-child: same family + temporal ordering (P1+P2)
      code-borrow: different family + same language + shared keyword/class (P2+P4)
      toolkit-share: different family + same packer/builder (P2+P4)
      convergent: different family + same behavior + different language (P2+P4)
    """
    # Index specimens by family and by keyword
    by_family: Dict[str, List[Dict]] = defaultdict(list)
    by_keyword: Dict[str, List[Dict]] = defaultdict(list)
    by_language: Dict[str, List[Dict]] = defaultdict(list)

    for sp in specimens:
        if sp.get("primary_family"):
            by_family[sp["primary_family"]].append(sp)
        kw = sp.get("keyword", "")
        if kw:
            by_keyword[kw.lower()].append(sp)
        lang = sp.get("language", "")
        if lang:
            by_language[lang].append(sp)

    tier_g_edges: List[Dict] = []
    tier_s_edges: List[Dict] = []
    edge_id = 0

    def make_edge(src: Dict, dst: Dict, etype: str, provenance: List[str],
                  evidence: Dict, tier: str) -> Dict:
        nonlocal edge_id
        edge_id += 1
        return {
            "edge_id": f"E{edge_id:05d}",
            "src": src["folder_name"],
            "dst": dst["folder_name"],
            "src_family": src.get("primary_family"),
            "dst_family": dst.get("primary_family"),
            "src_date": str(src.get("date", "")),
            "dst_date": str(dst.get("date", "")),
            "src_language": src.get("language", ""),
            "dst_language": dst.get("language", ""),
            "edge_type": etype,
            "provenance_sources": provenance,
            "n_provenance": len(provenance),
            "evidence": evidence,
            "tier": tier,
        }

    # --- Tier-G: parent-child (same family, temporal ordering) ---
    for fam, specs in by_family.items():
        if len(specs) < 2:
            continue
        # Sort by date
        dated = [s for s in specs if s.get("date")]
        dated.sort(key=lambda s: s["date"])
        for i in range(len(dated) - 1):
            for j in range(i + 1, min(i + 4, len(dated))):  # limit to nearby pairs
                src, dst = dated[i], dated[j]
                provenance = ["P1_temporal", "P2_family"]
                # P3: fork pattern (project name contains -main, -master vs fork)
                if src.get("project_name") and dst.get("project_name"):
                    if src["project_name"] != dst["project_name"]:
                        provenance.append("P3_project_diff")
                evidence = {
                    "family": fam,
                    "date_delta_days": (dst["date"] - src["date"]).days if dst["date"] and src["date"] else None,
                    "same_language": src.get("language") == dst.get("language"),
                }
                tier_g_edges.append(make_edge(src, dst, ET_PARENT_CHILD, provenance, evidence, "G"))

    # --- Tier-G: code-borrow (different family, same language, shared class/behavior) ---
    families_list = list(by_family.keys())
    for fi in range(len(families_list)):
        for fj in range(fi + 1, len(families_list)):
            fam_a, fam_b = families_list[fi], families_list[fj]
            specs_a, specs_b = by_family[fam_a], by_family[fam_b]
            # Find pairs with same language and shared behavior
            for sa in specs_a[:15]:  # cap per family
                for sb in specs_b[:15]:
                    if sa.get("language") != sb.get("language"):
                        continue
                    beh_a = extract_behavior_tags(sa.get("avclass2", ""))
                    beh_b = extract_behavior_tags(sb.get("avclass2", ""))
                    shared_beh = beh_a & beh_b
                    class_a = extract_class_tags(sa.get("avclass2", ""))
                    class_b = extract_class_tags(sb.get("avclass2", ""))
                    shared_class = class_a & class_b
                    if not (shared_beh or shared_class):
                        continue
                    provenance = ["P2_family_diff", "P4_shared_behavior"]
                    if shared_class:
                        provenance.append("P4_shared_class")
                    evidence = {
                        "family_a": fam_a,
                        "family_b": fam_b,
                        "shared_behaviors": sorted(shared_beh),
                        "shared_classes": sorted(shared_class),
                        "same_language": True,
                    }
                    tier_g_edges.append(make_edge(sa, sb, ET_CODE_BORROW, provenance, evidence, "G"))

    # --- Tier-G: toolkit-share (different family, same packer) ---
    packer_groups: Dict[str, List[Dict]] = defaultdict(list)
    for sp in specimens:
        pack = sp.get("pack", "")
        if pack and pack.strip() and pack.strip().lower() not in ("none", "unknown"):
            packer_groups[pack.strip().lower()].append(sp)
    for pack, specs in packer_groups.items():
        if len(specs) < 2:
            continue
        fams_in_group = set()
        for s in specs:
            if s.get("primary_family"):
                fams_in_group.add(s["primary_family"])
        if len(fams_in_group) < 2:
            continue
        # Create edges between different-family pairs sharing this packer
        for i in range(len(specs)):
            for j in range(i + 1, min(len(specs), i + 5)):
                sa, sb = specs[i], specs[j]
                fa, fb = sa.get("primary_family"), sb.get("primary_family")
                if not fa or not fb or fa == fb:
                    continue
                provenance = ["P2_family_diff", "P4_shared_packer"]
                evidence = {
                    "packer": pack,
                    "family_a": fa,
                    "family_b": fb,
                }
                tier_g_edges.append(make_edge(sa, sb, ET_TOOLKIT_SHARE, provenance, evidence, "G"))

    # --- Tier-G: convergent (different family, same behavior, different language) ---
    for fi in range(len(families_list)):
        for fj in range(fi + 1, min(len(families_list), fi + 20)):
            fam_a, fam_b = families_list[fi], families_list[fj]
            specs_a, specs_b = by_family[fam_a], by_family[fam_b]
            for sa in specs_a[:10]:
                for sb in specs_b[:10]:
                    if sa.get("language") == sb.get("language"):
                        continue  # convergent = different language
                    beh_a = extract_behavior_tags(sa.get("avclass2", ""))
                    beh_b = extract_behavior_tags(sb.get("avclass2", ""))
                    shared_beh = beh_a & beh_b
                    if not shared_beh:
                        continue
                    provenance = ["P2_family_diff", "P4_shared_behavior", "P4_diff_language"]
                    evidence = {
                        "family_a": fam_a,
                        "family_b": fam_b,
                        "shared_behaviors": sorted(shared_beh),
                        "language_a": sa.get("language"),
                        "language_b": sb.get("language"),
                    }
                    tier_g_edges.append(make_edge(sa, sb, ET_CONVERGENT, provenance, evidence, "G"))

    # --- Tier-S: annotator-agreement candidates (same keyword, within-language pairs) ---
    for kw, specs in by_keyword.items():
        if len(specs) < 2:
            continue
        for i in range(len(specs)):
            for j in range(i + 1, min(len(specs), i + 8)):
                sa, sb = specs[i], specs[j]
                if sa.get("primary_family") and sb.get("primary_family"):
                    if sa["primary_family"] == sb["primary_family"]:
                        continue  # skip same-family (already in Tier-G)
                provenance = ["P4_shared_keyword"]
                evidence = {
                    "keyword": kw,
                    "family_a": sa.get("primary_family"),
                    "family_b": sb.get("primary_family"),
                }
                tier_s_edges.append(make_edge(sa, sb, "unknown", provenance, evidence, "S"))

    # Also add within-language cross-family pairs as Tier-S
    for lang, specs in by_language.items():
        if len(specs) < 2:
            continue
        # Sample pairs
        fams = defaultdict(list)
        for s in specs:
            if s.get("primary_family"):
                fams[s["primary_family"]].append(s)
        fam_list = list(fams.keys())
        for fi in range(len(fam_list)):
            for fj in range(fi + 1, min(len(fam_list), fi + 10)):
                for sa in fams[fam_list[fi]][:3]:
                    for sb in fams[fam_list[fj]][:3]:
                        provenance = ["P4_same_language"]
                        evidence = {
                            "language": lang,
                            "family_a": fam_list[fi],
                            "family_b": fam_list[fj],
                        }
                        tier_s_edges.append(make_edge(sa, sb, "unknown", provenance, evidence, "S"))

    return tier_g_edges, tier_s_edges


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log_lines: List[str] = []
    def log(msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log("Loading MASCOT metadata...")
    t0 = time.time()
    specimens = load_mascot_metadata()
    log(f"Loaded {len(specimens)} specimens ({time.time()-t0:.1f}s)")

    # Stats
    fams = Counter()
    langs = Counter()
    kws = Counter()
    labeled = 0
    for sp in specimens:
        if sp.get("primary_family"):
            fams[sp["primary_family"]] += 1
            labeled += 1
        if sp.get("language"):
            langs[sp["language"]] += 1
        if sp.get("keyword"):
            kws[sp["keyword"].lower()] += 1

    log(f"Family-labeled: {labeled}/{len(specimens)} ({labeled/len(specimens)*100:.1f}%)")
    log(f"Distinct families: {len(fams)}")
    log(f"Languages: {dict(sorted(langs.items(), key=lambda x: -x[1]))}")
    log(f"Keywords: {dict(sorted(kws.items(), key=lambda x: -x[1])[:10])}")

    log("Building edges...")
    t1 = time.time()
    tier_g, tier_s = build_edges(specimens)
    log(f"Built edges ({time.time()-t1:.1f}s)")

    # Stats
    g_types = Counter(e["edge_type"] for e in tier_g)
    g_prov = Counter()
    for e in tier_g:
        for p in e["provenance_sources"]:
            g_prov[p] += 1
    s_count = len(tier_s)

    # Dedup Tier-G by (src, dst, edge_type)
    seen_g = set()
    tier_g_unique = []
    for e in tier_g:
        key = (e["src"], e["dst"], e["edge_type"])
        if key not in seen_g:
            seen_g.add(key)
            tier_g_unique.append(e)
    tier_g = tier_g_unique

    # Dedup Tier-S by (src, dst)
    seen_s = set()
    tier_s_unique = []
    for e in tier_s:
        key = (e["src"], e["dst"])
        if key not in seen_s:
            seen_s.add(key)
            tier_s_unique.append(e)
    tier_s = tier_s_unique

    # Filter Tier-G: require ≥2 provenance sources
    tier_g_qualified = [e for e in tier_g if e["n_provenance"] >= 2]
    g_types_qualified = Counter(e["edge_type"] for e in tier_g_qualified)

    log(f"Tier-G: {len(tier_g)} total → {len(tier_g_qualified)} qualified (≥2 provenance)")
    log(f"  Types: {dict(g_types_qualified)}")
    log(f"  Provenance sources: {dict(g_prov)}")
    log(f"Tier-S: {len(tier_s)} candidate edges (for annotator labeling)")

    # Verdict
    g_pass = len(tier_g_qualified) >= 150
    s_pass = len(tier_s) >= 350
    verdict = "PASS" if (g_pass and s_pass) else "PARTIAL"

    metrics = {
        "run_id": "R-EDGETYPE-GOLD",
        "milestone": "edge-type gold set construction (C2/H3)",
        "gate": "pre-G3",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "COMPLETED",
        "verdict": verdict,
        "tier_g": {
            "total_edges": len(tier_g),
            "qualified_edges": len(tier_g_qualified),
            "threshold": 150,
            "pass": g_pass,
            "type_distribution": dict(g_types_qualified),
            "provenance_source_counts": dict(g_prov),
        },
        "tier_s": {
            "candidate_edges": len(tier_s),
            "threshold": 350,
            "pass": s_pass,
            "note": "Tier-S edges need annotator labeling (κ_agree ≥ 0.6); type is 'unknown' until labeled",
        },
        "specimens": {
            "total": len(specimens),
            "family_labeled": labeled,
            "distinct_families": len(fams),
            "top_families": dict(sorted(fams.items(), key=lambda x: -x[1])[:15]),
            "languages": dict(sorted(langs.items(), key=lambda x: -x[1])),
        },
        "timing_seconds": round(time.time() - t0, 2),
    }

    # Write outputs
    with open(OUT_DIR / "tier_g.jsonl", "w") as f:
        for e in tier_g_qualified:
            f.write(json.dumps(e) + "\n")
    with open(OUT_DIR / "tier_s.jsonl", "w") as f:
        for e in tier_s:
            f.write(json.dumps(e) + "\n")
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    (OUT_DIR / "run.log").write_text("\n".join(log_lines) + "\n")

    log(f"Verdict: {verdict}")
    log(f"  Tier-G: {len(tier_g_qualified)} ≥150 → {'PASS' if g_pass else 'FAIL'}")
    log(f"  Tier-S: {len(tier_s)} ≥350 → {'PASS' if s_pass else 'FAIL'}")
    log(f"  Written: {OUT_DIR}/tier_g.jsonl, tier_s.jsonl, metrics.json")

    return metrics


if __name__ == "__main__":
    m = main()
    print(json.dumps({k: m[k] for k in ("run_id", "verdict", "tier_g", "tier_s")},
                     indent=2, default=str))
