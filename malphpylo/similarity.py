"""Trust-renormalized weighted similarity (Section 3.3) - formal properties enforced.

sim(s_i, s_j) = sum_{g in G_tr} alpha_g * sim_g / sum_{g in G_tr} alpha_g

Verified properties (Section 3.3, not assumed):
  (i)  symmetry: sim(s_i,s_j) = sim(s_j,s_i)
  (ii) sim in [0,1]
  (iii) NO imputation - untrusted groups removed from numerator AND denominator,
       never set to zero (packing never deflates similarity)
  (iv) if G_tr empty -> sim = BOTTOM (sentinel), excluded from MST + candidates

Temporal asymmetry is applied ONLY at edge orientation (Section 3.2 kappa), never
inside sim, keeping the similarity graph undirected as the maximum spanning
tree requires.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Set

import numpy as np

from .features import SampleFeatures


# Sentinel for sim = BOTTOM (Section 3.3 property iv): no trusted groups.
BOTTOM: Optional[float] = None


@dataclass(frozen=True)
class SimilarityResult:
    """A pairwise similarity with provenance for the evidence ledger (Section 3.6)."""
    value: Optional[float]      # None == BOTTOM (Section 3.3 iv)
    per_group: Dict[str, float]  # group -> sim_g (only for trusted groups)
    trusted_groups: Set[str]    # G_tr(i,j)
    used_alpha: Dict[str, float]


# ===========================================================================
# Per-group similarity functions (fixed per group type, Section 3.3)
# ===========================================================================

def _jaccard(a: frozenset, b: frozenset) -> float:
    """Jaccard for import/export sets (Section 3.3). Symmetric, in [0,1]."""
    if not a and not b:
        return 0.0  # both empty: no shared signal; not 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine for opcode n-gram distributions + byte/entropy histograms (Section 3.3)."""
    if a.size == 0 or b.size == 0:
        return 0.0
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _histogram_intersection(a: np.ndarray, b: np.ndarray) -> float:
    """Histogram intersection for string-entropy distributions (Section 3.3)."""
    if a.size == 0 or b.size == 0:
        return 0.0
    n = min(a.size, b.size)
    return float(np.minimum(a[:n], b[:n]).sum())


def _section_similarity(a_sections, b_sections) -> float:
    """Weighted categorical+scalar overlap for section metadata (Section 3.3)."""
    if not a_sections or not b_sections:
        return 0.0
    # match by closest entropy + name; scalar overlap on vsize/rsize
    best = 0.0
    for sa in a_sections:
        for sb in b_sections:
            name_match = 1.0 if sa.get("name") == sb.get("name") else 0.0
            ent_a, ent_b = sa.get("entropy", 0.0), sb.get("entropy", 0.0)
            ent_sim = 1.0 - min(abs(ent_a - ent_b) / 8.0, 1.0)
            vs_a, vs_b = max(sa.get("vsize", 1), 1), max(sb.get("vsize", 1), 1)
            size_sim = min(vs_a, vs_b) / max(vs_a, vs_b)
            score = 0.4 * name_match + 0.4 * ent_sim + 0.2 * size_sim
            best = max(best, score)
    return best


def _cfg_similarity(a_pct: np.ndarray, b_pct: np.ndarray) -> float:
    """CFG topology similarity via percentile-vector cosine (Section 3.3)."""
    if a_pct.size == 0 or b_pct.size == 0:
        return 0.0
    n = min(a_pct.size, b_pct.size)
    if np.all(a_pct[:n] == 0) or np.all(b_pct[:n] == 0):
        return 0.0
    return _cosine(a_pct[:n], b_pct[:n])


# ===========================================================================
# Group-level dispatcher
# ===========================================================================

def group_similarity(group: str, a: SampleFeatures, b: SampleFeatures) -> float:
    """Compute sim_g for a single feature group (Section 3.3). Returns float in [0,1]."""
    if group == "imports":
        return _jaccard(a.imports.symbol_set, b.imports.symbol_set)
    if group == "string_entropy":
        return _histogram_intersection(
            a.string_entropy.entropy_distribution,
            b.string_entropy.entropy_distribution,
        )
    if group == "section_metadata":
        return _section_similarity(
            a.section_metadata.sections, b.section_metadata.sections
        )
    if group == "opcode_ngrams":
        return _cosine(a.opcode_ngrams.distribution, b.opcode_ngrams.distribution)
    if group == "byte_histogram":
        # cosine on concatenation of byte_hist + entropy_hist
        a_vec = np.concatenate([a.byte_histogram.byte_hist, a.byte_histogram.entropy_hist])
        b_vec = np.concatenate([b.byte_histogram.byte_hist, b.byte_histogram.entropy_hist])
        return _cosine(a_vec, b_vec)
    if group == "cfg_topology":
        return _cfg_similarity(a.cfg_topology.percentiles, b.cfg_topology.percentiles)
    raise ValueError(f"unknown feature group: {group}")


# ===========================================================================
# Trust-renormalized weighted mean (Section 3.3)
# ===========================================================================

def pairwise_similarity(a: SampleFeatures,
                        b: SampleFeatures,
                        alpha: Dict[str, float],
                        enabled_groups: list) -> SimilarityResult:
    """Compute the trust-renormalized weighted-mean similarity (Section 3.3).

    Enforces all four formal properties by construction:
      (i)   symmetry - every sim_g is symmetric and G_tr is pair-symmetric
      (ii)  [0,1]    - weighted mean of [0,1] values is in [0,1]
      (iii) NO imputation - untrusted groups removed from numerator AND
            denominator (never zeroed)
      (iv)  BOTTOM sentinel when G_tr is empty
    """
    # G_tr(i,j): groups trusted for BOTH endpoints (Section 3.3 + Section 3.3)
    a_trusted = a.trusted_groups()
    b_trusted = b.trusted_groups()
    g_tr = (a_trusted & b_trusted) & set(enabled_groups)

    per_group: Dict[str, float] = {}
    used_alpha: Dict[str, float] = {}

    if not g_tr:
        # property (iv): no trusted groups -> BOTTOM, excluded from MST + candidates
        return SimilarityResult(
            value=BOTTOM, per_group=per_group,
            trusted_groups=set(), used_alpha=used_alpha,
        )

    num = 0.0
    den = 0.0
    for g in enabled_groups:
        if g not in g_tr:
            continue  # property (iii): removed from BOTH numerator and denominator
        sg = group_similarity(g, a, b)
        sg = max(0.0, min(1.0, sg))  # guard property (ii)
        ag = alpha.get(g, 0.0)
        if ag <= 0.0:
            continue
        num += ag * sg
        den += ag
        per_group[g] = sg
        used_alpha[g] = ag

    if den <= 0.0:
        return SimilarityResult(
            value=BOTTOM, per_group=per_group,
            trusted_groups=g_tr, used_alpha=used_alpha,
        )

    val = num / den  # property (ii): weighted mean of [0,1] is in [0,1]
    val = max(0.0, min(1.0, val))
    return SimilarityResult(
        value=val, per_group=per_group,
        trusted_groups=g_tr, used_alpha=used_alpha,
    )
