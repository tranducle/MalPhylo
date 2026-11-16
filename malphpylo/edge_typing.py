"""Edge-type classifier λ-decide (Section 3.5) - rule-based, NO training.

Single deterministic pass over (r_vec, r_bar_I, r_bar_B, kappa) returning
(lambda_hat, p) in Lambda x [0,1]. All thresholds theta_* are fixed a priori
from the calibration split, reported in the evidence ledger, and ablated (G4).

Rules (priority order):
  1. convergent   - r_bar_I < theta_I_lo AND only packer fingerprint shared
  2. toolkit-share -  r_bar_I >= theta_I_hi AND r_bar_B < theta_B_lo AND
                    >= k_tool shared rare/builder constants
  3. code-borrow  - r_bar_I >= theta_I_hi AND r_bar_B < theta_B_lo AND
                    index similarity localized to a contiguous sub-region
  4. parent-child - r_bar >= theta_pc AND kappa(u)<kappa(v) AND
                    |r_bar_I - r_bar_B| <= theta_delta (broad uniform sim)
  5. Reject       - dropped from E (logged); never a cross-family candidate

Candidate-admission predicate (Section 3.5): a pair survives iff inter-family AND
satisfies the disjunction of rules 2-4 antecedents. Rule 1 (convergent) is
admissible only as a *relabeling* of an already-admitted candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import EdgeTypingConfig
from .features import SampleFeatures
from .similarity import SimilarityResult, group_similarity
from .temporal import KappaKey, kappa_orientation


# Edge-type label set Lambda (Section 3.1)
PARENT_CHILD = "parent-child"
CODE_BORROW = "code-borrow"
TOOLKIT_SHARE = "toolkit-share"
CONVERGENT = "convergent"
REJECT = "reject"

LAMBDA_SET = (PARENT_CHILD, CODE_BORROW, TOOLKIT_SHARE, CONVERGENT)


@dataclass
class EdgeTypeDecision:
    """Output of λ-decide (Section 3.5)."""
    label: str          # in LAMBDA_SET ∪ {REJECT}
    confidence: float   # p in [0,1]
    rule_fired: int     # which rule (1-5) decided
    r_bar_I: float      # trust-renormalized index-group mean
    r_bar_B: float      # trust-renormalized background-group mean
    r_bar: float        # overall sim(u,v)
    admitted: bool      # candidate-admission predicate result
    reason: str = ""


def _renormalized_means(per_group: Dict[str, float],
                        alpha: Dict[str, float],
                        index_groups: List[str],
                        background_groups: List[str]) -> Tuple[Optional[float], Optional[float]]:
    """Compute (r_bar_I, r_bar_B) trust-renormalized means (Section 3.5)."""
    def _mean(groups: List[str]) -> Optional[float]:
        num = sum(alpha.get(g, 0.0) * per_group.get(g, 0.0) for g in groups
                  if g in per_group and alpha.get(g, 0.0) > 0)
        den = sum(alpha.get(g, 0.0) for g in groups
                  if g in per_group and alpha.get(g, 0.0) > 0)
        return num / den if den > 0 else None
    return _mean(index_groups), _mean(background_groups)


def _localized_index_run(per_group: Dict[str, float],
                         index_groups: List[str],
                         theta_loc: float) -> int:
    """Max contiguous run of index groups with sim_g >= theta_loc (Section 3.5 rule 3)."""
    # order index groups canonically; find longest run >= theta_loc
    best = cur = 0
    for g in index_groups:
        if per_group.get(g, 0.0) >= theta_loc:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _shared_rare_constants(a: SampleFeatures, b: SampleFeatures) -> int:
    """Count shared rare/builder constants (Section 3.5 rule 2 toolkit-share signal)."""
    return len(a.string_entropy.rare_string_set & b.string_entropy.rare_string_set)


def _packer_fingerprint_only(a: SampleFeatures,
                             b: SampleFeatures,
                             theta_fp: float) -> bool:
    """Rule 1 antecedent: pair shares only a documented compiler/packer
    fingerprint (entropy-histogram L2 distance <= theta_fp with empty
    rare-string overlap)."""
    eh_a = a.byte_histogram.entropy_hist
    eh_b = b.byte_histogram.entropy_hist
    if eh_a.size == 0 or eh_b.size == 0:
        return False
    n = min(eh_a.size, eh_b.size)
    l2 = float(np.linalg.norm(eh_a[:n] - eh_b[:n]))
    rare_overlap = len(a.string_entropy.rare_string_set & b.string_entropy.rare_string_set)
    return l2 <= theta_fp and rare_overlap == 0


def lambda_decide(sim: SimilarityResult,
                  a: SampleFeatures,
                  b: SampleFeatures,
                  kappa_u: KappaKey,
                  kappa_v: KappaKey,
                  alpha: Dict[str, float],
                  index_groups: List[str],
                  background_groups: List[str],
                  cfg: EdgeTypingConfig) -> EdgeTypeDecision:
    """Section 3.5 λ-decide: single deterministic pass. Returns (lambda_hat, p).

    No training. All thresholds fixed a priori and reported.
    """
    if sim.value is None:
        return EdgeTypeDecision(
            label=REJECT, confidence=0.0, rule_fired=5,
            r_bar_I=0.0, r_bar_B=0.0, r_bar=0.0, admitted=False,
            reason="sim_is_BOTTOM_no_trusted_groups",
        )

    r_bar = sim.value
    r_bar_I, r_bar_B = _renormalized_means(
        sim.per_group, alpha, index_groups, background_groups
    )
    if r_bar_I is None or r_bar_B is None:
        return EdgeTypeDecision(
            label=REJECT,
            confidence=0.0,
            rule_fired=5,
            r_bar_I=0.0 if r_bar_I is None else r_bar_I,
            r_bar_B=0.0 if r_bar_B is None else r_bar_B,
            r_bar=r_bar,
            admitted=False,
            reason="undefined_index_or_background_mean",
        )

    from .temporal import _TauBot
    both_bot = isinstance(kappa_u.tau, _TauBot) and isinstance(kappa_v.tau, _TauBot)
    if both_bot:
        oriented = False
    else:
        oriented = kappa_orientation(kappa_u, kappa_v) or kappa_orientation(kappa_v, kappa_u)

    # === Rule 1: convergent (highest priority, excludes ancestry) ===
    if r_bar_I < cfg.theta_I_lo and _packer_fingerprint_only(a, b, cfg.theta_fp):
        return EdgeTypeDecision(
            label=CONVERGENT, confidence=1.0 - r_bar, rule_fired=1,
            r_bar_I=r_bar_I, r_bar_B=r_bar_B, r_bar=r_bar,
            admitted=False,  # convergent is a RELABELING, not a generator
            reason="low_index_sim_and_packer_fingerprint_only",
        )

    # === Rule 2: toolkit-share ===
    k_shared = _shared_rare_constants(a, b)
    if (r_bar_I >= cfg.theta_I_hi
            and r_bar_B < cfg.theta_B_lo
            and k_shared >= cfg.k_tool):
        return EdgeTypeDecision(
            label=TOOLKIT_SHARE, confidence=min(r_bar_I, 1.0 - r_bar_B),
            rule_fired=2, r_bar_I=r_bar_I, r_bar_B=r_bar_B, r_bar=r_bar,
            admitted=True,
            reason=f"high_index_low_background_{k_shared}_shared_constants",
        )

    # === Rule 3: code-borrow ===
    loc_run = _localized_index_run(sim.per_group, index_groups, cfg.theta_loc)
    if (r_bar_I >= cfg.theta_I_hi
            and r_bar_B < cfg.theta_B_lo
            and loc_run >= cfg.L_bor):
        return EdgeTypeDecision(
            label=CODE_BORROW, confidence=r_bar_I * (1.0 - r_bar_B),
            rule_fired=3, r_bar_I=r_bar_I, r_bar_B=r_bar_B, r_bar=r_bar,
            admitted=True,
            reason=f"localized_index_run_{loc_run}",
        )

    # === Rule 4: parent-child ===
    if (r_bar >= cfg.theta_pc
            and (not cfg.require_kappa_orientation or oriented)
            and abs(r_bar_I - r_bar_B) <= cfg.theta_delta):
        return EdgeTypeDecision(
            label=PARENT_CHILD, confidence=r_bar,
            rule_fired=4, r_bar_I=r_bar_I, r_bar_B=r_bar_B, r_bar=r_bar,
            admitted=True,
            reason="broad_uniform_similarity_oriented",
        )

    # === Rule 5: Reject ===
    return EdgeTypeDecision(
        label=REJECT, confidence=0.0, rule_fired=5,
        r_bar_I=r_bar_I, r_bar_B=r_bar_B, r_bar=r_bar,
        admitted=False,
        reason="no_rule_antecedent_satisfied",
    )


def is_cross_family_candidate(a: SampleFeatures,
                              b: SampleFeatures,
                              decision: EdgeTypeDecision) -> bool:
    """Section 3.5 candidate-admission predicate: inter-family AND rules 2-4 admitted.

    Rule 1 (convergent) is admissible only as a relabeling, never a generator.
    """
    if a.family is None or b.family is None:
        return False
    if a.family == b.family:
        return False  # intra-family
    return decision.admitted  # rules 2-4 set admitted=True
