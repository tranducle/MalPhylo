"""Multi-signal temporal alignment (Section 5.4) + κ total order (Section 3.2).

Section 3.2 defines a strict total order kappa(v) = (tau(v), h(v)) under which
acyclicity is PROVEN BY CONSTRUCTION, not inspected post-hoc:

  rule 1: tau(v) = median(S(v)) if S(v) nonempty, else TAU_BOT sentinel.
          Real values precede TAU_BOT.
  rule 2: kappa(v) = (tau(v), h(v)); h injective => kappa injective => strict total order.
  rule 3: every directed edge u->v satisfies kappa(u) < kappa(v).
  rule 4: a directed cycle would imply kappa(v1)<...<kappa(v1) - contradiction.
  rule 5: edges between two TAU_BOT vertices forced to 'convergent'.
  rule 6: augmentation admits a candidate only if kappa(u) < kappa(v).
"""
from __future__ import annotations

import hashlib
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .config import TemporalConfig
from .features import SampleFeatures


# Sentinel class for temporally-unrooted vertices (Section 3.2 rule 1).
# Ordered AFTER every real value (Section 3.2 rule 2).
class _TauBot:
    """τ⊥ sentinel. Singleton; compares greater than any real float."""
    _instance: Optional["_TauBot"] = None
    def __new__(cls) -> "_TauBot":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    def __lt__(self, other: Any) -> bool:
        return False  # TAU_BOT is never less than anything
    def __gt__(self, other: Any) -> bool:
        return isinstance(other, (int, float))  # greater than all reals
    def __le__(self, other: Any) -> bool:
        return isinstance(other, _TauBot)
    def __ge__(self, other: Any) -> bool:
        return True
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, _TauBot)
    def __hash__(self) -> int:
        return hash("__TAU_BOT__")
    def __repr__(self) -> str:
        return "TAU_BOT"


TAU_BOT = _TauBot()


@dataclass(frozen=True, order=False)
class KappaKey:
    """κ(v) = (τ(v), h(v)) total-order key (Section 3.2 rule 2).

    Ordering: real τ precedes TAU_BOT; ties broken by lexicographic SHA-256.
    """
    tau: Any   # float | _TauBot
    h: str     # SHA-256 hex (256-bit), compared lexicographically

    def __lt__(self, other: "KappaKey") -> bool:
        # TAU_BOT ordering: reals < TAU_BOT (Section 3.2 rule 2)
        s_bot = isinstance(self.tau, _TauBot)
        o_bot = isinstance(other.tau, _TauBot)
        if s_bot and o_bot:
            return self.h < other.h
        if s_bot and not o_bot:
            return False  # TAU_BOT is after reals
        if o_bot and not s_bot:
            return True   # real is before TAU_BOT
        # both real
        if self.tau != other.tau:
            return self.tau < other.tau
        return self.h < other.h


@dataclass
class TemporalLabel:
    """Per-vertex temporal state (Section 5.4 + Section 3.2)."""
    vertex_id: str
    tau: Any                       # float | _TauBot
    signals_used: List[str]        # which Section 5.4 signals contributed
    conflicts: List[str]           # recorded conflicts (Section 5.4, not silently resolved)
    is_internal: bool = False      # V_I inferred ancestor (Section 3.1)


def compute_tau(signals: Dict[str, Optional[float]],
                mode: str = "hierarchical") -> Tuple[Any, List[str], List[str]]:
    """Compute temporal label tau(v); returns TAU_BOT if no signal is present.

    In default 'hierarchical' mode (Section 5.4), prioritizes reliable sparse
    'string_markers', falls back to high-availability 'vt_first_seen' or
    'first_seen', and otherwise falls back to the median of remaining signals.
    'median' mode aggregates all available signals via statistics.median.
    """
    available = {k: float(v) for k, v in signals.items() if v is not None}
    if not available:
        return TAU_BOT, [], []

    vals = sorted(available.values())
    all_keys = sorted(available.keys())
    if mode == "median":
        tau = statistics.median(vals)
        used = all_keys
    elif mode == "hierarchical":
        if "string_markers" in available:
            tau = available["string_markers"]
            used = ["string_markers"]
        elif "vt_first_seen" in available:
            tau = available["vt_first_seen"]
            used = ["vt_first_seen"]
        elif "first_seen" in available:
            tau = available["first_seen"]
            used = ["first_seen"]
        else:
            tau = statistics.median(vals)
            used = all_keys
    else:
        raise ValueError(f"unknown compute_tau mode: {mode}")

    # conflict detection: signals disagree by > 30 days => record conflict
    conflicts: List[str] = []
    if len(vals) >= 2:
        spread = max(vals) - min(vals)
        if spread > 30 * 86400:  # >30 days among epoch-second signals
            conflicts.append(
                f"signal_spread_{spread:.0f}s_among_{all_keys}"
            )

    return tau, used, conflicts


def build_kappa(vertex_id: str,
                features: Optional[SampleFeatures],
                tau: Any,
                child_hashes: Optional[List[str]] = None) -> KappaKey:
    """Construct κ(v) for a leaf or inferred-internal vertex (Section 3.1 + Section 3.2).

    Leaf (V_L):   h(v) = SHA-256(byte content)  -> features.sha256
    Internal (V_I): h(v) = SHA-256("INT" || sort(children hashes))  (Section 3.1)
    """
    if features is not None:
        return KappaKey(tau=tau, h=features.sha256)
    # inferred internal node
    if child_hashes is None:
        child_hashes = []
    payload = b"INT" + b"".join(sorted(h.encode() for h in child_hashes))
    h = hashlib.sha256(payload).hexdigest()
    return KappaKey(tau=tau, h=h)


def internal_tau(child_taus: List[Any], epsilon: int = 1) -> Any:
    """Section 3.1: τ(v) = min(child τ) - ε for V_I nodes.

    If any child is TAU_BOT, v inherits TAU_BOT (and its incident edges are
    forced 'convergent' per Section 3.2 rule 5).
    """
    if any(isinstance(t, _TauBot) for t in child_taus):
        return TAU_BOT
    real_taus = [t for t in child_taus if not isinstance(t, _TauBot)]
    if not real_taus:
        return TAU_BOT
    return min(real_taus) - epsilon


def kappa_orientation(u: KappaKey, v: KappaKey) -> bool:
    """Section 3.2 rule 3 + rule 6: returns True iff κ(u) < κ(v) (edge u->v admissible)."""
    return u < v
