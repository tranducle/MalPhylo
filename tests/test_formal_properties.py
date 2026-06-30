"""Unit tests for MalPhylo formal properties (§3.3, §2.4).

These tests enforce the methodology invariants that must hold by construction:
  - §3.3 (i)  sim symmetry
  - §3.3 (ii) sim in [0,1]
  - §3.3 (iii) no imputation (untrusted groups removed from num AND den)
  - §3.3 (iv) sim = BOTTOM when G_tr empty
  - §2.4 acyclicity by construction (kappa total order)
  - §2.1.1 V_I hash key injectivity
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# ensure src on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from malphpylo.config import PackingConfig, RunConfig, FeatureGroupConfig, PackingConfig as PC
from malphpylo.features import (
    ByteHistogramFeatures, CFGTopologyFeatures, ImportFeatures,
    OpcodeNgramFeatures, SampleFeatures, SectionMetadataFeatures,
    StringEntropyFeatures, extract_features_from_bytes,
)
from malphpylo.similarity import BOTTOM, pairwise_similarity
from malphpylo.temporal import (
    KappaKey, TAU_BOT, build_kappa, compute_tau, internal_tau,
    kappa_orientation,
)


# ---------- helpers ----------

def _make_sample(sid, family, data, platform="win32_pe"):
    return extract_features_from_bytes(data, sid, family, platform, PackingConfig())


def _two_similar_samples():
    """Two samples with overlapping bytes -> nonzero similarity."""
    base = b"hello world " * 200 + b"BUILDER_MIRAI_1" + b"\x00" * 500
    return (
        _make_sample("s1", "mirai", base),
        _make_sample("s2", "mirai", base + b"\xff" * 100),
    )


# ---------- §3.3 (i) symmetry ----------

def test_similarity_symmetry():
    """§3.3 (i): sim(s_i, s_j) == sim(s_j, s_i)."""
    a, b = _two_similar_samples()
    alpha = {"opcode_ngrams": 0.2, "cfg_topology": 0.15, "imports": 0.2,
             "string_entropy": 0.15, "section_metadata": 0.1, "byte_histogram": 0.2}
    groups = ["opcode_ngrams", "cfg_topology", "imports", "string_entropy",
              "section_metadata", "byte_histogram"]
    r_ab = pairwise_similarity(a, b, alpha, groups)
    r_ba = pairwise_similarity(b, a, alpha, groups)
    assert r_ab.value is not None and r_ba.value is not None
    assert abs(r_ab.value - r_ba.value) < 1e-12, "similarity must be symmetric"


# ---------- §3.3 (ii) range ----------

def test_similarity_in_unit_interval():
    """§3.3 (ii): sim in [0, 1]."""
    a, b = _two_similar_samples()
    alpha = {"byte_histogram": 1.0}
    r = pairwise_similarity(a, b, alpha, ["byte_histogram"])
    assert r.value is not None
    assert 0.0 <= r.value <= 1.0


# ---------- §3.3 (iii) no imputation ----------

def test_no_imputation_packed():
    """§3.3 (iii): untrusted groups removed from numerator AND denominator.

    A packed sample has opcode_ngrams + cfg_topology untrusted (§3.2).
    The similarity must NOT deflate to zero — it uses only trusted groups.
    """
    base = b"shared import symbols " * 100 + b"\x00" * 500
    a = _make_sample("a", "fam", base)
    b = _make_sample("b", "fam", base + b"\x90" * 200)
    # force a to be packed: mark opcode + cfg untrusted
    a.opcode_ngrams.trusted = False
    a.cfg_topology.trusted = False
    alpha = {"opcode_ngrams": 0.5, "byte_histogram": 0.5}
    r = pairwise_similarity(a, b, alpha, ["opcode_ngrams", "byte_histogram"])
    # untrusted opcode_ngrams excluded -> value comes only from byte_histogram
    # NOT zeroed (would be imputation)
    assert r.value is not None
    assert "opcode_ngrams" not in r.per_group
    assert "byte_histogram" in r.per_group
    assert r.value > 0.0  # not deflated by the untrusted group


# ---------- §3.3 (iv) BOTTOM sentinel ----------

def test_bottom_sentinel_when_no_trusted_groups():
    """§3.3 (iv): sim = BOTTOM when G_tr(i,j) is empty."""
    a, b = _two_similar_samples()
    # mark ALL groups untrusted for both
    for s in (a, b):
        s.opcode_ngrams.trusted = False
        s.cfg_topology.trusted = False
        s.imports.trusted = False
        s.string_entropy.trusted = False
        s.section_metadata.trusted = False
        s.byte_histogram.trusted = False
    alpha = {"byte_histogram": 1.0}
    r = pairwise_similarity(a, b, alpha, ["byte_histogram"])
    assert r.value is BOTTOM, "sim must be BOTTOM when no trusted groups"


# ---------- §2.4 kappa total order ----------

def test_kappa_total_order_real_precedes_tau_bot():
    """§2.4 rule 2: every real value precedes TAU_BOT."""
    k_real = KappaKey(tau=1000.0, h="aaa")
    k_bot = KappaKey(tau=TAU_BOT, h="zzz")
    assert k_real < k_bot
    assert not (k_bot < k_real)


def test_kappa_orientation_admits_only_increasing():
    """§2.4 rule 3: edge u->v admissible iff kappa(u) < kappa(v)."""
    k1 = KappaKey(tau=100.0, h="a")
    k2 = KappaKey(tau=200.0, h="b")
    assert kappa_orientation(k1, k2) is True
    assert kappa_orientation(k2, k1) is False


def test_kappa_injective_on_distinct_hashes():
    """§2.4 rule 2: kappa injective when h injective => strict total order."""
    k1 = KappaKey(tau=100.0, h="a")
    k2 = KappaKey(tau=100.0, h="b")  # same tau, different h
    assert k1 < k2 or k2 < k1  # never equal => total order
    assert not (k1 < k2 and k2 < k1)  # antisymmetric


# ---------- §2.1.1 V_I hash key ----------

def test_internal_node_hash_uses_int_prefix():
    """§2.1.1: V_I h(v) = SHA-256('INT' || sort(children hashes)); injective."""
    child_hashes = ["abc", "def"]
    k = build_kappa("INT_0", None, TAU_BOT, child_hashes=child_hashes)
    # different child sets => different hash
    k2 = build_kappa("INT_1", None, TAU_BOT, child_hashes=["def", "abc"])
    assert k.h == k2.h  # sort makes order-independent
    k3 = build_kappa("INT_2", None, TAU_BOT, child_hashes=["abc", "xyz"])
    assert k.h != k3.h


def test_internal_tau_min_child_minus_epsilon():
    """§2.1.1: tau(V_I) = min(child tau) - epsilon."""
    assert internal_tau([100.0, 200.0, 300.0], epsilon=1) == 99.0
    # any TAU_BOT child => inherits TAU_BOT (§2.1.1 + §2.4 rule 5)
    assert internal_tau([100.0, TAU_BOT], epsilon=1) is TAU_BOT


# ---------- §4.3 temporal median ----------

def test_compute_tau_median_of_signals():
    """§4.3 + §2.4 rule 1: tau = median(available signals)."""
    tau, used, conflicts = compute_tau({"pe_timestamp": 100.0, "vt_first_seen": 300.0})
    assert tau == 200.0  # median
    assert set(used) == {"pe_timestamp", "vt_first_seen"}


def test_compute_tau_bot_when_no_signals():
    """§2.4 rule 1: S(v) empty => tau = TAU_BOT."""
    tau, used, conflicts = compute_tau({})
    assert tau is TAU_BOT
    assert used == []


# ---------- §3.2 packing detection ----------

def test_packing_detection_high_entropy():
    """§3.2: high-entropy section triggers packing detection."""
    from malphpylo.features import detect_packing, SectionMetadataFeatures, ByteHistogramFeatures
    sec = SectionMetadataFeatures(sections=[{"name": ".text", "entropy": 7.5}])
    bh = ByteHistogramFeatures(byte_hist=np.zeros(256), entropy_hist=np.zeros(16))
    packed, reason = detect_packing(sec, bh, PackingConfig())
    assert packed
    assert "high_entropy" in reason


def test_packing_detection_known_signature():
    """§3.2: known packer signature in section name triggers detection."""
    from malphpylo.features import detect_packing, SectionMetadataFeatures, ByteHistogramFeatures
    sec = SectionMetadataFeatures(sections=[{"name": "UPX0", "entropy": 5.0}])
    bh = ByteHistogramFeatures(byte_hist=np.zeros(256), entropy_hist=np.zeros(16))
    packed, reason = detect_packing(sec, bh, PackingConfig())
    assert packed
    assert "UPX" in reason


# ---------- config validation ----------

def test_config_alpha_must_sum_to_one():
    """§3.3: alpha vector must sum to 1."""
    from malphpylo.config import RunConfig, FeatureGroupConfig
    bad_alpha = {"opcode_ngrams": 0.5}  # sums to 0.5, not 1
    cfg = RunConfig(
        run_id="test", run_type="smoke", dataset_path="synthetic",
        platform="win32_pe", feature_weights=bad_alpha,
    )
    with pytest.raises(ValueError, match="sum to 1"):
        cfg.validate()
