"""Formal-property and core-pipeline unit tests for MalPhylo.

Verifies the invariant properties of:
  - Trust-renormalized similarity (Proposition 1): symmetry, [0,1] bounds,
    no missing-group imputation, BOTTOM sentinel when G_tr is empty.
  - Temporal order kappa (Section 3.2): total order with TAU_BOT, injectivity,
    hierarchical temporal fallback, internal V_I hash key & tau computation.
  - Packing detection (Section 3.3): high entropy, known signatures, untrusted
    unextracted opcode/CFG groups, and zero-vector _cfg_similarity=0.0.
  - Edge typing lambda-decide (Section 3.5): undefined I/B rejection and
    bidirectional kappa orientation in Rule 4.
  - DAG construction (Section 3.4): V_I polytomy rewiring (no orphan nodes),
    both-TAU_BOT convergent forcing, Kahn acyclicity check, and config parsing.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

# ensure repo root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from malphpylo.config import EdgeTypingConfig, PackingConfig, RunConfig, load_config
from malphpylo.dag import construct_dag
from malphpylo.edge_typing import REJECT, lambda_decide
from malphpylo.features import (
    ByteHistogramFeatures,
    CFGTopologyFeatures,
    ImportFeatures,
    OpcodeNgramFeatures,
    SampleFeatures,
    SectionMetadataFeatures,
    StringEntropyFeatures,
    extract_features_from_bytes,
)
from malphpylo.similarity import BOTTOM, _cfg_similarity, pairwise_similarity
from malphpylo.temporal import (
    KappaKey,
    TAU_BOT,
    build_kappa,
    compute_tau,
    internal_tau,
    kappa_orientation,
)


# === helpers ===

def _make_sample(sid: str,
                 family: str = "famA",
                 packed: bool = False,
                 byte_seed: int = 1,
                 rare_strings: frozenset = frozenset(),
                 temporal_signals=None) -> SampleFeatures:
    rng = np.random.default_rng(byte_seed)
    data = bytes(rng.integers(32, 127, size=2048, dtype=np.uint8))
    sf = extract_features_from_bytes(data, sid, family, "win32_pe", PackingConfig())
    if packed:
        sf.packed = True
        sf.cfg_topology.trusted = False
        sf.opcode_ngrams.trusted = False
    if rare_strings:
        sf.string_entropy = StringEntropyFeatures(
            entropy_distribution=sf.string_entropy.entropy_distribution,
            rare_string_set=rare_strings,
            trusted=True,
        )
    if temporal_signals is not None:
        sf.temporal_signals = temporal_signals
    return sf


_DEFAULT_ALPHA = {
    "opcode_ngrams": 0.20, "cfg_topology": 0.15, "imports": 0.20,
    "string_entropy": 0.15, "section_metadata": 0.10, "byte_histogram": 0.20,
}
_ALL_GROUPS = list(_DEFAULT_ALPHA.keys())


# === Proposition 1 (i) symmetry ===

def test_similarity_symmetry():
    """Proposition 1 (i): sim(a, b) == sim(b, a)."""
    a = _make_sample("a", byte_seed=1)
    b = _make_sample("b", byte_seed=2)
    r_ab = pairwise_similarity(a, b, _DEFAULT_ALPHA, _ALL_GROUPS)
    r_ba = pairwise_similarity(b, a, _DEFAULT_ALPHA, _ALL_GROUPS)
    assert r_ab.value is not None
    assert abs(r_ab.value - r_ba.value) < 1e-12
    # symmetry even under asymmetric packing
    b_pack = _make_sample("bp", packed=True, byte_seed=2)
    r_1 = pairwise_similarity(a, b_pack, _DEFAULT_ALPHA, _ALL_GROUPS)
    r_2 = pairwise_similarity(b_pack, a, _DEFAULT_ALPHA, _ALL_GROUPS)
    assert abs(r_1.value - r_2.value) < 1e-12


# === Proposition 1 (ii) range ===

def test_similarity_in_unit_interval():
    """Proposition 1 (ii): sim in [0, 1] when G_tr is non-empty."""
    for s1, s2 in [(1, 1), (1, 2), (3, 99), (42, 43)]:
        a = _make_sample("a", byte_seed=s1)
        b = _make_sample("b", byte_seed=s2)
        r = pairwise_similarity(a, b, _DEFAULT_ALPHA, _ALL_GROUPS)
        assert r.value is not None
        assert 0.0 <= r.value <= 1.0


# === Proposition 1 (iii) no imputation ===

def test_no_imputation_packed():
    """Proposition 1 (iii): packed samples with identical trusted signals still
    score near 1.0 over G_tr, never deflated toward 0 by missing opcodes/CFG."""
    a = _make_sample("a", byte_seed=7, packed=False)
    b = _make_sample("b", byte_seed=7, packed=True)
    # mark both cfg_topology & opcode_ngrams untrusted for a too except opcode check
    r = pairwise_similarity(a, b, _DEFAULT_ALPHA, _ALL_GROUPS)
    assert "opcode_ngrams" not in r.trusted_groups
    assert "cfg_topology" not in r.trusted_groups
    assert "byte_histogram" in r.trusted_groups
    assert r.per_group["byte_histogram"] > 0.99
    for g in ("opcode_ngrams", "cfg_topology"):
        assert g not in r.used_alpha


def test_cfg_similarity_zero_vectors_return_zero():
    """All-zero CFG percentile vectors must yield 0.0 (never 1.0 via epsilon)."""
    z = np.zeros(5, dtype=np.float64)
    assert _cfg_similarity(z, z) == 0.0


def test_unextracted_opcode_and_cfg_marked_untrusted():
    """Empty opcode distributions and zero-function CFGs must be marked untrusted."""
    sf = _make_sample("empty_cfg", byte_seed=5, packed=False)
    assert sf.cfg_topology.n_functions == 0
    assert sf.cfg_topology.trusted is False
    empty_bytes_sf = extract_features_from_bytes(b"A", "tiny", "famA", "win32_pe", PackingConfig())
    assert empty_bytes_sf.opcode_ngrams.distribution.size == 0
    assert empty_bytes_sf.opcode_ngrams.trusted is False


# === Proposition 1 (iv) BOTTOM sentinel ===

def test_bottom_sentinel_when_no_trusted_groups():
    """Proposition 1 (iv): G_tr = empty => sim = BOTTOM."""
    a = _make_sample("a", byte_seed=1)
    b = _make_sample("b", byte_seed=2)
    for sf in (a, b):
        sf.opcode_ngrams.trusted = False
        sf.cfg_topology.trusted = False
        sf.imports.trusted = False
        sf.string_entropy.trusted = False
        sf.section_metadata.trusted = False
        sf.byte_histogram.trusted = False
    r = pairwise_similarity(a, b, _DEFAULT_ALPHA, _ALL_GROUPS)
    assert r.value is BOTTOM


# === Section 3.2 kappa total order ===

def test_kappa_total_order_real_precedes_tau_bot():
    """Section 3.2: every real temporal value precedes TAU_BOT."""
    k_real = KappaKey(tau=1000.0, h="aaa")
    k_bot = KappaKey(tau=TAU_BOT, h="zzz")
    assert k_real < k_bot
    assert not (k_bot < k_real)


def test_kappa_orientation_admits_only_increasing():
    """Section 3.2: edge u->v admissible iff kappa(u) < kappa(v)."""
    k1 = KappaKey(tau=100.0, h="a")
    k2 = KappaKey(tau=200.0, h="b")
    assert kappa_orientation(k1, k2) is True
    assert kappa_orientation(k2, k1) is False


def test_kappa_injective_on_distinct_hashes():
    """Section 3.2: kappa injective when h injective => strict total order."""
    k1 = KappaKey(tau=100.0, h="a")
    k2 = KappaKey(tau=100.0, h="b")
    assert k1 < k2 or k2 < k1
    assert not (k1 < k2 and k2 < k1)


# === Section 3.1 V_I hash key and tau ===

def test_internal_node_hash_uses_int_prefix():
    """Section 3.1: V_I h(v) = SHA-256('INT' || sort(children hashes))."""
    child_hashes = ["abc", "def"]
    k = build_kappa("INT_0", None, TAU_BOT, child_hashes=child_hashes)
    k2 = build_kappa("INT_1", None, TAU_BOT, child_hashes=["def", "abc"])
    assert k.h == k2.h
    k3 = build_kappa("INT_2", None, TAU_BOT, child_hashes=["abc", "xyz"])
    assert k.h != k3.h


def test_internal_tau_min_child_minus_epsilon():
    """Section 3.1: tau(V_I) = min(child tau) - epsilon; inherits TAU_BOT."""
    assert internal_tau([100.0, 200.0, 300.0], epsilon=1) == 99.0
    assert internal_tau([100.0, TAU_BOT], epsilon=1) is TAU_BOT


# === Section 5.4 temporal hierarchy & median ablation mode ===

def test_compute_tau_hierarchical_fallback():
    """Section 5.4: string_markers > first_seen > remaining median."""
    tau, used, conflicts = compute_tau(
        {"pe_timestamp": 100.0, "vt_first_seen": 300.0, "string_markers": 500.0}
    )
    assert tau == 500.0
    assert used == ["string_markers"]
    # fallback to vt_first_seen when string_markers absent
    tau2, used2, _ = compute_tau({"pe_timestamp": 100.0, "vt_first_seen": 300.0})
    assert tau2 == 300.0
    assert used2 == ["vt_first_seen"]
    # explicit median mode preserved for C3 comparison
    tau3, _, _ = compute_tau({"pe_timestamp": 100.0, "vt_first_seen": 300.0}, mode="median")
    assert tau3 == 200.0


def test_compute_tau_bot_when_no_signals():
    """Section 3.2: empty signal set => tau = TAU_BOT."""
    tau, used, _ = compute_tau({})
    assert tau is TAU_BOT
    assert used == []


# === Section 3.3 packing detection ===

def test_packing_detection_high_entropy():
    """Section 3.3: high-entropy section triggers packing detection."""
    from malphpylo.features import detect_packing
    sec = SectionMetadataFeatures(sections=[{"name": ".text", "entropy": 7.5}])
    bh = ByteHistogramFeatures(byte_hist=np.zeros(256), entropy_hist=np.zeros(16))
    packed, reason = detect_packing(sec, bh, PackingConfig())
    assert packed
    assert "high_entropy" in reason


def test_packing_detection_known_signature():
    """Section 3.3: known packer signature in section name triggers detection."""
    from malphpylo.features import detect_packing
    sec = SectionMetadataFeatures(sections=[{"name": "UPX0", "entropy": 5.0}])
    bh = ByteHistogramFeatures(byte_hist=np.zeros(256), entropy_hist=np.zeros(16))
    packed, reason = detect_packing(sec, bh, PackingConfig())
    assert packed
    assert "UPX" in reason


# === Section 3.5 edge typing invariants ===

def test_edge_typing_undefined_subset_rejects_and_bidirectional_kappa():
    """Section 3.5: empty I or B trusted subset rejects via Rule 5; Rule 4 works regardless of arg order."""
    a = _make_sample("a", family="famA", byte_seed=11, temporal_signals={"string_markers": 100.0})
    b = _make_sample("b", family="famB", byte_seed=11, temporal_signals={"string_markers": 200.0})
    # strip all B-group trust so r_bar_B is undefined
    for sf in (a, b):
        sf.opcode_ngrams.trusted = False
        sf.byte_histogram.trusted = False
        sf.cfg_topology.trusted = False
    sim_undef = pairwise_similarity(a, b, _DEFAULT_ALPHA, _ALL_GROUPS)
    dec_undef = lambda_decide(
        sim_undef, a, b, KappaKey(100.0, a.sha256), KappaKey(200.0, b.sha256),
        _DEFAULT_ALPHA, ["imports", "string_entropy", "section_metadata"],
        ["opcode_ngrams", "byte_histogram", "cfg_topology"], EdgeTypingConfig(),
    )
    assert dec_undef.label == REJECT
    assert dec_undef.reason == "undefined_index_or_background_mean"

    # Rule 4 bidirectional kappa check when caller swaps argument order (b, a)
    a2 = _make_sample("a2", family="famA", byte_seed=11)
    b2 = _make_sample("b2", family="famB", byte_seed=11)
    sim_ok = pairwise_similarity(a2, b2, _DEFAULT_ALPHA, _ALL_GROUPS)
    dec_swapped = lambda_decide(
        sim_ok, b2, a2, KappaKey(200.0, b2.sha256), KappaKey(100.0, a2.sha256),
        _DEFAULT_ALPHA, ["imports", "string_entropy", "section_metadata"],
        ["opcode_ngrams", "byte_histogram", "cfg_topology"],
        EdgeTypingConfig(theta_I_hi=0.99, theta_pc=0.5, theta_delta=1.0),
    )
    assert dec_swapped.label == "parent-child"


# === Section 3.4 DAG V_I connectivity & both-TAU_BOT handling ===

def test_vi_polytomy_nodes_are_connected_and_acyclic():
    """Section 3.4: inserted V_I nodes rewire polytomy edges with zero degree-0 orphans."""
    parent = _make_sample("p", family="famA", byte_seed=10, temporal_signals={"string_markers": 100.0})
    children = [
        _make_sample(f"c{k}", family="famA", byte_seed=10, temporal_signals={"string_markers": 200.0 + k * 10.0})
        for k in range(4)
    ]
    cfg = RunConfig(run_id="t_vi", run_type="smoke", dataset_path="synthetic", platform="win32_pe")
    res = construct_dag([parent] + children, cfg)
    assert res.acyclicity_verified is True
    assert res.n_internal_inserted >= 1
    vi_nodes = [n for n, v in res.vertices.items() if v.is_internal]
    assert len(vi_nodes) == res.n_internal_inserted
    for vn in vi_nodes:
        assert res.graph.in_degree(vn) >= 1
        assert res.graph.out_degree(vn) >= 3


def test_both_tau_bot_cross_family_forced_convergent():
    """Section 3.2/3.4: non-MST cross-family pair with both TAU_BOT triggers forced convergent rejection."""
    shared = frozenset(["KIT_MARKER_1", "KIT_MARKER_2", "KIT_MARKER_3", "KIT_MARKER_4"])
    # Create two same-family pairs linked by stronger intra-family MST edges plus a timed cross-bridge,
    # leaving (a, b) with empty temporal signals for cross-family augmentation evaluation.
    a0 = _make_sample("a0", family="famA", byte_seed=30, rare_strings=shared, temporal_signals={"string_markers": 100.0})
    a = _make_sample("a", family="famA", byte_seed=30, rare_strings=shared, temporal_signals={})
    b0 = _make_sample("b0", family="famB", byte_seed=30, rare_strings=shared, temporal_signals={"string_markers": 200.0})
    b = _make_sample("b", family="famB", byte_seed=30, rare_strings=shared, temporal_signals={})
    cfg = RunConfig(
        run_id="t_bot", run_type="smoke", dataset_path="synthetic", platform="win32_pe",
        edge_typing=EdgeTypingConfig(theta_I_hi=0.1, theta_B_lo=1.1, theta_pc=0.1, theta_delta=1.0),
    )
    res = construct_dag([a0, a, b0, b], cfg)
    assert res.acyclicity_verified is True
    reasons = [r["reason"] for r in res.rejected]
    assert "both_TAU_BOT_forced_convergent" in reasons


# === config validation & loading ===

def test_config_alpha_must_sum_to_one():
    """Section 3.3: alpha vector must sum to 1."""
    bad_alpha = {"opcode_ngrams": 0.5}
    cfg = RunConfig(
        run_id="test", run_type="smoke", dataset_path="synthetic",
        platform="win32_pe", feature_weights=bad_alpha,
    )
    with pytest.raises(ValueError, match="sum to 1"):
        cfg.validate()


def test_load_config_reads_theta_fp(tmp_path):
    """load_config must parse theta_fp rather than silently dropping it."""
    yml = tmp_path / "custom.yaml"
    yml.write_text(
        """run_id: custom
run_type: smoke
dataset_path: synthetic
platform: win32_pe
feature_weights:
  opcode_ngrams: 0.20
  cfg_topology: 0.15
  imports: 0.20
  string_entropy: 0.15
  section_metadata: 0.10
  byte_histogram: 0.20
features:
  enabled: [opcode_ngrams, cfg_topology, imports, string_entropy, section_metadata, byte_histogram]
  index_groups: [imports, string_entropy, section_metadata]
  background_groups: [opcode_ngrams, byte_histogram, cfg_topology]
edge_typing:
  theta_fp: 0.123
""",
        encoding="utf-8",
    )
    cfg = load_config(yml)
    assert abs(cfg.edge_typing.theta_fp - 0.123) < 1e-12


def test_compute_tau_conflict_lists_all_available_signals():
    """Conflict string must record all available signal keys even in hierarchical mode."""
    tau, used, conflicts = compute_tau(
        {"pe_timestamp": 1000000000.0, "string_markers": 1010000000.0}
    )
    assert tau == 1010000000.0
    assert used == ["string_markers"]
    assert len(conflicts) == 1
    assert "pe_timestamp" in conflicts[0] and "string_markers" in conflicts[0]


def test_detect_packing_normal_low_entropy_not_packed():
    """Normal unpacked low-entropy binary must not be misclassified as packed."""
    from malphpylo.features import detect_packing
    sec = SectionMetadataFeatures(sections=[{"name": ".text", "entropy": 4.2}])
    eh = np.zeros(16, dtype=np.float64)
    eh[8] = 1.0  # center ~4.25 < 6.5
    bh = ByteHistogramFeatures(byte_hist=np.full(256, 1.0 / 256.0), entropy_hist=eh)
    packed, reason = detect_packing(sec, bh, PackingConfig())
    assert packed is False
    assert reason == ""


def test_mst_skel_extract_feature_vector_reads_nested_imports():
    """run_g4_mst_skel.extract_feature_vector must read n_symbols from nested feature_groups."""
    import importlib.util
    mod_path = Path(__file__).resolve().parent.parent / "scripts" / "baselines" / "run_g4_mst_skel.py"
    spec = importlib.util.spec_from_file_location("run_g4_mst_skel", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sample_doc = {
        "feature_groups": {
            "byte_histogram": {"byte_hist": [0.0] * 256, "entropy_hist": [0.0] * 16},
            "string_entropy": {"entropy_distribution": [0.0] * 16, "rare_strings": ["s1", "s2"]},
            "section_metadata": {"sections": [{"name": ".text"}, {"name": ".data"}]},
            "imports": {"trusted": True, "n_symbols": 3, "symbol_set": ["a", "b", "c"]},
        }
    }
    vec = mod.extract_feature_vector(sample_doc)
    assert vec[-3:] .tolist() == [2.0, 2.0, 3.0]


def test_run_pipeline_reconciles_edge_counts_and_closes_handlers(tmp_path):
    """run_pipeline must reconcile n_mst + n_augment + n_internal_edges == n_edges_total and close file logs."""
    from dataclasses import replace
    from malphpylo.pipeline import run_pipeline
    cfg = load_config(Path(__file__).resolve().parent.parent / "configs" / "runs" / "r_smoke_1.yaml")
    cfg = replace(cfg, output_dir=str(tmp_path), sample_limit=60, run_id="t_reconcile")
    m = run_pipeline(cfg)
    assert m["status"] == "COMPLETE"
    assert m["n_mst_edges"] + m["n_augment_edges"] + m["n_internal_edges"] == m["n_edges_total"]
    assert m["n_internal_edges"] == m["n_internal_VI"]
