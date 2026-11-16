"""
MalPhylo configuration schema and loader.

Per pipeline configuration schema: every run reads a YAML config (dataset path, split,
feature weights, seeds, output dir). No hardcoded values in scripts.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass(frozen=True)
class FeatureGroupConfig:
    """Per-group enable + which groups are 'index' (I) vs 'background' (B).

    Per Section 3.5: I = {imports, string-entropy, section-metadata};
    B = {opcodes, byte/entropy histograms, CFG topology}.
    """
    enabled: List[str] = field(default_factory=lambda: [
        "opcode_ngrams", "cfg_topology", "imports",
        "string_entropy", "section_metadata", "byte_histogram",
    ])
    index_groups: List[str] = field(default_factory=lambda: [
        "imports", "string_entropy", "section_metadata",
    ])
    background_groups: List[str] = field(default_factory=lambda: [
        "opcode_ngrams", "byte_histogram", "cfg_topology",
    ])


@dataclass(frozen=True)
class PackingConfig:
    """Packing detection / trust-renormalization thresholds (Section 3.3).

    All thresholds fixed a priori and reported in the evidence ledger.
    """
    section_entropy_threshold: float = 7.2   # bits; packed-section detector
    min_code_entropy: float = 6.5            # below this -> suspect packed
    known_packer_signatures: List[str] = field(default_factory=lambda: [
        "UPX", "Themida", "ASPack", "PECompact", "MPRESS",
    ])


@dataclass(frozen=True)
class TemporalConfig:
    """Multi-signal temporal alignment (Section 5.4) + κ total order (Section 3.2).

    Signals ranked by reliability; conflicts recorded in evidence ledger.
    """
    signals: List[str] = field(default_factory=lambda: [
        "pe_timestamp",        # (1) compilation time, spoofable
        "certificate_validity",# (2) Authenticode
        "vt_first_seen",       # (3) VirusTotal first-submission
        "code_reuse_order",    # (4) code-reuse temporal ordering
        "string_markers",      # (5) compiler version / build-path dates
    ])
    epsilon_seconds: int = 1   # Section 3.1: V_I tau = min-child-tau - epsilon
    sentinel_label: str = "TAU_BOT"  # τ⊥ sentinel class (Section 3.2 rule 1)


@dataclass(frozen=True)
class EdgeTypingConfig:
    """Edge-type decision thresholds (Section 3.5). Fixed a priori, ablated (G4)."""
    # rule 1: convergent
    theta_I_lo: float = 0.20
    theta_fp: float = 0.05    # entropy-histogram L2 distance for packer fingerprint
    # rule 2: toolkit-share
    theta_I_hi: float = 0.55
    theta_B_lo: float = 0.35
    k_tool: int = 3           # >= k_tool shared rare/builder constants
    # rule 3: code-borrow
    theta_loc: float = 0.50   # localized index-group similarity threshold
    L_bor: int = 2            # min run length of localized index groups
    # rule 4: parent-child
    theta_pc: float = 0.60    # overall similarity threshold
    theta_delta: float = 0.20 # |r_bar_I - r_bar_B| <= theta_delta (uniform sim)
    # candidate admission (Section 3.5 predicate)
    require_kappa_orientation: bool = True


@dataclass(frozen=True)
class MSTConfig:
    """MST skeleton (Section 3.4 step 2). Maximum spanning tree (sim is a proximity)."""
    top_k_neighbors: int = 50         # prune to top-k for scalability
    handle_disconnection: bool = True # component-merge when top-k disconnects


@dataclass(frozen=True)
class RunConfig:
    """Top-level config for a single run."""
    run_id: str
    run_type: str                      # "smoke" | "official"
    dataset_path: str
    platform: str                      # "win32_pe" | "android"
    sample_limit: Optional[int] = None # for smoke runs
    output_dir: str = "results"
    seed: int = 42
    feature_weights: Dict[str, float] = field(default_factory=lambda: {
        # alpha vector, sum = 1, set a priori (Section 3.3). Ablated in G4.
        "opcode_ngrams": 0.20,
        "cfg_topology": 0.15,
        "imports": 0.20,
        "string_entropy": 0.15,
        "section_metadata": 0.10,
        "byte_histogram": 0.20,
    })
    features: FeatureGroupConfig = field(default_factory=FeatureGroupConfig)
    packing: PackingConfig = field(default_factory=PackingConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    edge_typing: EdgeTypingConfig = field(default_factory=EdgeTypingConfig)
    mst: MSTConfig = field(default_factory=MSTConfig)

    def validate(self) -> None:
        """Validate config invariants before any run."""
        # alpha vector must sum to 1 (Section 3.3)
        total = sum(self.feature_weights.values())
        if not (0.999 <= total <= 1.001):
            raise ValueError(
                f"feature_weights alpha must sum to 1.0, got {total:.6f}"
            )
        # every enabled group has a weight
        for g in self.features.enabled:
            if g not in self.feature_weights:
                raise ValueError(f"missing feature_weight for enabled group '{g}'")
        # I and B partition enabled groups
        all_g = set(self.features.enabled)
        I = set(self.features.index_groups)
        B = set(self.features.background_groups)
        if (I | B) != all_g or (I & B):
            raise ValueError("index/background groups must partition enabled groups")


def load_config(path: str | Path) -> RunConfig:
    """Load a YAML run config and validate it."""
    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    f_raw = raw.get("features") or {}
    p_raw = raw.get("packing") or {}
    t_raw = raw.get("temporal") or {}
    e_raw = raw.get("edge_typing") or {}
    m_raw = raw.get("mst") or {}

    f_def = FeatureGroupConfig()
    p_def = PackingConfig()
    t_def = TemporalConfig()
    e_def = EdgeTypingConfig()
    m_def = MSTConfig()

    features = FeatureGroupConfig(
        enabled=f_raw.get("enabled", f_def.enabled),
        index_groups=f_raw.get("index_groups", f_def.index_groups),
        background_groups=f_raw.get("background_groups", f_def.background_groups),
    )
    packing = PackingConfig(
        section_entropy_threshold=p_raw.get("section_entropy_threshold", p_def.section_entropy_threshold),
        min_code_entropy=p_raw.get("min_code_entropy", p_def.min_code_entropy),
        known_packer_signatures=p_raw.get("known_packer_signatures", p_def.known_packer_signatures),
    )
    temporal = TemporalConfig(
        signals=t_raw.get("signals", t_def.signals),
        epsilon_seconds=t_raw.get("epsilon_seconds", t_def.epsilon_seconds),
        sentinel_label=t_raw.get("sentinel_label", t_def.sentinel_label),
    )
    edge_typing = EdgeTypingConfig(
        theta_I_lo=e_raw.get("theta_I_lo", e_def.theta_I_lo),
        theta_fp=e_raw.get("theta_fp", e_def.theta_fp),
        theta_I_hi=e_raw.get("theta_I_hi", e_def.theta_I_hi),
        theta_B_lo=e_raw.get("theta_B_lo", e_def.theta_B_lo),
        k_tool=e_raw.get("k_tool", e_def.k_tool),
        theta_loc=e_raw.get("theta_loc", e_def.theta_loc),
        L_bor=e_raw.get("L_bor", e_def.L_bor),
        theta_pc=e_raw.get("theta_pc", e_def.theta_pc),
        theta_delta=e_raw.get("theta_delta", e_def.theta_delta),
        require_kappa_orientation=e_raw.get("require_kappa_orientation", e_def.require_kappa_orientation),
    )
    mst = MSTConfig(
        top_k_neighbors=m_raw.get("top_k_neighbors", m_def.top_k_neighbors),
        handle_disconnection=m_raw.get("handle_disconnection", m_def.handle_disconnection),
    )
    cfg = RunConfig(
        run_id=raw["run_id"],
        run_type=raw["run_type"],
        dataset_path=raw["dataset_path"],
        platform=raw["platform"],
        sample_limit=raw.get("sample_limit"),
        output_dir=raw.get("output_dir", "results"),
        seed=raw.get("seed", 42),
        feature_weights=raw.get("feature_weights"),
        features=features,
        packing=packing,
        temporal=temporal,
        edge_typing=edge_typing,
        mst=mst,
    )
    cfg.validate()
    return cfg
