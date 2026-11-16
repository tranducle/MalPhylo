"""Feature extractor φ (Section 3.3) - interpretable binary features via static analysis.

Six feature groups (per Section 3.3):
  1. opcode_ngrams   - top-K opcode 2/3-gram distributions (normalized)
  2. cfg_topology    - per-function CFG stats aggregated to sample percentiles
  3. imports         - normalized import/export symbol sets (Jaccard)
  4. string_entropy  - printable-string Shannon entropies + rare-string set
  5. section_metadata -  per-section name/entropy/vsize/rsize/characteristics
  6. byte_histogram  - EMBER-style 256-bin byte + entropy histograms

Packing handling (Section 3.3): detect packers; mark affected groups *untrusted*
(down-weight, never discard). NO dynamic execution (static analysis only).
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .config import PackingConfig


# ===========================================================================
# Feature containers - one per group, each carrying a `trusted` flag (Section 3.3)
# ===========================================================================

@dataclass
class OpcodeNgramFeatures:
    distribution: np.ndarray   # normalized distribution over top-K 2/3-grams
    vocab: Tuple[str, ...]     # the canonical vocabulary (for alignment)
    trusted: bool = True

@dataclass
class CFGTopologyFeatures:
    # per-function stats -> sample-level percentiles
    percentiles: np.ndarray    # [p10,p25,p50,p75,p90] of {nodes,edges,diameter,density,clustering}
    n_functions: int = 0
    trusted: bool = True

@dataclass
class ImportFeatures:
    symbol_set: frozenset      # normalized import/export names
    trusted: bool = True

@dataclass
class StringEntropyFeatures:
    entropy_distribution: np.ndarray  # histogram of printable-string entropies
    rare_string_set: frozenset        # rare/builder strings (shared toolkit marker)
    trusted: bool = True

@dataclass
class SectionMetadataFeatures:
    # list of per-section records: (name, entropy, vsize, rsize, characteristics)
    sections: List[Dict[str, Any]] = field(default_factory=list)
    trusted: bool = True

@dataclass
class ByteHistogramFeatures:
    byte_hist: np.ndarray     # 256-bin byte histogram
    entropy_hist: np.ndarray  # entropy histogram (EMBER-style)
    trusted: bool = True


@dataclass
class SampleFeatures:
    """Full feature vector φ(s) for one sample (Section 3.3)."""
    sample_id: str
    family: Optional[str]
    sha256: str                      # h(v) for leaf vertices (Section 3.2)
    platform: str
    opcode_ngrams: OpcodeNgramFeatures
    cfg_topology: CFGTopologyFeatures
    imports: ImportFeatures
    string_entropy: StringEntropyFeatures
    section_metadata: SectionMetadataFeatures
    byte_histogram: ByteHistogramFeatures
    # raw temporal signals (Section 5.4) extracted here for the temporal module
    temporal_signals: Dict[str, Optional[float]] = field(default_factory=dict)
    packed: bool = False             # packing detection result (Section 3.3)

    def trusted_groups(self) -> Set[str]:
        """G_tr membership test (Section 3.3): which groups are trusted for this sample."""
        groups: Set[str] = set()
        if self.opcode_ngrams.trusted: groups.add("opcode_ngrams")
        if self.cfg_topology.trusted: groups.add("cfg_topology")
        if self.imports.trusted: groups.add("imports")
        if self.string_entropy.trusted: groups.add("string_entropy")
        if self.section_metadata.trusted: groups.add("section_metadata")
        if self.byte_histogram.trusted: groups.add("byte_histogram")
        return groups


# ===========================================================================
# Helpers
# ===========================================================================

_SHANNON_LOG2 = math.log2


def _shannon_entropy(byte_counts: Sequence[int]) -> float:
    total = sum(byte_counts)
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in byte_counts:
        if c > 0:
            p = c / total
            ent -= p * _SHANNON_LOG2(p)
    return ent


def _normalize_symbol(name: str) -> str:
    """Normalize an import/export symbol name (case-fold, strip decoration)."""
    n = name.strip()
    # strip common decoration: leading __, @N suffixes, dll prefix
    n = re.sub(r'@\d+$', '', n)
    n = re.sub(r'^_+', '', n)
    return n.lower()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ===========================================================================
# Byte + entropy histograms (EMBER-style) - robust, always trusted
# ===========================================================================

def _extract_byte_histogram(data: bytes) -> ByteHistogramFeatures:
    counts = np.zeros(256, dtype=np.float64)
    for b in data:
        counts[b] += 1.0
    total = counts.sum()
    if total > 0:
        byte_hist = counts / total
    else:
        byte_hist = counts
    # EMBER-style sliding-window entropy histogram (1024-byte windows -> 16 bins)
    # simplified to global entropy for the smoke run; full EMBER binning at G1 scale.
    window_size = 1024
    ent_bins = np.zeros(16, dtype=np.float64)
    if len(data) >= window_size:
        n_windows = 0
        for i in range(0, len(data) - window_size + 1, window_size):
            window = data[i:i + window_size]
            wc = np.zeros(256, dtype=np.float64)
            for b in window:
                wc[b] += 1.0
            ent = _shannon_entropy(wc.astype(int))
            bidx = min(int(ent / 8.0 * 16), 15)
            ent_bins[bidx] += 1.0
            n_windows += 1
        if n_windows > 0:
            ent_bins /= n_windows
    return ByteHistogramFeatures(byte_hist=byte_hist, entropy_hist=ent_bins)


# ===========================================================================
# Packing detection (Section 3.3)
# ===========================================================================

def detect_packing(section_features: SectionMetadataFeatures,
                   byte_features: ByteHistogramFeatures,
                   cfg: PackingConfig) -> Tuple[bool, str]:
    """Detect packing via section entropy + known signatures (Section 3.3).

    Returns (is_packed, reason). Packed sections trigger trust-renormalization:
    affected feature groups are marked *untrusted* (down-weighted, not discarded).
    """
    # high-entropy section heuristic
    for sec in section_features.sections:
        if sec.get("entropy", 0.0) >= cfg.section_entropy_threshold:
            return True, f"high_entropy_section:{sec.get('name','?')}"
    # known packer signature in section names
    name_blob = " ".join(s.get("name", "").upper() for s in section_features.sections)
    for sig in cfg.known_packer_signatures:
        if sig.upper() in name_blob:
            return True, f"signature:{sig}"
    # High global entropy across sliding windows indicates packed/encrypted payload
    if byte_features.entropy_hist.size > 0 and np.any(byte_features.entropy_hist > 0):
        centers = np.arange(16) * 0.5 + 0.25
        glob = float(np.average(centers, weights=byte_features.entropy_hist))
        if glob >= cfg.section_entropy_threshold:
            return True, f"high_global_entropy:{glob:.2f}"
    return False, ""


def apply_packing_trust(features: SampleFeatures, reason: str) -> None:
    """Section 3.3: mark packing-affected groups untrusted (down-weight, never discard).

    CFG topology and opcode n-grams are most affected by packing (obfuscated
    control flow + encrypted code). Byte histogram and section metadata remain
    trusted (they describe the packed artifact itself, which is still signal).
    Imports/strings may survive packing and remain trusted individually.
    """
    features.packed = True
    features.cfg_topology.trusted = False
    features.opcode_ngrams.trusted = False


# ===========================================================================
# Synthetic / file-backed extraction
# ===========================================================================

def extract_features_from_bytes(data: bytes,
                                sample_id: str,
                                family: Optional[str],
                                platform: str,
                                packing_cfg: PackingConfig) -> SampleFeatures:
    """Extract φ(s) from raw bytes (platform-agnostic fallback path).

    Used when a PE/ELF parser is unavailable or for smoke-test synthetic data.
    Full PE/APK parsing is in extract_pe_features / extract_apk_features.
    """
    sha = _sha256_bytes(data)

    # byte + entropy histograms (always available)
    byte_features = _extract_byte_histogram(data)

    # printable strings + their entropies
    printable = re.findall(rb'[\x20-\x7e]{4,}', data)
    str_ents = []
    rare_strings = set()
    for s in printable:
        sb = s.decode('ascii', errors='ignore')
        sc = Counter(sb)
        ent = _shannon_entropy(list(sc.values()))
        str_ents.append(ent)
        if ent > 3.5 and len(sb) >= 6:   # rare/high-entropy -> builder-string marker
            rare_strings.add(sb)

    if str_ents:
        ent_arr = np.array(str_ents, dtype=np.float64)
        ent_hist, _ = np.histogram(ent_arr, bins=16, range=(0.0, 8.0), density=False)
        ent_hist = ent_hist / max(ent_hist.sum(), 1.0)
    else:
        ent_hist = np.zeros(16, dtype=np.float64)

    # opcode n-grams: derive a pseudo-opcode stream from byte bigrams (fallback)
    # Real PE extraction uses disassembly; this is a deterministic fallback.
    bigrams = Counter()
    for i in range(0, len(data) - 1, 2):
        bigrams[(data[i], data[i + 1])] += 1
    top = bigrams.most_common(64)
    # most_common returns [((a, b), count), ...] - key is the bigram tuple
    vocab = tuple(f"bg_{ka}_{kb}" for (ka, kb), _ in top) if top else tuple()
    if top:
        dist = np.array([c for (_, c) in top], dtype=np.float64)
        dist /= dist.sum()
    else:
        dist = np.zeros(0, dtype=np.float64)

    # section metadata: fallback - treat whole file as one ".raw" section
    section_byte_counts = [Counter(data[i:i + 65536]) for i in range(0, len(data), 65536)]
    sections = []
    for i, sc in enumerate(section_byte_counts[:8]):
        sec_ent = _shannon_entropy(list(sc.values()))
        sections.append({
            "name": f".s{i}",
            "entropy": sec_ent,
            "vsize": len(data),
            "rsize": len(data),
            "characteristics": 0,
        })

    section_features = SectionMetadataFeatures(sections=sections)
    opcode_features = OpcodeNgramFeatures(distribution=dist, vocab=vocab, trusted=bool(dist.size > 0))
    cfg_features = CFGTopologyFeatures(percentiles=np.zeros(5), n_functions=0, trusted=False)
    import_features = ImportFeatures(symbol_set=frozenset())
    string_features = StringEntropyFeatures(
        entropy_distribution=ent_hist, rare_string_set=frozenset(rare_strings)
    )

    sample = SampleFeatures(
        sample_id=sample_id, family=family, sha256=sha, platform=platform,
        opcode_ngrams=opcode_features, cfg_topology=cfg_features,
        imports=import_features, string_entropy=string_features,
        section_metadata=section_features, byte_histogram=byte_features,
    )

    # packing detection + trust renormalization (Section 3.3)
    packed, reason = detect_packing(section_features, byte_features, packing_cfg)
    if packed:
        apply_packing_trust(sample, reason)

    return sample


def extract_pe_features(path: str | Path,
                        sample_id: str,
                        family: Optional[str],
                        packing_cfg: PackingConfig) -> SampleFeatures:
    """Extract static phi(s) from a Win32 PE file on disk via pefile."""
    return extract_pe_features_from_bytes(
        Path(path).read_bytes(), sample_id, family, packing_cfg
    )


def extract_pe_features_from_bytes(data: bytes,
                                   sample_id: str,
                                   family: Optional[str],
                                   packing_cfg: PackingConfig) -> SampleFeatures:
    """Extract static phi(s) in memory from raw Win32 PE bytes via pefile.

    Extracts PE timestamp, imports, sections, byte/entropy histograms, and
    printable strings without writing raw executables to disk. Unextracted
    disassembly groups (opcode_ngrams, cfg_topology) remain marked untrusted.
    """
    import pefile  # local import: optional dependency
    sha = _sha256_bytes(data)
    byte_features = _extract_byte_histogram(data)

    pe_ts: Optional[float] = None
    imports_set: Set[str] = set()
    sections: List[Dict[str, Any]] = []

    try:
        pe = pefile.PE(data=data, fast_load=True)
        # PE timestamp (Section 5.4 signal 1) - spoofable, cross-checked later
        if hasattr(pe, "FILE_HEADER") and pe.FILE_HEADER.TimeDateStamp:
            pe_ts = float(pe.FILE_HEADER.TimeDateStamp)
        # sections
        pe.parse_data_directories()
        for s in pe.sections:
            name = s.Name.rstrip(b'\x00').decode('ascii', errors='ignore')
            sc = Counter(s.get_data())
            sections.append({
                "name": name,
                "entropy": _shannon_entropy(list(sc.values())),
                "vsize": s.Misc_VirtualSize,
                "rsize": s.SizeOfRawData,
                "characteristics": int(s.Characteristics),
            })
        # imports
        if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                dll = entry.dll.decode('ascii', errors='ignore').lower()
                for imp in entry.imports:
                    nm = imp.name.decode('ascii', errors='ignore') if imp.name else f"ord_{imp.ordinal}"
                    imports_set.add(_normalize_symbol(f"{dll}!{nm}"))
        pe.close()
    except Exception:
        # parse failure -> fall back to byte-path features (logged upstream)
        return extract_features_from_bytes(data, sample_id, family, "win32_pe", packing_cfg)

    section_features = SectionMetadataFeatures(sections=sections)
    import_features = ImportFeatures(symbol_set=frozenset(imports_set))

    # printable strings
    printable = re.findall(rb'[\x20-\x7e]{4,}', data)
    str_ents = []
    rare_strings = set()
    for s in printable:
        sb = s.decode('ascii', errors='ignore')
        sc = Counter(sb)
        ent = _shannon_entropy(list(sc.values()))
        str_ents.append(ent)
        if ent > 3.5 and len(sb) >= 6:
            rare_strings.add(sb)
    if str_ents:
        ent_arr = np.array(str_ents, dtype=np.float64)
        ent_hist, _ = np.histogram(ent_arr, bins=16, range=(0.0, 8.0), density=False)
        ent_hist = ent_hist / max(ent_hist.sum(), 1.0)
    else:
        ent_hist = np.zeros(16, dtype=np.float64)
    string_features = StringEntropyFeatures(
        entropy_distribution=ent_hist, rare_string_set=frozenset(rare_strings)
    )

    # Disassembly-dependent modalities are not extracted here and remain untrusted
    opcode_features = OpcodeNgramFeatures(distribution=np.zeros(0), vocab=tuple(), trusted=False)
    cfg_features = CFGTopologyFeatures(percentiles=np.zeros(5), n_functions=0, trusted=False)

    sample = SampleFeatures(
        sample_id=sample_id, family=family, sha256=sha, platform="win32_pe",
        opcode_ngrams=opcode_features, cfg_topology=cfg_features,
        imports=import_features, string_entropy=string_features,
        section_metadata=section_features, byte_histogram=byte_features,
        temporal_signals={"pe_timestamp": pe_ts},
    )

    packed, reason = detect_packing(section_features, byte_features, packing_cfg)
    if packed:
        apply_packing_trust(sample, reason)

    return sample


def extract_apk_features(path: str | Path,
                         sample_id: str,
                         family: Optional[str],
                         packing_cfg: PackingConfig) -> SampleFeatures:
    """Extract φ(s) from an Android APK via androguard (static analysis only)."""
    data = Path(path).read_bytes()
    sha = _sha256_bytes(data)
    byte_features = _extract_byte_histogram(data)

    try:
        from androguard.core.apk import APK
        apk = APK(data)
        imports_set: Set[str] = set()
        for cls in apk.get_dex_names():
            imports_set.add(cls.lower())
        # APK timestamp from zip
        import zipfile
        with zipfile.ZipFile(path) as zf:
            zi = zf.infolist()[0] if zf.infolist() else None
            apk_ts = float(zi.date_time[0]) if zi else None  # year-only proxy
        section_features = SectionMetadataFeatures(sections=[{
            "name": ".dex", "entropy": _shannon_entropy(
                list(Counter(data[:65536]).values())),
            "vsize": len(data), "rsize": len(data), "characteristics": 0,
        }])
        import_features = ImportFeatures(symbol_set=frozenset(imports_set))
    except Exception:
        return extract_features_from_bytes(data, sample_id, family, "android", packing_cfg)

    printable = re.findall(rb'[\x20-\x7e]{4,}', data)
    str_ents = []
    rare_strings = set()
    for s in printable:
        sb = s.decode('ascii', errors='ignore')
        sc = Counter(sb)
        ent = _shannon_entropy(list(sc.values()))
        str_ents.append(ent)
        if ent > 3.5 and len(sb) >= 6:
            rare_strings.add(sb)
    if str_ents:
        ent_arr = np.array(str_ents, dtype=np.float64)
        ent_hist, _ = np.histogram(ent_arr, bins=16, range=(0.0, 8.0), density=False)
        ent_hist = ent_hist / max(ent_hist.sum(), 1.0)
    else:
        ent_hist = np.zeros(16, dtype=np.float64)
    string_features = StringEntropyFeatures(
        entropy_distribution=ent_hist, rare_string_set=frozenset(rare_strings)
    )
    opcode_features = OpcodeNgramFeatures(distribution=np.zeros(0), vocab=tuple(), trusted=False)
    cfg_features = CFGTopologyFeatures(percentiles=np.zeros(5), n_functions=0, trusted=False)

    sample = SampleFeatures(
        sample_id=sample_id, family=family, sha256=sha, platform="android",
        opcode_ngrams=opcode_features, cfg_topology=cfg_features,
        imports=import_features, string_entropy=string_features,
        section_metadata=section_features, byte_histogram=byte_features,
        temporal_signals={"apk_timestamp_year": apk_ts} if apk_ts else {},
    )

    packed, reason = detect_packing(section_features, byte_features, packing_cfg)
    if packed:
        apply_packing_trust(sample, reason)

    return sample


def extract(path_or_bytes: str | Path | bytes,
            sample_id: str,
            family: Optional[str],
            platform: str,
            packing_cfg: PackingConfig) -> SampleFeatures:
    """Dispatch to the correct extractor by platform / input type."""
    if isinstance(path_or_bytes, (str, Path)):
        p = Path(path_or_bytes)
        if platform == "win32_pe":
            return extract_pe_features(p, sample_id, family, packing_cfg)
        elif platform == "android":
            return extract_apk_features(p, sample_id, family, packing_cfg)
        else:
            return extract_features_from_bytes(p.read_bytes(), sample_id, family, platform, packing_cfg)
    else:
        return extract_features_from_bytes(path_or_bytes, sample_id, family, platform, packing_cfg)
