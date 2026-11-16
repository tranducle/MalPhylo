"""Structured JSON logging + evidence-ledger emission.

Per pipeline configuration schema: structured JSON logs; per-stage checkpointing;
output artifacts include metrics.json, run.log, evidence ledgers (JSONL).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional


class EvidenceLedger:
    """Append-only JSONL evidence ledger (Section 3.6 auditable evidence chain)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def record(self, event_type: str, payload: Dict[str, Any]) -> None:
        rec = {
            "ts": time.time(),
            "event": event_type,
            **payload,
        }
        self._fh.write(json.dumps(rec, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "EvidenceLedger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def setup_logger(run_dir: str | Path, run_id: str) -> logging.Logger:
    """Structured logger writing to run_dir/run.log + stderr."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"malphpylo.{run_id}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(run_dir / "run.log")
    sh = logging.StreamHandler()
    fmt = logging.Formatter(
        '{"ts":"%(asctime)s","run":"%(name)s","level":"%(levelname)s","msg":"%(message)s"}'
    )
    for h in (fh, sh):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def dump_metrics(run_dir: str | Path, metrics: Dict[str, Any]) -> None:
    """Write metrics.json for the run."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
