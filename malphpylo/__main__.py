"""MalPhylo CLI entry point.

Usage:
    python -m malphpylo run configs/r_smoke_1.yaml
    python -m malphpylo run configs/r_smoke_1.yaml --limit 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import RunConfig, load_config
from .pipeline import run_pipeline


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="malphpylo", description="MalPhylo pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run a pipeline from a YAML config")
    run_p.add_argument("config", help="Path to run config YAML")
    run_p.add_argument("--limit", type=int, default=None,
                       help="Override sample_limit (for quick smoke runs)")
    run_p.add_argument("--seed", type=int, default=None, help="Override seed")

    args = parser.parse_args(argv)

    if args.cmd == "run":
        cfg = load_config(args.config)
        if args.limit is not None:
            # rebuild config with overridden limit (frozen dataclass -> replace)
            from dataclasses import replace
            cfg = replace(cfg, sample_limit=args.limit, run_id=f"{cfg.run_id}_n{args.limit}")
        if args.seed is not None:
            from dataclasses import replace
            cfg = replace(cfg, seed=args.seed)
        metrics = run_pipeline(cfg)
        print(json.dumps(metrics, indent=2, default=str))
        return 0 if metrics.get("acyclicity_verified") else 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
