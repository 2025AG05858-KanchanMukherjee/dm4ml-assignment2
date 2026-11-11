#!/usr/bin/env python
"""CLI entry point:  python run_pipeline.py [--config configs/config.yaml] [--from-stage build_features] [--only train_mf]"""
import argparse
import sys

from src.orchestration.pipeline import build_dag, run_dag
from src.utils import load_config


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the end-to-end recommendation data pipeline")
    ap.add_argument("--config", default=None)
    ap.add_argument("--from-stage", default=None, choices=[t.name for t in build_dag()])
    ap.add_argument("--only", nargs="*", default=None, choices=[t.name for t in build_dag()])
    a = ap.parse_args()
    summary = run_dag(load_config(a.config), only=a.only, start_from=a.from_stage)
    for r in summary["tasks"]:
        print(f"{r['status']:<8} {r['task']:<18} {r['duration_s']:>6.1f}s")
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
