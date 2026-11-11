"""Shared helpers: config loading, paths, logging, seeding, JSON output."""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
_LOG_CONFIGURED = False


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(path) as fh:
        return yaml.safe_load(fh)


def path_for(cfg: dict, key: str) -> Path:
    """Resolve a directory declared under `paths:` and make sure it exists."""
    p = ROOT / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_logger(name: str) -> logging.Logger:
    global _LOG_CONFIGURED
    if not _LOG_CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        _LOG_CONFIGURED = True
    return logging.getLogger(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def write_json(obj, path: str | Path) -> None:
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=_default)


def read_json(path: str | Path):
    with open(path) as fh:
        return json.load(fh)
