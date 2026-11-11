"""Stage 1 - Data ingestion.

Produces three raw CSVs (users, items, interactions) in data/raw and a manifest
(row counts, columns, MD5) so every downstream run can be traced to its input.
Sources: `synthetic` (default, documented generator) or `csv` (copy real files
such as MovieLens that were mapped to the three-table schema, see README).
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingestion.synthetic import generate_clean, inject_defects
from src.utils import get_logger, path_for, set_seed, write_json

log = get_logger("ingestion")
REQUIRED = {
    "users": ["user_id", "age", "gender", "country", "signup_date"],
    "items": ["item_id", "title", "genres", "release_year"],
    "interactions": ["user_id", "item_id", "rating", "timestamp"],
}


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cfg: dict) -> dict:
    raw = path_for(cfg, "raw")
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    src = cfg["data"]["source"]

    if src == "synthetic":
        users, items, inter = generate_clean(cfg, rng)
        if cfg["data"].get("inject_defects", True):
            users, items, inter, defects = inject_defects(users, items, inter, rng)
            write_json(defects, raw / "_injected_defects.json")
        else:
            users = users.assign(signup_date=users.signup_date.dt.strftime("%Y-%m-%d"))
            inter = inter.assign(timestamp=inter.timestamp.dt.strftime("%Y-%m-%d %H:%M:%S"))
        users.to_csv(raw / "users.csv", index=False)
        items.to_csv(raw / "items.csv", index=False)
        inter.to_csv(raw / "interactions.csv", index=False)
    elif src == "csv":
        src_dir = Path(cfg["data"]["csv_dir"])
        for name in REQUIRED:
            shutil.copy(src_dir / f"{name}.csv", raw / f"{name}.csv")
    else:
        raise ValueError(f"unknown data.source: {src}")

    manifest = {}
    for name, cols in REQUIRED.items():
        f = raw / f"{name}.csv"
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{f.name} lacks required columns {missing}")
        manifest[name] = {"rows": len(df), "columns": list(df.columns), "md5": _md5(f)}
        log.info("ingested %-12s rows=%d", name, len(df))
    write_json(manifest, raw / "_manifest.json")
    return manifest
