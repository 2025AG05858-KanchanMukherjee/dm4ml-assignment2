"""Experiment tracking.

Uses MLflow when requested (backend: auto|mlflow).  Otherwise the same API
writes an MLflow-compatible *file store* under mlruns/:

    mlruns/<experiment>/<run_id>/{meta.yaml, params/<k>, metrics/<k>, artifacts/...}

so params, metrics and artifacts are tracked either way and the calling code
never changes.  (MLflow could not be installed in the offline build sandbox;
the MLflow branch is a thin, standard wrapper around mlflow.log_* calls.)
"""
from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path

import yaml

from src.utils import path_for


class Tracker:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.experiment = cfg["tracking"]["experiment"]
        backend = cfg["tracking"].get("backend", "auto")
        if backend not in ("auto", "mlflow", "local"):
            raise ValueError(f"unknown tracking.backend: {backend}")
        self._mlflow = None
        if backend in ("auto", "mlflow"):
            try:
                import mlflow  # noqa
                self._mlflow = mlflow
                mlflow.set_tracking_uri("file:" + str(path_for(cfg, "mlruns")))
                mlflow.set_experiment(self.experiment)
            except ImportError:
                if backend == "mlflow":
                    raise
        self.backend = "mlflow" if self._mlflow else "local-file-store"
        self.root = path_for(cfg, "mlruns") / self.experiment
        self.run_id = None
        self.run_dir: Path | None = None

    # ---- run lifecycle
    def start_run(self, run_name: str, run_id: str | None = None, tags: dict | None = None) -> str:
        if self._mlflow:
            r = self._mlflow.start_run(run_id=run_id, run_name=None if run_id else run_name, tags=tags)
            self.run_id = r.info.run_id
            return self.run_id
        self.run_id = run_id or uuid.uuid4().hex
        self.run_dir = self.root / self.run_id
        for sub in ("params", "metrics", "artifacts"):
            (self.run_dir / sub).mkdir(parents=True, exist_ok=True)
        if run_id is None:
            meta = {"run_id": self.run_id, "run_name": run_name, "experiment": self.experiment,
                    "start_time_ms": int(time.time() * 1000), "tags": tags or {}}
            (self.run_dir / "meta.yaml").write_text(yaml.safe_dump(meta))
        return self.run_id

    def end_run(self) -> None:
        if self._mlflow:
            self._mlflow.end_run()

    # ---- logging
    def log_params(self, params: dict) -> None:
        if self._mlflow:
            self._mlflow.log_params(params); return
        for k, v in params.items():
            (self.run_dir / "params" / k).write_text(str(v))

    def log_metrics(self, metrics: dict, step: int = 0) -> None:
        clean = {k.replace("@", "_at_"): float(v) for k, v in metrics.items() if v is not None}
        if self._mlflow:
            self._mlflow.log_metrics(clean, step=step); return
        now = int(time.time() * 1000)
        for k, v in clean.items():
            with open(self.run_dir / "metrics" / k, "a") as fh:
                fh.write(f"{now} {v} {step}\n")

    def log_artifact(self, path: str | Path) -> None:
        if self._mlflow:
            self._mlflow.log_artifact(str(path)); return
        shutil.copy(path, self.run_dir / "artifacts" / Path(path).name)
