"""Workflow definition + a small dependency-aware runner.

The DAG below is the single definition of the workflow.  It is executed by
  * this module's `run_dag` (works anywhere: only needs the repo's requirements), and
  * dags/recommendation_pipeline_dag.py (same tasks/dependencies as an Apache Airflow DAG).

Runner guarantees: topological execution order, retries, fail-fast with downstream
tasks marked `skipped`, and a JSON run log (reports/pipeline_run_log.json) with
per-task status, duration, and result.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from src.evaluation import evaluate
from src.features import store
from src.ingestion import ingest
from src.preparation import clean
from src.reporting import diagrams
from src.training import train
from src.transformation import split_derive
from src.utils import get_logger, path_for, write_json
from src.validation import validate

log = get_logger("orchestration")


@dataclass
class Task:
    name: str
    fn: Callable[[dict], object]
    deps: list[str] = field(default_factory=list)
    retries: int = 1


def build_dag() -> list[Task]:
    return [
        Task("ingest_data", ingest.run),
        Task("validate_data", validate.run, ["ingest_data"]),
        Task("prepare_data", clean.run, ["validate_data"]),
        Task("transform_split", split_derive.run, ["prepare_data"]),
        Task("build_features", store.run, ["transform_split"]),
        Task("train_baseline", train.train_baseline, ["build_features"]),
        Task("train_mf", train.train_mf, ["build_features"]),
        Task("train_hybrid", train.train_hybrid, ["build_features", "train_mf"]),
        Task("evaluate_models", evaluate.run, ["train_baseline", "train_mf", "train_hybrid"]),
    ]


def topo_order(tasks: list[Task]) -> list[Task]:
    by_name, done, order = {t.name: t for t in tasks}, set(), []
    pending = list(tasks)
    while pending:
        ready = [t for t in pending if all(d in done for d in t.deps)]
        if not ready:
            raise ValueError("cycle or missing dependency in DAG")
        for t in ready:                       # tasks in the same 'ready' wave could run in parallel (Airflow does)
            order.append(t); done.add(t.name); pending.remove(t)
    return order


def run_dag(cfg: dict, only: list[str] | None = None, start_from: str | None = None) -> dict:
    tasks = topo_order(build_dag())
    names = [t.name for t in tasks]
    if start_from:
        tasks = [t for t in tasks if names.index(t.name) >= names.index(start_from)]
    if only:
        tasks = [t for t in tasks if t.name in only]

    status, records = {}, []
    run_started = datetime.now().isoformat(timespec="seconds")
    for t in tasks:
        rec = {"task": t.name, "deps": t.deps, "attempts": 0}
        failed_dep = [d for d in t.deps if status.get(d) in ("failed", "skipped")]
        if failed_dep:
            rec.update(status="skipped", reason=f"upstream failed: {failed_dep}", duration_s=0.0)
            status[t.name] = "skipped"; records.append(rec); continue
        t0 = time.time()
        for attempt in range(1, t.retries + 2):
            rec["attempts"] = attempt
            try:
                log.info("▶ %s (attempt %d)", t.name, attempt)
                result = t.fn(cfg)
                rec.update(status="success", result_keys=list(result.keys()) if isinstance(result, dict) else None)
                break
            except Exception as exc:                                  # noqa: BLE001
                rec.update(status="failed", error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc(limit=3))
                log.error("✖ %s failed: %s", t.name, exc)
        rec["duration_s"] = round(time.time() - t0, 2)
        status[t.name] = rec["status"]
        records.append(rec)
        log.info("%s %s in %.1fs", "✔" if rec["status"] == "success" else "✖", t.name, rec["duration_s"])

    summary = {"started": run_started, "finished": datetime.now().isoformat(timespec="seconds"),
               "config_seed": cfg["seed"], "success": all(r["status"] == "success" for r in records), "tasks": records}
    reports = path_for(cfg, "reports")
    write_json(summary, reports / "pipeline_run_log.json")
    try:
        diagrams.render_dag_run(summary, reports / "figures" / "dag_run.png")
    except Exception as exc:                                          # noqa: BLE001  (diagram is non-critical)
        log.warning("could not render DAG run figure: %s", exc)
    return summary
