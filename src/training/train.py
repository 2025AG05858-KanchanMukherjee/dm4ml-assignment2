"""Stages 5-6 - Model training (3 tasks: baseline, MF, hybrid) with experiment tracking.

Each task: loads processed data + the persisted FeatureStore -> fits -> pickles the model
to models/ -> logs params, training metrics and artifacts to the tracker and records the
run id in models/run_registry.json so the evaluation stage can append test metrics to the
same run.
"""
from __future__ import annotations

import itertools
import pickle

import pandas as pd

from src.evaluation.metrics import evaluate_model
from src.features.store import FeatureStore
from src.tracking.tracker import Tracker
from src.training.models import ALSMatrixFactorization, HybridRecommender, PopularityRecommender
from src.utils import get_logger, path_for, read_json, write_json

log = get_logger("training")


def _load(cfg):
    processed = path_for(cfg, "processed")
    train = pd.read_csv(processed / "train.csv", parse_dates=["timestamp"])
    fs = FeatureStore.load(path_for(cfg, "features"))
    return train, fs, processed


def _register(cfg, key, run_id):
    reg_path = path_for(cfg, "models") / "run_registry.json"
    reg = read_json(reg_path) if reg_path.exists() else {}
    reg[key] = run_id
    write_json(reg, reg_path)


def _fit_and_track(cfg, model, train, fs, extra_params=None, extra_artifacts=()):
    models_dir = path_for(cfg, "models")
    tr = Tracker(cfg)
    run_id = tr.start_run(model.name, tags={"stage": "training", "backend": tr.backend})
    model.fit(train, fs)
    params = {**model.params, "seed": cfg["seed"], "as_of": str(fs.as_of), "n_train_rows": len(train),
              "feature_store_users": len(fs.user_ids), "feature_store_items": len(fs.item_ids), **(extra_params or {})}
    tr.log_params(params)
    tr.log_metrics(model.train_metrics(train))
    if hasattr(model, "history"):
        for step, v in enumerate(model.history):
            tr.log_metrics({"als_iter_train_rmse": v}, step=step)
    path = models_dir / f"{model.name}.pkl"
    with open(path, "wb") as fh:
        pickle.dump(model, fh)
    tr.log_artifact(path)
    for a in extra_artifacts:
        tr.log_artifact(a)
    tr.end_run()
    _register(cfg, model.name, run_id)
    log.info("trained %-28s run_id=%s backend=%s", model.name, run_id[:8], tr.backend)
    return model


def train_baseline(cfg: dict) -> dict:
    train, fs, _ = _load(cfg)
    m = _fit_and_track(cfg, PopularityRecommender(), train, fs)
    return {"model": m.name}


def train_mf(cfg: dict) -> dict:
    train, fs, _ = _load(cfg)
    p = cfg["models"]["als_mf"]
    m = _fit_and_track(cfg, ALSMatrixFactorization(seed=cfg["seed"], **p), train, fs)
    return {"model": m.name, "final_train_rmse": m.history[-1]}


def tune_hybrid_weights(cfg: dict, train: pd.DataFrame, fs: FeatureStore):
    """Grid-search blend weights on a validation window = newest `validation_fraction` of TRAIN.
    The test set is never touched here (no tuning on test)."""
    users = pd.read_csv(path_for(cfg, "processed") / "users.csv", parse_dates=["signup_date"])
    items = pd.read_csv(path_for(cfg, "processed") / "items.csv")
    cut = train.timestamp.quantile(1 - cfg["split"]["validation_fraction"])
    sub, val = train[train.timestamp <= cut], train[train.timestamp > cut]
    fs_sub = FeatureStore.build(sub, users, items, cfg)              # features as of the sub-train period only
    mf = ALSMatrixFactorization(seed=cfg["seed"], **cfg["models"]["als_mf"]).fit(sub, fs_sub)
    grid = cfg["models"]["hybrid"]["grid"]
    rows = []
    for w_mf, w_pop, w_aff in itertools.product(grid["w_mf"], grid["w_pop"], grid["w_aff"]):
        if w_mf == w_pop == w_aff == 0:
            continue
        h = HybridRecommender(mf, w_mf, w_pop, w_aff)
        h.fs = fs_sub
        res = evaluate_model(h, sub, val, fs_sub, ks=[cfg["evaluation"]["primary_k"]],
                             threshold=cfg["split"]["relevance_threshold"])
        rows.append({"w_mf": w_mf, "w_pop": w_pop, "w_aff": w_aff,
                     "val_ndcg": res[f"ndcg@{cfg['evaluation']['primary_k']}"]})
    res = pd.DataFrame(rows).sort_values("val_ndcg", ascending=False).reset_index(drop=True)
    return res


def train_hybrid(cfg: dict) -> dict:
    train, fs, _ = _load(cfg)
    reports = path_for(cfg, "reports")
    grid = tune_hybrid_weights(cfg, train, fs)
    grid.to_csv(reports / "hybrid_weight_search.csv", index=False)
    best = grid.iloc[0]
    log.info("hybrid weights: mf=%.1f pop=%.1f aff=%.1f (val NDCG %.4f)", best.w_mf, best.w_pop, best.w_aff, best.val_ndcg)
    mf = ALSMatrixFactorization(seed=cfg["seed"], **cfg["models"]["als_mf"])
    h = HybridRecommender(mf, best.w_mf, best.w_pop, best.w_aff)
    _fit_and_track(cfg, h, train, fs, {"val_ndcg_at_primary_k": float(best.val_ndcg),
                                       "tuning_grid_size": len(grid)}, [reports / "hybrid_weight_search.csv"])
    return {"model": h.name, "w_mf": best.w_mf, "w_pop": best.w_pop, "w_aff": best.w_aff}
