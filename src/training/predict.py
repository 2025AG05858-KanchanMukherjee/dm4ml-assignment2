"""Online prediction: top-K recommendations for one user, using the persisted model + feature store.

    python -m src.training.predict --user U00010 --k 10
Unknown users (cold start) automatically receive popularity-driven recommendations,
because the feature store returns documented defaults (empty history, uniform genre affinity).
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd

from src.evaluation.metrics import top_k_indices
from src.features.store import FeatureStore
from src.utils import load_config, path_for


def recommend(user_id: str, k: int = 10, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    fs = FeatureStore.load(path_for(cfg, "features"))
    with open(path_for(cfg, "models") / "hybrid_mf_popularity_genre.pkl", "rb") as fh:
        model = pickle.load(fh)
    model.fs = fs
    model.mf.fs = fs
    train = pd.read_csv(path_for(cfg, "processed") / "train.csv")
    items = pd.read_csv(path_for(cfg, "processed") / "items.csv").set_index("item_id")

    scores = model.score_matrix([user_id])
    seen = np.zeros_like(scores, dtype=bool)
    for it in train.loc[train.user_id == user_id, "item_id"]:
        seen[0, fs.iidx[it]] = True
    idx = top_k_indices(scores, seen, k)[0]
    ids = [fs.item_ids[i] for i in idx]
    out = items.loc[ids, ["title", "genres"]].reset_index()
    out.insert(0, "rank", range(1, k + 1))
    out["score"] = scores[0, idx].round(3)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    print(recommend(a.user, a.k).to_string(index=False))
