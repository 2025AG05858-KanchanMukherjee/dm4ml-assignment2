"""Stage 4 - Data transformation.

* time-based train/test split (no random shuffling -> no look-ahead leakage)
* derived datasets built from TRAIN only: user activity summary, item popularity,
  and the sparse user-item interaction matrix (+ index maps)
"""
from __future__ import annotations

import pandas as pd
from scipy import sparse

from src.utils import get_logger, path_for, write_json

log = get_logger("transformation")


def time_split(inter: pd.DataFrame, test_fraction: float):
    """Newest `test_fraction` of interactions (by timestamp) become the test set."""
    cutoff = inter["timestamp"].quantile(1 - test_fraction)
    train = inter[inter.timestamp <= cutoff]
    test = inter[inter.timestamp > cutoff]
    return train.reset_index(drop=True), test.reset_index(drop=True), cutoff


def user_activity_summary(train: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    g = train.groupby("user_id")
    out = g.agg(n_interactions=("item_id", "size"), avg_rating=("rating", "mean"),
                first_interaction=("timestamp", "min"), last_interaction=("timestamp", "max"),
                n_active_days=("timestamp", lambda s: s.dt.normalize().nunique())).reset_index()
    out["days_since_last_interaction"] = (as_of - out["last_interaction"]).dt.total_seconds() / 86400
    return out


def item_popularity(train: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    g = train.groupby("item_id").agg(n_interactions=("user_id", "size"), avg_rating=("rating", "mean")).reset_index()
    out = items[["item_id", "title"]].merge(g, on="item_id", how="left")
    out["n_interactions"] = out["n_interactions"].fillna(0).astype(int)
    out["popularity_rank"] = out["n_interactions"].rank(ascending=False, method="min").astype(int)
    return out.sort_values("popularity_rank").reset_index(drop=True)


def interaction_matrix(train: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame):
    uidx = {u: k for k, u in enumerate(users.user_id)}
    iidx = {i: k for k, i in enumerate(items.item_id)}
    m = sparse.csr_matrix((train.rating.values.astype(float),
                           (train.user_id.map(uidx).values, train.item_id.map(iidx).values)),
                          shape=(len(uidx), len(iidx)))
    return m, uidx, iidx


def run(cfg: dict) -> dict:
    processed = path_for(cfg, "processed")
    users = pd.read_csv(processed / "users.csv", parse_dates=["signup_date"])
    items = pd.read_csv(processed / "items.csv")
    inter = pd.read_csv(processed / "interactions.csv", parse_dates=["timestamp"])

    train, test, cutoff = time_split(inter, cfg["split"]["test_fraction"])
    assert train.timestamp.max() < test.timestamp.min() or train.timestamp.max() <= cutoff
    train.to_csv(processed / "train.csv", index=False)
    test.to_csv(processed / "test.csv", index=False)

    as_of = train.timestamp.max()
    user_activity_summary(train, as_of).to_csv(processed / "user_activity_summary.csv", index=False)
    item_popularity(train, items).to_csv(processed / "item_popularity.csv", index=False)
    m, uidx, iidx = interaction_matrix(train, users, items)
    sparse.save_npz(processed / "interaction_matrix.npz", m)
    pd.Series(uidx, name="row").rename_axis("user_id").reset_index().to_csv(processed / "matrix_user_index.csv", index=False)
    pd.Series(iidx, name="col").rename_axis("item_id").reset_index().to_csv(processed / "matrix_item_index.csv", index=False)

    n_cold = int((~test.user_id.isin(train.user_id)).sum())
    info = {"cutoff": str(cutoff), "train_rows": len(train), "test_rows": len(test),
            "train_range": [str(train.timestamp.min()), str(train.timestamp.max())],
            "test_range": [str(test.timestamp.min()), str(test.timestamp.max())],
            "matrix_shape": list(m.shape), "matrix_density": float(m.nnz / (m.shape[0] * m.shape[1])),
            "test_rows_from_users_unseen_in_train": n_cold}
    write_json(info, path_for(cfg, "reports") / "split_summary.json")
    log.info("split cutoff=%s train=%d test=%d density=%.3f", cutoff, len(train), len(test), info["matrix_density"])
    return info
