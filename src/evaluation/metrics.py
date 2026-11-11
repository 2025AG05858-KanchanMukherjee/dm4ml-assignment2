"""Offline evaluation metrics (all implemented from scratch and unit-tested).

Ranking protocol
  * relevant item  = test interaction with rating >= relevance_threshold
  * candidates     = every catalogue item the user has NOT interacted with in train
  * evaluated users= users with >= 1 relevant test item
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def top_k_indices(scores: np.ndarray, seen: np.ndarray, k: int) -> np.ndarray:
    """Top-k item columns per row after masking already-seen items.
    A stable sort makes ties resolve by item index, so results are deterministic."""
    masked = np.where(seen, -np.inf, scores)
    return np.argsort(-masked, axis=1, kind="stable")[:, :k]


def precision_recall_ndcg(recs: np.ndarray, relevant: list[set], k: int) -> dict:
    p, r, n = [], [], []
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    for row, rel in zip(recs, relevant):
        hits = np.array([1.0 if it in rel else 0.0 for it in row[:k]])
        p.append(hits.sum() / k)
        r.append(hits.sum() / len(rel))
        dcg = float((hits * disc).sum())
        idcg = float(disc[: min(len(rel), k)].sum())
        n.append(dcg / idcg)
    return {f"precision@{k}": float(np.mean(p)), f"recall@{k}": float(np.mean(r)), f"ndcg@{k}": float(np.mean(n))}


def coverage(recs: np.ndarray, n_items: int) -> float:
    """Share of the catalogue that appears in at least one user's top-K list."""
    return len(np.unique(recs)) / n_items


def intra_list_diversity(recs: np.ndarray, item_vectors: np.ndarray) -> float:
    """Mean pairwise (1 - cosine similarity) between recommended items, averaged over users.
    Item vectors are genre multi-hot vectors, so 1.0 = no shared genres, 0.0 = identical genres."""
    norm = np.linalg.norm(item_vectors, axis=1, keepdims=True)
    V = item_vectors / np.where(norm == 0, 1, norm)
    vals = []
    for row in recs:
        sim = V[row] @ V[row].T
        k = len(row)
        vals.append(1.0 - (sim.sum() - np.trace(sim)) / (k * (k - 1)))
    return float(np.mean(vals))


def rating_errors(y_true, y_pred) -> dict:
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return {"rmse": float(np.sqrt(np.mean(d ** 2))), "mae": float(np.mean(np.abs(d)))}


def evaluate_model(model, train: pd.DataFrame, test: pd.DataFrame, fs, ks=(5, 10), threshold=4) -> dict:
    """Full evaluation of one fitted model on a held-out period."""
    rel_df = test[test.rating >= threshold]
    rel_df = rel_df[rel_df.item_id.isin(fs.iidx) & rel_df.user_id.isin(fs.uidx)]
    relevant = rel_df.groupby("user_id")["item_id"].apply(lambda s: {fs.iidx[i] for i in s})
    users = list(relevant.index)

    scores = model.score_matrix(users)
    seen = np.zeros(scores.shape, dtype=bool)
    tr = train[train.user_id.isin(users)]
    rows = tr.user_id.map({u: n for n, u in enumerate(users)}).values
    seen[rows, tr.item_id.map(fs.iidx).values] = True

    out = {"n_eval_users": len(users)}
    for k in ks:
        recs = top_k_indices(scores, seen, k)
        out.update(precision_recall_ndcg(recs, list(relevant.values), k))
        out[f"coverage@{k}"] = coverage(recs, len(fs.item_ids))
        out[f"diversity@{k}"] = intra_list_diversity(recs, fs.item_genre.values)
    t = test[test.item_id.isin(fs.iidx) & test.user_id.isin(fs.uidx)]
    out.update(rating_errors(t.rating.values, model.predict_rating(t.user_id.values, t.item_id.values)))
    out["n_rating_rows"] = len(t)
    return out
