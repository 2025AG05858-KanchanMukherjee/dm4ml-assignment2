"""Recommenders.  Every model exposes the same interface so the evaluator is model-agnostic:

    fit(train, fs)                      -> self
    score_matrix(user_ids)              -> ndarray (len(user_ids) x n_items), higher = better
    predict_rating(user_ids, item_ids)  -> ndarray of predicted ratings (for RMSE / MAE)

Item/user index spaces always come from the FeatureStore, which is how the
same feature definitions are reused in training and prediction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


def _zscore_rows(m: np.ndarray) -> np.ndarray:
    sd = m.std(axis=1, keepdims=True)
    return (m - m.mean(axis=1, keepdims=True)) / np.where(sd < 1e-9, 1.0, sd)


class PopularityRecommender:
    """Baseline: recommend the globally most-interacted items (same list for everyone)."""
    name = "popularity_baseline"

    def fit(self, train: pd.DataFrame, fs) -> "PopularityRecommender":
        self.fs = fs
        self.params = {"model": "popularity", "score": "train interaction count", "rating_head": "bayesian item mean"}
        return self

    def score_matrix(self, user_ids) -> np.ndarray:
        pop = self.fs.get_item_features(self.fs.item_ids).item_popularity.values.astype(float)
        return np.tile(pop, (len(list(user_ids)), 1))

    def predict_rating(self, user_ids, item_ids) -> np.ndarray:
        return self.fs.get_item_features(item_ids).item_bayes_rating.values

    def train_metrics(self, train: pd.DataFrame) -> dict:
        pred = self.predict_rating(train.user_id, train.item_id)   # in-sample
        return {"train_rmse": float(np.sqrt(np.mean((pred - train.rating.values) ** 2)))}


class ALSMatrixFactorization:
    """Biased matrix factorisation fitted with Alternating Least Squares (explicit ratings).

    r_ui ~ mu + b_u + b_i + p_u . q_i ; regularised (weighted-lambda) ridge solves per user/item.
    Users/items with no training data fall back to mu + biases (=0 for unknown users).
    """
    name = "als_matrix_factorization"

    def __init__(self, n_factors=16, reg=0.6, n_iters=12, bias_reg=10.0, seed=42):
        self.n_factors, self.reg, self.n_iters, self.bias_reg, self.seed = n_factors, reg, n_iters, bias_reg, seed

    def fit(self, train: pd.DataFrame, fs) -> "ALSMatrixFactorization":
        self.fs = fs
        nu, ni, k = len(fs.user_ids), len(fs.item_ids), self.n_factors
        u = train.user_id.map(fs.uidx).values
        i = train.item_id.map(fs.iidx).values
        r = train.rating.values.astype(float)
        self.mu = r.mean()
        bu, bi = np.zeros(nu), np.zeros(ni)
        for _ in range(5):                                           # damped alternating bias estimates
            bi = np.bincount(i, r - self.mu - bu[u], ni) / (self.bias_reg + np.bincount(i, minlength=ni))
            bu = np.bincount(u, r - self.mu - bi[i], nu) / (self.bias_reg + np.bincount(u, minlength=nu))
        self.bu, self.bi = bu, bi
        res = r - self.mu - bu[u] - bi[i]

        by_user = sparse.csr_matrix((res, (u, i)), shape=(nu, ni))
        by_item = by_user.T.tocsr()
        rng = np.random.default_rng(self.seed)
        P = 0.1 * rng.standard_normal((nu, k))
        Q = 0.1 * rng.standard_normal((ni, k))
        eye = np.eye(k)

        def solve(mat, fixed, out):
            for row in range(mat.shape[0]):
                s, e = mat.indptr[row], mat.indptr[row + 1]
                if s == e:
                    out[row] = 0.0
                    continue
                F = fixed[mat.indices[s:e]]
                out[row] = np.linalg.solve(F.T @ F + self.reg * (e - s) * eye, F.T @ mat.data[s:e])

        self.history = []
        for it in range(self.n_iters):
            solve(by_user, Q, P)
            solve(by_item, P, Q)
            pred = np.clip(self.mu + bu[u] + bi[i] + (P[u] * Q[i]).sum(1), 1, 5)
            self.history.append(float(np.sqrt(np.mean((pred - r) ** 2))))
        self.P, self.Q = P, Q
        self.params = {"model": "als_mf", "n_factors": k, "reg": self.reg, "n_iters": self.n_iters,
                       "bias_reg": self.bias_reg, "seed": self.seed}
        return self

    def score_matrix(self, user_ids) -> np.ndarray:
        rows = [self.fs.uidx.get(x, -1) for x in user_ids]
        out = np.empty((len(rows), len(self.fs.item_ids)))
        for n, r in enumerate(rows):
            if r < 0:                                             # unknown user: item-level prediction only
                out[n] = self.mu + self.bi
            else:
                out[n] = self.mu + self.bu[r] + self.bi + self.Q @ self.P[r]
        return np.clip(out, 1, 5)

    def predict_rating(self, user_ids, item_ids) -> np.ndarray:
        u = np.array([self.fs.uidx.get(x, -1) for x in user_ids])
        i = np.array([self.fs.iidx.get(x, -1) for x in item_ids])
        pred = np.full(len(u), self.mu, dtype=float)
        ok_i = i >= 0
        pred[ok_i] += self.bi[i[ok_i]]
        ok = (u >= 0) & ok_i
        pred[ok] += self.bu[u[ok]] + (self.P[u[ok]] * self.Q[i[ok]]).sum(1)
        return np.clip(pred, 1, 5)

    def train_metrics(self, train: pd.DataFrame) -> dict:
        pred = self.predict_rating(train.user_id, train.item_id)
        return {"train_rmse": float(np.sqrt(np.mean((pred - train.rating.values) ** 2))),
                "final_iter_rmse": self.history[-1]}


class HybridRecommender:
    """Blend of three per-user z-scored signals computed over all candidate items:

        score = w_mf * z(MF predicted rating) + w_pop * z(log1p popularity) + w_aff * z(genre affinity)

    MF captures personalised quality, popularity captures what people choose to consume,
    genre affinity (from the feature store) captures stated taste.  Weights are chosen by
    grid search on a validation window carved from the END of the training period.
    """
    name = "hybrid_mf_popularity_genre"

    def __init__(self, mf: ALSMatrixFactorization, w_mf=1.0, w_pop=1.0, w_aff=1.0):
        self.mf, self.w_mf, self.w_pop, self.w_aff = mf, w_mf, w_pop, w_aff

    def fit(self, train, fs):
        self.fs = fs
        if not hasattr(self.mf, "P"):
            self.mf.fit(train, fs)
        self.params = {"model": "hybrid", "w_mf": self.w_mf, "w_pop": self.w_pop, "w_aff": self.w_aff,
                       **{f"mf_{k}": v for k, v in self.mf.params.items() if k != "model"}}
        return self

    def score_matrix(self, user_ids) -> np.ndarray:
        user_ids = list(user_ids)
        pop = np.log1p(self.fs.get_item_features(self.fs.item_ids).item_popularity.values.astype(float))
        s = self.w_pop * _zscore_rows(np.tile(pop, (len(user_ids), 1)))
        if self.w_mf:
            s = s + self.w_mf * _zscore_rows(self.mf.score_matrix(user_ids))
        if self.w_aff:
            s = s + self.w_aff * _zscore_rows(self.fs.affinity_scores(user_ids))
        return s

    def predict_rating(self, user_ids, item_ids):
        return self.mf.predict_rating(user_ids, item_ids)       # rating head = the MF component

    def train_metrics(self, train):
        return self.mf.train_metrics(train)
