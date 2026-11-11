"""Feature management layer (a small, file-backed feature store).

Why custom instead of Feast: no network/DB needed, ~150 lines, and it demonstrates
the two ideas that matter -- ONE definition of every feature shared by training and
prediction, and POINT-IN-TIME correctness (features only see train data up to `as_of`).

Usage (identical in training, evaluation and serving):
    fs = FeatureStore.build(train, users, items, cfg)   # compute (training time)
    fs.save(dir) ;  fs = FeatureStore.load(dir)         # persist / reuse (serving time)
    fs.get_user_features(["U00001"]) ; fs.get_pair_features(users, items)
    fs.affinity_scores(users)  # (n_users x n_items) matrix for fast candidate scoring
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import sparse

from src.utils import ROOT, write_json


def load_registry() -> dict:
    with open(ROOT / "configs" / "feature_registry.yaml") as fh:
        return yaml.safe_load(fh)


class FeatureStore:
    def __init__(self, user_features, item_features, affinity, item_genre, genres, as_of, global_stats, users_meta):
        self.user_features = user_features.set_index("user_id")
        self.item_features = item_features.set_index("item_id")
        self.affinity = affinity.set_index("user_id")[genres]       # user x genre
        self.item_genre = item_genre                                  # item x genre (0/1), index = item_id
        self.genres = list(genres)
        self.as_of = pd.Timestamp(as_of)
        self.global_stats = global_stats
        self.users_meta = users_meta.set_index("user_id")
        # fixed, shared index spaces for every model
        self.user_ids = list(self.users_meta.index)
        self.item_ids = list(self.item_genre.index)
        self.uidx = {u: k for k, u in enumerate(self.user_ids)}
        self.iidx = {i: k for k, i in enumerate(self.item_ids)}

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, train: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame, cfg: dict) -> "FeatureStore":
        thr = cfg["split"]["relevance_threshold"]
        C = cfg["features"]["bayes_prior_strength"]
        as_of = train["timestamp"].max()
        gmean = float(train.rating.mean())
        gstats = {"global_mean_rating": gmean, "global_like_ratio": float((train.rating >= thr).mean()),
                  "median_age": float(users.age.median()), "n_train": int(len(train))}

        # ---- genre structures
        genres = sorted({g for s in items.genres for g in s.split("|")})
        item_genre = pd.DataFrame(0.0, index=items.item_id.values, columns=genres)
        for iid, s in zip(items.item_id, items.genres):
            item_genre.loc[iid, s.split("|")] = 1.0
        n_gen = item_genre.sum(axis=1)

        # ---- interaction feature: user x genre affinity
        uids, iids = list(users.user_id), list(item_genre.index)
        uidx, iidx = {u: k for k, u in enumerate(uids)}, {i: k for k, i in enumerate(iids)}
        R = sparse.csr_matrix((train.rating.values.astype(float),
                               (train.user_id.map(uidx).values, train.item_id.map(iidx).values)),
                              shape=(len(uids), len(iids)))
        aff = np.asarray(R @ (item_genre.values / n_gen.values[:, None]))     # sparse @ dense -> dense
        row = aff.sum(axis=1, keepdims=True)
        aff = np.where(row > 0, aff / np.where(row == 0, 1, row), 1.0 / len(genres))     # cold users -> uniform
        affinity = pd.DataFrame(aff, columns=genres).assign(user_id=uids)

        # ---- user features
        g = train.groupby("user_id")
        uf = pd.DataFrame({
            "user_activity_count": g.size(), "user_avg_rating": g.rating.mean(),
            "user_rating_std": g.rating.std().fillna(0.0),
            "user_like_ratio": g.rating.apply(lambda s: (s >= thr).mean()),
            "days_since_last_interaction": (as_of - g.timestamp.max()).dt.total_seconds() / 86400,
            "user_history_days": (g.timestamp.max() - g.timestamp.min()).dt.total_seconds() / 86400,
        }).reset_index()
        uf = users[["user_id", "age", "signup_date"]].merge(uf, on="user_id", how="left")
        uf["user_tenure_days"] = (as_of - pd.to_datetime(uf.signup_date)).dt.total_seconds() / 86400
        uf["user_activity_count"] = uf.user_activity_count.fillna(0).astype(int)
        for col, val in {"user_avg_rating": gmean, "user_rating_std": 0.0, "user_like_ratio": gstats["global_like_ratio"],
                         "user_history_days": 0.0}.items():
            uf[col] = uf[col].fillna(val)
        uf["days_since_last_interaction"] = uf.days_since_last_interaction.fillna(uf.user_tenure_days)
        pref_idx = aff.argmax(axis=1)
        pref = np.array(genres)[pref_idx]
        has_hist = (uf.user_activity_count.values > 0)
        uf["preferred_category"] = np.where(has_hist, pref, "Unknown")
        uf["preferred_category_share"] = np.where(has_hist, aff.max(axis=1), 0.0)
        uf = uf.drop(columns="signup_date")

        # ---- item features
        gi = train.groupby("item_id")
        it = pd.DataFrame({"item_popularity": gi.size(), "item_avg_rating": gi.rating.mean(), "_sum": gi.rating.sum(),
                           "item_like_ratio": gi.rating.apply(lambda s: (s >= thr).mean()),
                           "item_days_since_last_interaction": (as_of - gi.timestamp.max()).dt.total_seconds() / 86400})
        it = items[["item_id", "release_year", "genres"]].merge(it.reset_index(), on="item_id", how="left")
        it["item_popularity"] = it.item_popularity.fillna(0).astype(int)
        it["_sum"] = it["_sum"].fillna(0.0)
        it["item_avg_rating"] = it.item_avg_rating.fillna(gmean)
        it["item_bayes_rating"] = (C * gmean + it["_sum"]) / (C + it.item_popularity)
        it["item_like_ratio"] = it.item_like_ratio.fillna(gstats["global_like_ratio"])
        it["item_popularity_rank"] = it.item_popularity.rank(ascending=False, method="min").astype(int)
        it["item_age_years"] = (as_of.year - it.release_year).astype(float)
        it["item_days_since_last_interaction"] = it.item_days_since_last_interaction.fillna(
            (as_of - train.timestamp.min()).total_seconds() / 86400)
        it["item_n_genres"] = it.genres.str.count(r"\|") + 1
        it["item_primary_genre"] = it.genres.str.split("|").str[0]
        it = it.drop(columns=["_sum", "release_year", "genres"])

        item_genre_for_store = item_genre.reset_index().rename(columns={"index": "item_id"}).set_index("item_id")
        fs = cls(uf, it, affinity, item_genre_for_store, genres, as_of, gstats,
                 users[["user_id"]].assign(age=users.age.values, signup_date=pd.to_datetime(users.signup_date).values))
        fs._check_against_registry()
        return fs

    def _check_against_registry(self):
        reg = load_registry()
        for entity, df in (("user_features", self.user_features), ("item_features", self.item_features)):
            declared = {f["name"] for f in reg[entity]}
            assert declared == set(df.columns), f"{entity} drift: {declared ^ set(df.columns)}"

    # ------------------------------------------------------------------ persistence
    def save(self, directory: Path) -> None:
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        self.user_features.reset_index().to_csv(directory / "user_features.csv", index=False)
        self.item_features.reset_index().to_csv(directory / "item_features.csv", index=False)
        self.affinity.reset_index().to_csv(directory / "user_genre_affinity.csv", index=False)
        self.item_genre.reset_index().to_csv(directory / "item_genre_matrix.csv", index=False)
        self.users_meta.reset_index().to_csv(directory / "users_meta.csv", index=False)
        write_json({"as_of": str(self.as_of), "genres": self.genres, "global_stats": self.global_stats},
                   directory / "feature_meta.json")

    @classmethod
    def load(cls, directory: Path) -> "FeatureStore":
        import json
        d = Path(directory)
        meta = json.load(open(d / "feature_meta.json"))
        return cls(pd.read_csv(d / "user_features.csv"), pd.read_csv(d / "item_features.csv"),
                   pd.read_csv(d / "user_genre_affinity.csv"), pd.read_csv(d / "item_genre_matrix.csv").set_index("item_id"),
                   meta["genres"], meta["as_of"], meta["global_stats"],
                   pd.read_csv(d / "users_meta.csv", parse_dates=["signup_date"]))

    # ------------------------------------------------------------------ online lookups
    def get_user_features(self, user_ids) -> pd.DataFrame:
        """Unknown users (true cold start) receive documented defaults from the registry."""
        ids = list(user_ids)
        out = self.user_features.reindex(ids)
        g = self.global_stats
        defaults = {"user_activity_count": 0, "user_avg_rating": g["global_mean_rating"], "user_rating_std": 0.0,
                    "user_like_ratio": g["global_like_ratio"], "days_since_last_interaction": np.nan,
                    "user_history_days": 0.0, "user_tenure_days": 0.0, "preferred_category": "Unknown",
                    "preferred_category_share": 0.0, "age": g["median_age"]}
        for c, v in defaults.items():
            out[c] = out[c].fillna(v)
        out["days_since_last_interaction"] = out["days_since_last_interaction"].fillna(out["user_tenure_days"])
        return out

    def get_item_features(self, item_ids) -> pd.DataFrame:
        ids = list(item_ids)
        out = self.item_features.reindex(ids)
        g = self.global_stats
        out["item_popularity"] = out.item_popularity.fillna(0)
        out["item_bayes_rating"] = out.item_bayes_rating.fillna(g["global_mean_rating"])
        return out

    def affinity_scores(self, user_ids) -> np.ndarray:
        """(len(user_ids) x n_items) genre_affinity_score; unknown users get a uniform profile."""
        A = self.affinity.reindex(list(user_ids)).fillna(1.0 / len(self.genres)).values
        n_gen = self.item_genre.values.sum(axis=1)
        return A @ (self.item_genre.values / n_gen[:, None]).T

    def get_pair_features(self, user_ids, item_ids) -> pd.DataFrame:
        u = self.get_user_features(user_ids).reset_index(drop=True)
        i = self.get_item_features(item_ids).reset_index(drop=True)
        A = self.affinity.reindex(list(user_ids)).fillna(1.0 / len(self.genres))
        G = self.item_genre.reindex(list(item_ids)).fillna(0.0)
        n = G.values.sum(axis=1)
        aff = (A.values * G.values).sum(axis=1) / np.where(n == 0, 1, n)
        return pd.DataFrame({
            "user_id": list(user_ids), "item_id": list(item_ids), "genre_affinity_score": aff,
            "item_bayes_minus_user_avg": i.item_bayes_rating.values - u.user_avg_rating.values,
            "preferred_category_match": (i.item_primary_genre.values == u.preferred_category.values).astype(int)})


def write_catalog(cfg: dict, reports_dir: Path) -> pd.DataFrame:
    reg = load_registry()
    rows = []
    for entity, label in (("user_features", "user"), ("item_features", "item"), ("interaction_features", "interaction")):
        for f in reg[entity]:
            rows.append({"entity": label, "feature": f["name"], "dtype": f["dtype"], "description": f["description"],
                         "formula": f["formula"], "cold_start_default": f["default_cold_start"]})
    cat = pd.DataFrame(rows)
    cat.to_csv(reports_dir / "feature_catalog.csv", index=False)
    md = ["# Feature Catalog", "",
          "Generated from `configs/feature_registry.yaml`. All features are computed on the **training period only**, "
          f"as of the last training timestamp (point-in-time safe), bad relevance threshold = {cfg['split']['relevance_threshold']}, "
          f"Bayesian prior strength C = {cfg['features']['bayes_prior_strength']}. The same `FeatureStore` code path serves "
          "model training, evaluation and online prediction.", "",
          "| Entity | Feature | Type | Description | Formula | Cold-start default |", "|---|---|---|---|---|---|"]
    md += [f"| {r.entity} | `{r.feature}` | {r.dtype} | {r.description} | `{r.formula}` | {r.cold_start_default} |"
           for r in cat.itertuples()]
    (reports_dir / "feature_catalog.md").write_text("\n".join(md).replace("bad relevance", "relevance") + "\n")
    return cat


def run(cfg: dict) -> dict:
    from src.utils import get_logger, path_for
    log = get_logger("features")
    processed, feats, reports = path_for(cfg, "processed"), path_for(cfg, "features"), path_for(cfg, "reports")
    train = pd.read_csv(processed / "train.csv", parse_dates=["timestamp"])
    users = pd.read_csv(processed / "users.csv", parse_dates=["signup_date"])
    items = pd.read_csv(processed / "items.csv")
    fs = FeatureStore.build(train, users, items, cfg)
    fs.save(feats)
    cat = write_catalog(cfg, reports)
    log.info("materialised %d users x %d items, %d catalogued features", len(fs.user_ids), len(fs.item_ids), len(cat))
    return {"n_users": len(fs.user_ids), "n_items": len(fs.item_ids), "n_features": len(cat), "as_of": str(fs.as_of)}
