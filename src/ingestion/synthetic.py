"""Synthetic data generator (MovieLens-like) with controlled defect injection.

The clean data has a real latent structure (user taste vectors, genre
preferences, item quality/popularity) so that a personalised model has
something to learn that a popularity baseline cannot.  Defects are then
injected on purpose and logged so the validation stage can be checked
against ground truth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GENRES = ["Action", "Comedy", "Drama", "Sci-Fi", "Romance", "Thriller",
          "Horror", "Animation", "Documentary", "Fantasy", "Mystery", "Adventure"]
COUNTRIES = ["IN", "US", "UK", "DE", "BR", "CA"]
LATENT_DIM = 6


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def generate_clean(cfg: dict, rng: np.random.Generator):
    d = cfg["data"]
    n_users, n_items, G = d["n_users"], d["n_items"], len(GENRES)
    start, end = pd.Timestamp(d["start_date"]), pd.Timestamp(d["end_date"])

    # ---------------- items ----------------
    genre_share = rng.dirichlet(np.full(G, 3.0))
    primary = rng.choice(G, n_items, p=genre_share)
    n_genres = rng.choice([1, 2, 3], n_items, p=[0.45, 0.40, 0.15])
    genre_mat = np.zeros((n_items, G))
    genre_strings = []
    for i in range(n_items):
        others = [g for g in range(G) if g != primary[i]]
        extra = list(rng.choice(others, n_genres[i] - 1, replace=False))
        gs = [primary[i]] + extra
        genre_mat[i, gs] = 1
        genre_strings.append("|".join(GENRES[g] for g in gs))
    items = pd.DataFrame({
        "item_id": [f"I{i:04d}" for i in range(1, n_items + 1)],
        "title": [f"{GENRES[primary[i]]} Story {i + 1:04d}" for i in range(n_items)],
        "genres": genre_strings,
        "release_year": rng.integers(1980, 2024, n_items),
    })
    popularity = rng.lognormal(0.0, 1.1, n_items)
    quality = rng.normal(0.0, 0.45, n_items)
    item_vec = rng.normal(0, 1, (n_items, LATENT_DIM))

    # ---------------- users ----------------
    signup = start + pd.to_timedelta(rng.uniform(0, (end - start).days - 120, n_users), unit="D")
    users = pd.DataFrame({
        "user_id": [f"U{u:05d}" for u in range(1, n_users + 1)],
        "age": np.clip(rng.normal(34, 12, n_users), 14, 75).round().astype(int),
        "gender": rng.choice(["M", "F", "Other"], n_users, p=[0.48, 0.48, 0.04]),
        "country": rng.choice(COUNTRIES, n_users, p=[0.30, 0.25, 0.15, 0.10, 0.10, 0.10]),
        "signup_date": signup.normalize(),
    })

    # ---------------- interactions ----------------
    rows = []
    for u in range(n_users):
        pref = rng.dirichlet(np.full(G, 0.25))          # concentrated genre taste
        uvec = rng.normal(0, 1, LATENT_DIM)
        ubias = rng.normal(0, 0.4)
        # what the user chooses to consume depends on popularity AND taste
        score = np.log(popularity) + 4.0 * (genre_mat @ pref) + 0.5 * (item_vec @ uvec)
        n = int(np.clip(rng.lognormal(np.log(32), 0.7), 8, 250))
        idx = rng.choice(n_items, size=min(n, n_items), replace=False, p=_softmax(score))
        aff = genre_mat[idx] @ pref
        latent = (2.65 + quality[idx] + ubias + 0.35 * (item_vec[idx] @ uvec)
                  + 1.5 * (aff - 0.15) + rng.normal(0, 0.7, len(idx)))
        rating = np.clip(np.round(latent), 1, 5).astype(int)
        span = (end - signup[u]).total_seconds()
        ts = signup[u] + pd.to_timedelta(rng.uniform(0, span, len(idx)), unit="s")
        rows.append(pd.DataFrame({"user_id": users.user_id[u], "item_id": items.item_id.values[idx],
                                  "rating": rating, "timestamp": ts.floor("s")}))
    inter = pd.concat(rows, ignore_index=True).sample(frac=1.0, random_state=int(rng.integers(1e9)))
    return users, items, inter.reset_index(drop=True)


def inject_defects(users, items, inter, rng):
    """Corrupt the clean tables (as strings, like a real CSV) and return a ground-truth log."""
    u, i, x = users.astype(str).copy(), items.astype(str).copy(), inter.astype(str).copy()
    u["signup_date"] = pd.to_datetime(users["signup_date"]).dt.strftime("%Y-%m-%d")
    x["timestamp"] = pd.to_datetime(inter["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    log: dict[str, int] = {}

    # ---- messy-but-valid formatting (standardised later in preparation) ----
    g_map = {"M": ["M", "male", "Male"], "F": ["F", "female", "Female"], "Other": ["Other"]}
    u["gender"] = [rng.choice(g_map[v]) for v in users["gender"]]
    cmask = rng.random(len(u)) < 0.15
    u.loc[cmask, "country"] = [rng.choice([" %s " % c.lower(), c.lower()]) for c in u.loc[cmask, "country"]]
    gmask = rng.random(len(i)) < 0.10
    i.loc[gmask, "genres"] = i.loc[gmask, "genres"].str.lower()
    log["format_inconsistent_country_rows"] = int(cmask.sum())
    log["format_inconsistent_genre_rows"] = int(gmask.sum())

    # ---- interactions: disjoint random row groups ----
    perm = rng.permutation(len(x))
    def take(n):
        nonlocal perm
        out, perm = perm[:n], perm[n:]
        return out
    spec = {
        "missing_rating": 100, "missing_user_id": 30, "rating_out_of_range": 120,
        "rating_non_numeric": 30, "timestamp_future": 80, "timestamp_unparsable": 30,
        "timestamp_before_signup": 60, "orphan_user_id": 80, "orphan_item_id": 60,
    }
    g = {k: take(n) for k, n in spec.items()}
    x.loc[g["missing_rating"], "rating"] = ""
    x.loc[g["missing_user_id"], "user_id"] = ""
    x.loc[g["rating_out_of_range"], "rating"] = rng.choice(["0", "6", "7", "-1", "10", "99"], len(g["rating_out_of_range"]))
    x.loc[g["rating_non_numeric"], "rating"] = rng.choice(["five", "N/A", "?"], len(g["rating_non_numeric"]))
    x.loc[g["timestamp_future"], "timestamp"] = "2026-03-15 10:00:00"
    x.loc[g["timestamp_unparsable"], "timestamp"] = rng.choice(["not-a-date", "32/13/2023", "yesterday"], len(g["timestamp_unparsable"]))
    x.loc[g["timestamp_before_signup"], "timestamp"] = "2022-06-01 00:00:00"
    x.loc[g["orphan_user_id"], "user_id"] = [f"U{90000 + k}" for k in range(len(g["orphan_user_id"]))]
    x.loc[g["orphan_item_id"], "item_id"] = [f"I{9000 + k}" for k in range(len(g["orphan_item_id"]))]
    log.update({k: int(len(v)) for k, v in g.items()})

    clean_pool = perm  # rows untouched by the defects above
    exact = rng.choice(clean_pool, 150, replace=False)
    conflict = rng.choice(np.setdiff1d(clean_pool, exact), 60, replace=False)
    dup_exact = x.loc[exact].copy()
    dup_conf = x.loc[conflict].copy()
    dup_conf["rating"] = ((dup_conf["rating"].astype(int) % 5) + 1).astype(str)
    dup_conf["timestamp"] = (pd.to_datetime(dup_conf["timestamp"]) + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d %H:%M:%S")
    x = pd.concat([x, dup_exact, dup_conf], ignore_index=True)
    log["duplicate_exact_rows"], log["duplicate_user_item_conflicting"] = 150, 60

    # ---- users ----
    up = rng.permutation(len(u))
    n_miss_country, n_miss_age, n_bad_age, n_bad_signup, n_miss_id = 40, 25, 20, 8, 5
    a, up = up[:n_miss_country], up[n_miss_country:]; u.loc[a, "country"] = ""
    a, up = up[:n_miss_age], up[n_miss_age:]; u.loc[a, "age"] = ""
    a, up = up[:n_bad_age], up[n_bad_age:]; u.loc[a, "age"] = rng.choice(["0", "150", "-5", "abc"], len(a))
    a, up = up[:n_bad_signup], up[n_bad_signup:]; u.loc[a, "signup_date"] = "31-02-2023"
    a, up = up[:n_miss_id], up[n_miss_id:]; u.loc[a, "user_id"] = ""
    dup_u = u.loc[rng.choice(up, 10, replace=False)]
    u = pd.concat([u, dup_u], ignore_index=True)
    log.update({"users_missing_country": n_miss_country, "users_missing_age": n_miss_age,
                "users_invalid_age": n_bad_age, "users_invalid_signup_date": n_bad_signup,
                "users_missing_user_id": n_miss_id, "users_duplicate_exact_rows": 10})

    # ---- items ----
    ip = rng.permutation(len(i))
    a, ip = ip[:20], ip[20:]; i.loc[a, "title"] = ""
    a, ip = ip[:10], ip[10:]; i.loc[a, "release_year"] = rng.choice(["2099", "1066", "unknown"], len(a))
    a, ip = ip[:8], ip[8:]; i.loc[a, "genres"] = ""
    a, ip = ip[:3], ip[3:]; i.loc[a, "item_id"] = ""
    dup_key = i.loc[rng.choice(ip, 5, replace=False)].copy()
    dup_key["title"] = dup_key["title"] + " (re-issue)"
    i = pd.concat([i, dup_key], ignore_index=True)
    log.update({"items_missing_title": 20, "items_invalid_release_year": 10, "items_missing_genres": 8,
                "items_missing_item_id": 3, "items_duplicate_item_id": 5})

    x = x.sample(frac=1.0, random_state=int(rng.integers(1e9))).reset_index(drop=True)
    return u, i, x, log
