"""Stage 3 - Data preparation: cleaning, imputation, de-duplication, standardisation.

Input : data/interim/*_valid.csv   (typed, ERROR rows already quarantined)
Output: data/processed/{users,items,interactions}.csv + reports/cleaning_summary.json
"""
from __future__ import annotations

import pandas as pd

from src.ingestion.synthetic import GENRES
from src.utils import get_logger, path_for, write_json

log = get_logger("preparation")
GENDER_MAP = {"m": "M", "male": "M", "f": "F", "female": "F", "other": "Other"}
CANON_GENRE = {g.lower(): g for g in GENRES}


def clean_users(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    stats = {}
    df["country"] = df["country"].fillna("").str.strip().str.upper().replace("", "UNKNOWN")
    stats["country_missing_filled"] = int((df["country"] == "UNKNOWN").sum())
    df["gender"] = df["gender"].fillna("").str.strip().str.lower().map(GENDER_MAP).fillna("Unknown")
    df["age_was_imputed"] = df["age"].isna()
    median_age = df["age"].median()
    df["age"] = df["age"].fillna(median_age).round().astype(int)
    stats["age_imputed_with_median"] = {"count": int(df["age_was_imputed"].sum()), "median": float(median_age)}
    n0 = len(df)
    df = df.drop_duplicates("user_id", keep="first")
    stats["duplicates_removed"] = n0 - len(df)
    return df.reset_index(drop=True), stats


def _std_genres(s) -> str:
    if not isinstance(s, str) or not s.strip():
        return "Unknown"
    parts = [CANON_GENRE.get(p.strip().lower(), p.strip().title()) for p in s.split("|") if p.strip()]
    return "|".join(dict.fromkeys(parts)) or "Unknown"   # order-preserving de-dup


def clean_items(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    stats = {}
    stats["genres_standardised_or_filled"] = int((df["genres"].fillna("") != df["genres"].map(_std_genres)).sum())
    df["genres"] = df["genres"].map(_std_genres)
    stats["title_missing_filled"] = int((df["title"].fillna("").str.strip() == "").sum())
    df["title"] = df["title"].fillna("").str.strip().replace("", "Unknown title")
    med = df["release_year"].median()
    stats["release_year_imputed_with_median"] = {"count": int(df["release_year"].isna().sum()), "median": float(med)}
    df["release_year"] = df["release_year"].fillna(med).astype(int)
    n0 = len(df)
    df = df.drop_duplicates("item_id", keep="first")
    stats["duplicates_removed"] = n0 - len(df)
    return df.reset_index(drop=True), stats


def clean_interactions(df: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["rating"] = df["rating"].astype(int)
    n0 = len(df)
    df = df.sort_values("timestamp", kind="stable").drop_duplicates(["user_id", "item_id"], keep="last")
    stats = {"duplicates_removed": n0 - len(df)}
    # final integrity guard: validation guarantees this, preparation re-asserts it
    assert df.user_id.isin(users.user_id).all() and df.item_id.isin(items.item_id).all(), "referential integrity broken"
    assert df.rating.between(1, 5).all()
    return df.reset_index(drop=True), stats


def run(cfg: dict) -> dict:
    interim, processed, reports = (path_for(cfg, k) for k in ("interim", "processed", "reports"))
    users, s_u = clean_users(pd.read_csv(interim / "users_valid.csv"))
    items, s_i = clean_items(pd.read_csv(interim / "items_valid.csv"))
    inter, s_x = clean_interactions(pd.read_csv(interim / "interactions_valid.csv"), users, items)
    users.to_csv(processed / "users.csv", index=False)
    items.to_csv(processed / "items.csv", index=False)
    inter.to_csv(processed / "interactions.csv", index=False)
    summary = {"users": {**s_u, "rows": len(users)}, "items": {**s_i, "rows": len(items)},
               "interactions": {**s_x, "rows": len(inter)}}
    write_json(summary, reports / "cleaning_summary.json")
    log.info("prepared users=%d items=%d interactions=%d", len(users), len(items), len(inter))
    return summary
