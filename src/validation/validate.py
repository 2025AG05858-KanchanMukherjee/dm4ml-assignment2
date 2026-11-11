"""Stage 2 - Data validation.

Rule-based checks (a lightweight stand-in for Great Expectations / Pandera, which
could not be installed offline) for: missing values, duplicates, data types,
invalid ratings, invalid timestamps and referential integrity.

Severity model
  ERROR -> the record is unusable: it is moved to the error dataset (data/errors)
  WARN  -> the record is kept but the bad field is blanked (NaN) and logged;
           the preparation stage imputes it.
Every violation (one row can violate several rules) is written to
data/errors/validation_issues.csv; a human-readable summary goes to
reports/data_validation_report.md.
"""
from __future__ import annotations

import json

import pandas as pd

from src.utils import get_logger, path_for, read_json, write_json

log = get_logger("validation")
ERROR, WARN = "ERROR", "WARN"


def _blank(s: pd.Series) -> pd.Series:
    return s.isna() | s.astype(str).str.strip().eq("")


class Table:
    """Collects rule violations for one table."""

    def __init__(self, name: str, df: pd.DataFrame):
        self.name, self.df = name, df
        self.fatal = pd.Series(False, index=df.index)
        self.issues: list[pd.DataFrame] = []

    def flag(self, mask: pd.Series, rule: str, severity: str, desc: str, col: str | None = None):
        mask = pd.Series(mask, index=self.df.index).fillna(False).astype(bool)
        if not mask.any():
            return
        sub = self.df[mask]
        self.issues.append(pd.DataFrame({
            "table": self.name, "csv_line": sub.index + 2, "rule": rule, "severity": severity,
            "description": desc, "column": col or "",
            "offending_value": sub[col].astype(str).values if col else "",
            "record": [json.dumps(r) for r in sub.to_dict("records")],
        }))
        if severity == ERROR:
            self.fatal |= mask

    def issues_df(self) -> pd.DataFrame:
        cols = ["table", "csv_line", "rule", "severity", "description", "column", "offending_value", "record"]
        return pd.concat(self.issues, ignore_index=True) if self.issues else pd.DataFrame(columns=cols)

    def drop_duplicates(self, subset: list[str], rule: str, desc: str, keep: str = "first", order_by=None):
        """Flag duplicates among rows that are still valid (so each defect is reported once)."""
        cand = self.df[~self.fatal]
        if order_by is not None:
            cand = cand.loc[order_by[cand.index].sort_values(kind="stable").index]
        dup = cand.duplicated(subset=subset, keep=keep)
        mask = pd.Series(False, index=self.df.index)
        mask[dup[dup].index] = True
        self.flag(mask, rule, ERROR, desc)


def _read(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return df.apply(lambda c: c.str.strip())


# ---------------------------------------------------------------- users
def validate_users(df: pd.DataFrame, v: dict) -> tuple[Table, pd.DataFrame]:
    t = Table("users", df)
    t.flag(_blank(df.user_id), "missing_user_id", ERROR, "primary key is empty", "user_id")
    t.drop_duplicates(list(df.columns), "duplicate_row", "exact duplicate of an earlier row")
    t.flag(df.user_id.duplicated() & ~_blank(df.user_id) & ~t.fatal, "duplicate_key", ERROR,
           "user_id already present with different attributes", "user_id")
    signup = pd.to_datetime(df.signup_date, format="%Y-%m-%d", errors="coerce")
    t.flag(_blank(df.signup_date), "missing_signup_date", ERROR, "signup_date is empty", "signup_date")
    t.flag(~_blank(df.signup_date) & signup.isna(), "invalid_signup_date", ERROR,
           "signup_date is not a valid YYYY-MM-DD date", "signup_date")
    t.flag(signup > pd.Timestamp(v["reference_date"]), "future_signup_date", ERROR,
           "signup_date is after the reference date", "signup_date")
    age = pd.to_numeric(df.age, errors="coerce")
    t.flag(_blank(df.age), "missing_age", WARN, "age is empty (imputed later)", "age")
    t.flag(~_blank(df.age) & age.isna(), "invalid_dtype_age", WARN, "age is not numeric (set to NaN)", "age")
    t.flag(age.notna() & ((age < v["age_min"]) | (age > v["age_max"])), "age_out_of_range", WARN,
           f"age outside [{v['age_min']}, {v['age_max']}] (set to NaN)", "age")
    t.flag(_blank(df.country), "missing_country", WARN, "country is empty (imputed later)", "country")
    t.flag(_blank(df.gender), "missing_gender", WARN, "gender is empty (imputed later)", "gender")
    bad_age = age.isna() | (age < v["age_min"]) | (age > v["age_max"])
    typed = df.assign(age=age.where(~bad_age), signup_date=signup)[~t.fatal]
    return t, typed.reset_index(drop=True)


# ---------------------------------------------------------------- items
def validate_items(df: pd.DataFrame, v: dict) -> tuple[Table, pd.DataFrame]:
    t = Table("items", df)
    t.flag(_blank(df.item_id), "missing_item_id", ERROR, "primary key is empty", "item_id")
    t.drop_duplicates(list(df.columns), "duplicate_row", "exact duplicate of an earlier row")
    t.flag(df.item_id.duplicated() & ~_blank(df.item_id) & ~t.fatal, "duplicate_key", ERROR,
           "item_id already present with different attributes (first kept)", "item_id")
    yr = pd.to_numeric(df.release_year, errors="coerce")
    t.flag(~_blank(df.release_year) & yr.isna(), "invalid_dtype_release_year", WARN,
           "release_year is not numeric (set to NaN)", "release_year")
    t.flag(_blank(df.release_year), "missing_release_year", WARN, "release_year is empty", "release_year")
    t.flag(yr.notna() & ((yr < v["release_year_min"]) | (yr > v["release_year_max"])), "release_year_out_of_range",
           WARN, f"release_year outside [{v['release_year_min']}, {v['release_year_max']}] (set to NaN)", "release_year")
    t.flag(_blank(df.title), "missing_title", WARN, "title is empty (imputed later)", "title")
    t.flag(_blank(df.genres), "missing_genres", WARN, "genres is empty (imputed later)", "genres")
    bad = yr.isna() | (yr < v["release_year_min"]) | (yr > v["release_year_max"])
    typed = df.assign(release_year=yr.where(~bad))[~t.fatal]
    return t, typed.reset_index(drop=True)


# ---------------------------------------------------------------- interactions
def validate_interactions(df: pd.DataFrame, v: dict, users_ok: pd.DataFrame, items_ok: pd.DataFrame):
    t = Table("interactions", df)
    for c in ["user_id", "item_id", "rating", "timestamp"]:
        t.flag(_blank(df[c]), f"missing_{c}", ERROR, f"{c} is empty", c)

    r = pd.to_numeric(df.rating, errors="coerce")
    t.flag(~_blank(df.rating) & r.isna(), "invalid_dtype_rating", ERROR, "rating is not numeric", "rating")
    t.flag(r.notna() & ((r < v["rating_min"]) | (r > v["rating_max"])), "rating_out_of_range", ERROR,
           f"rating outside [{v['rating_min']}, {v['rating_max']}]", "rating")

    ts = pd.to_datetime(df.timestamp, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    t.flag(~_blank(df.timestamp) & ts.isna(), "invalid_timestamp", ERROR,
           "timestamp is not a valid 'YYYY-MM-DD HH:MM:SS' datetime", "timestamp")
    t.flag(ts > pd.Timestamp(v["reference_date"]), "timestamp_in_future", ERROR,
           "timestamp is after the reference date", "timestamp")
    t.flag(ts < pd.Timestamp(v["min_valid_timestamp"]), "timestamp_too_old", ERROR,
           "timestamp is before the minimum valid date", "timestamp")

    t.flag(~_blank(df.user_id) & ~df.user_id.isin(users_ok.user_id), "fk_user_not_found", ERROR,
           "user_id has no row in the validated users table (referential integrity)", "user_id")
    t.flag(~_blank(df.item_id) & ~df.item_id.isin(items_ok.item_id), "fk_item_not_found", ERROR,
           "item_id has no row in the validated items table (referential integrity)", "item_id")
    signup = df.user_id.map(users_ok.set_index("user_id").signup_date)
    t.flag(ts.notna() & signup.notna() & (ts < signup), "timestamp_before_signup", ERROR,
           "interaction happened before the user signed up", "timestamp")

    t.drop_duplicates(["user_id", "item_id", "rating", "timestamp"], "duplicate_row",
                      "exact duplicate of an earlier interaction")
    t.drop_duplicates(["user_id", "item_id"], "duplicate_user_item", "user already interacted with item; "
                      "only the most recent interaction is kept", keep="last", order_by=ts)

    ok = ~t.fatal
    typed = pd.DataFrame({"user_id": df.user_id[ok], "item_id": df.item_id[ok],
                          "rating": r[ok].astype(int), "timestamp": ts[ok]}).reset_index(drop=True)
    return t, typed


# ---------------------------------------------------------------- orchestration
def run(cfg: dict) -> dict:
    raw, interim, errors, reports = (path_for(cfg, k) for k in ("raw", "interim", "errors", "reports"))
    v = cfg["validation"]

    t_users, users_ok = validate_users(_read(raw / "users.csv"), v)
    t_items, items_ok = validate_items(_read(raw / "items.csv"), v)
    t_inter, inter_ok = validate_interactions(_read(raw / "interactions.csv"), v, users_ok, items_ok)

    users_ok.to_csv(interim / "users_valid.csv", index=False)
    items_ok.to_csv(interim / "items_valid.csv", index=False)
    inter_ok.to_csv(interim / "interactions_valid.csv", index=False)

    summary, all_issues = {}, []
    for t, ok in ((t_users, users_ok), (t_items, items_ok), (t_inter, inter_ok)):
        issues = t.issues_df()
        all_issues.append(issues)
        # error dataset = the full original record of every quarantined row + why
        bad = t.df[t.fatal].copy()
        why = (issues[issues.severity == ERROR].groupby("csv_line")["rule"].apply(lambda s: ";".join(sorted(set(s)))))
        bad["violated_rules"] = (bad.index + 2).map(why)
        bad.to_csv(errors / f"{t.name}_rejected.csv", index=False)
        summary[t.name] = {
            "rows_in": len(t.df), "rows_valid": len(ok), "rows_quarantined": int(t.fatal.sum()),
            "rows_with_warnings": int(issues[issues.severity == WARN].csv_line.nunique()) if len(issues) else 0,
            "rules": (issues.groupby(["rule", "severity"]).size().rename("violations").reset_index().to_dict("records")
                      if len(issues) else []),
        }
        assert summary[t.name]["rows_in"] == summary[t.name]["rows_valid"] + summary[t.name]["rows_quarantined"]
        log.info("%-12s in=%d valid=%d quarantined=%d", t.name, len(t.df), len(ok), int(t.fatal.sum()))

    pd.concat(all_issues, ignore_index=True).to_csv(errors / "validation_issues.csv", index=False)
    write_json(summary, reports / "data_validation_report.json")
    _write_markdown(summary, raw, reports)
    return summary


def _write_markdown(summary: dict, raw, reports) -> None:
    L = ["# Data Validation Report", "",
         "Generated automatically by `src/validation/validate.py`. ERROR rows are quarantined in `data/errors/`; "
         "WARN fields are blanked and imputed during preparation. One row can violate several rules.", "",
         "## Row reconciliation", "", "| Table | Rows in | Valid | Quarantined | Rows with warnings |",
         "|---|---:|---:|---:|---:|"]
    for n, s in summary.items():
        L.append(f"| {n} | {s['rows_in']} | {s['rows_valid']} | {s['rows_quarantined']} | {s['rows_with_warnings']} |")
    L += ["", "`Rows in = Valid + Quarantined` is asserted for every table.", "", "## Violations by rule", "",
          "| Table | Rule | Severity | Violations |", "|---|---|---|---:|"]
    for n, s in summary.items():
        for r in sorted(s["rules"], key=lambda r: -r["violations"]):
            L.append(f"| {n} | `{r['rule']}` | {r['severity']} | {r['violations']} |")
    inj = raw / "_injected_defects.json"
    if inj.exists():
        d = read_json(inj)
        det = {(n, r["rule"]): r["violations"] for n, s in summary.items() for r in s["rules"]}
        pairs = [
            ("Missing rating", d["missing_rating"], det.get(("interactions", "missing_rating"), 0)),
            ("Missing user_id (interactions)", d["missing_user_id"], det.get(("interactions", "missing_user_id"), 0)),
            ("Rating out of range", d["rating_out_of_range"], det.get(("interactions", "rating_out_of_range"), 0)),
            ("Non-numeric rating", d["rating_non_numeric"], det.get(("interactions", "invalid_dtype_rating"), 0)),
            ("Future timestamp", d["timestamp_future"], det.get(("interactions", "timestamp_in_future"), 0)),
            ("Unparsable timestamp", d["timestamp_unparsable"], det.get(("interactions", "invalid_timestamp"), 0)),
            ("Timestamp before signup", d["timestamp_before_signup"], det.get(("interactions", "timestamp_before_signup"), 0)),
            ("Orphan item_id", d["orphan_item_id"], det.get(("interactions", "fk_item_not_found"), 0)),
            ("Orphan user_id (+ cascade from rejected users)", d["orphan_user_id"], det.get(("interactions", "fk_user_not_found"), 0)),
            ("Exact duplicate interactions", d["duplicate_exact_rows"], det.get(("interactions", "duplicate_row"), 0)),
            ("Conflicting (user,item) duplicates", d["duplicate_user_item_conflicting"], det.get(("interactions", "duplicate_user_item"), 0)),
            ("Users: duplicate rows", d["users_duplicate_exact_rows"], det.get(("users", "duplicate_row"), 0)),
            ("Users: invalid signup_date", d["users_invalid_signup_date"], det.get(("users", "invalid_signup_date"), 0)),
            ("Users: invalid age (dtype + range)", d["users_invalid_age"], det.get(("users", "invalid_dtype_age"), 0) + det.get(("users", "age_out_of_range"), 0)),
            ("Users: missing age", d["users_missing_age"], det.get(("users", "missing_age"), 0)),
            ("Users: missing country", d["users_missing_country"], det.get(("users", "missing_country"), 0)),
            ("Items: duplicate item_id", d["items_duplicate_item_id"], det.get(("items", "duplicate_key"), 0)),
            ("Items: missing title", d["items_missing_title"], det.get(("items", "missing_title"), 0)),
            ("Items: missing genres", d["items_missing_genres"], det.get(("items", "missing_genres"), 0)),
        ]
        L += ["", "## Detection check against injected defects (synthetic data only)", "",
              "The generator logs exactly what it corrupted (`data/raw/_injected_defects.json`). "
              "Detected can exceed injected where one defect cascades (e.g. a rejected user makes that user's "
              "interactions orphans) and can be lower where a row is caught first by another rule "
              "(each defect is reported once, by the first rule that fires).", "",
              "| Defect | Injected | Detected |", "|---|---:|---:|"]
        L += [f"| {a} | {b} | {c} |" for a, b, c in pairs]
    (reports / "data_validation_report.md").write_text("\n".join(L) + "\n")
