"""Stage 7 - Model evaluation on the time-held-out TEST period; compares all models."""
from __future__ import annotations

import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation.metrics import evaluate_model
from src.features.store import FeatureStore
from src.tracking.tracker import Tracker
from src.utils import get_logger, path_for, read_json, write_json

log = get_logger("evaluation")
MODEL_ORDER = ["popularity_baseline", "als_matrix_factorization", "hybrid_mf_popularity_genre"]
LABELS = {"popularity_baseline": "Popularity (baseline)", "als_matrix_factorization": "ALS matrix factorisation",
          "hybrid_mf_popularity_genre": "Hybrid (MF + pop + genre)"}


def run(cfg: dict) -> dict:
    processed, models_dir, reports = (path_for(cfg, k) for k in ("processed", "models", "reports"))
    train = pd.read_csv(processed / "train.csv", parse_dates=["timestamp"])
    test = pd.read_csv(processed / "test.csv", parse_dates=["timestamp"])
    fs = FeatureStore.load(path_for(cfg, "features"))
    registry = read_json(models_dir / "run_registry.json")
    ks, pk = cfg["evaluation"]["k_values"], cfg["evaluation"]["primary_k"]

    results = {}
    for name in MODEL_ORDER:
        with open(models_dir / f"{name}.pkl", "rb") as fh:
            model = pickle.load(fh)
        model.fs = fs                                   # same feature definitions as training
        res = evaluate_model(model, train, test, fs, ks=ks, threshold=cfg["split"]["relevance_threshold"])
        results[name] = res
        tr = Tracker(cfg)
        tr.start_run(name, run_id=registry[name])
        tr.log_metrics({f"test_{k}": v for k, v in res.items()})
        tr.end_run()
        log.info("%-28s P@%d=%.4f R@%d=%.4f NDCG@%d=%.4f cov=%.3f div=%.3f RMSE=%.3f", name, pk,
                 res[f"precision@{pk}"], pk, res[f"recall@{pk}"], pk, res[f"ndcg@{pk}"],
                 res[f"coverage@{pk}"], res[f"diversity@{pk}"], res["rmse"])

    df = pd.DataFrame(results).T.loc[MODEL_ORDER]
    df.index = [LABELS[i] for i in df.index]
    df.index.name = "model"
    df.to_csv(reports / "model_comparison.csv")
    write_json(results, reports / "model_comparison.json")
    _write_markdown(df, ks, reports)
    _plot(df, ks, reports)
    return results


def _write_markdown(df, ks, reports):
    cols = [c for k in ks for c in (f"precision@{k}", f"recall@{k}", f"ndcg@{k}", f"coverage@{k}", f"diversity@{k}")] + ["rmse", "mae"]
    L = ["# Model Evaluation Results", "",
         f"Held-out **test period** (newest interactions by time); evaluated on {int(df['n_eval_users'].iloc[0])} users with "
         "at least one relevant (rating >= 4) test item; already-seen items are excluded from recommendations. "
         f"RMSE/MAE are computed on {int(df['n_rating_rows'].iloc[0])} test ratings. Best value per column in **bold**.", "",
         "| Model | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    best = {c: (df[c].min() if c in ("rmse", "mae") else df[c].max()) for c in cols}
    for name, row in df.iterrows():
        cells = [f"**{row[c]:.4f}**" if abs(row[c] - best[c]) < 1e-12 else f"{row[c]:.4f}" for c in cols]
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L += ["", "Metric definitions: Precision@K, Recall@K, NDCG@K (binary relevance, log2 discount); Coverage@K = share of the "
          "catalogue appearing in any user's top-K; Diversity@K = mean pairwise (1 - cosine) distance between genre vectors "
          "of the K recommended items (intra-list diversity). RMSE/MAE: predicted vs actual rating (popularity uses the "
          "Bayesian item mean; hybrid reuses the MF rating head)."]
    (reports / "model_comparison.md").write_text("\n".join(L) + "\n")


def _plot(df, ks, reports):
    k = max(ks)
    metrics = [f"precision@{k}", f"recall@{k}", f"ndcg@{k}", f"coverage@{k}", f"diversity@{k}"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 3.4))
    colors = ["#9aa5b1", "#2f6fb0", "#e08a1e"]
    for ax, m in zip(axes, metrics):
        ax.bar(range(len(df)), df[m], color=colors)
        ax.set_title(m, fontsize=11)
        ax.set_xticks(range(len(df)), ["Pop.", "MF", "Hybrid"], fontsize=9)
        for x, v in enumerate(df[m]):
            ax.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, df[m].max() * 1.18)
    fig.tight_layout()
    (reports / "figures").mkdir(exist_ok=True)
    fig.savefig(reports / "figures" / "model_comparison.png", dpi=150)
    plt.close(fig)
