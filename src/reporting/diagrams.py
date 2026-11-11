"""Architecture diagram (Graphviz) and DAG-run figure (matplotlib, drawn from the real run log)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ARCH_DOT = r'''
digraph G {
  rankdir=TB; fontname="Helvetica"; nodesep=0.35; ranksep=0.45; size="8,10"; bgcolor="white";
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, margin="0.15,0.08", color="#444455"];
  edge [color="#555566", arrowsize=0.7, fontname="Helvetica", fontsize=9];

  subgraph cluster_src { label="Sources"; style="rounded,dashed"; color="#999999"; fontsize=11;
    syn [label="Synthetic generator\n(users, items, interactions\n+ injected defects)", fillcolor="#eef2f7"];
    csv [label="Real CSV files\n(e.g. MovieLens)", fillcolor="#eef2f7"]; }

  ing [label="1  Ingestion\nraw CSV + manifest (MD5)", fillcolor="#dbe9f7"];
  val [label="2  Validation\nnulls · dtypes · ratings\ntimestamps · duplicates · FK", fillcolor="#dbe9f7"];
  prep [label="3  Preparation\nimpute · de-dup · standardise", fillcolor="#dbe9f7"];
  trans [label="4  Transformation\ntime split · activity summary\npopularity · sparse matrix", fillcolor="#dbe9f7"];
  feat [label="5  Feature store\nuser · item · interaction\nfeatures (point-in-time)", fillcolor="#fde8c8"];

  subgraph cluster_train { label="6  Training (tracked)"; style="rounded"; color="#999999"; fontsize=11;
    m1 [label="Popularity\nbaseline", fillcolor="#e6f2e0"];
    m2 [label="ALS matrix\nfactorisation", fillcolor="#e6f2e0"];
    m3 [label="Hybrid\nMF + pop + genre", fillcolor="#e6f2e0"]; }

  ev [label="7  Evaluation\nP/R/NDCG@K · coverage\ndiversity · RMSE/MAE", fillcolor="#e6f2e0"];
  mlf [label="Experiment tracking\nparams · metrics · artifacts\n(MLflow / local store)", shape=cylinder, fillcolor="#f3e3f3"];
  err [label="data/errors\nquarantined rows +\nvalidation report", shape=note, fillcolor="#fadbd8"];
  rep [label="reports/\ncomparison · catalog\nrun log", shape=note, fillcolor="#f5f5f5"];
  srv [label="Prediction\nrecommend(user, K)\nreuses feature store", fillcolor="#fde8c8"];
  orch [label="8  Orchestration: Airflow DAG / local DAG runner (task dependencies, retries, run log)", shape=box, style="filled,dashed", fillcolor="#fffbe6"];

  syn -> ing; csv -> ing; ing -> val; val -> prep; val -> err [style=dashed, label="ERROR rows"];
  prep -> trans; trans -> feat; feat -> m1; feat -> m2; m2 -> m3 [label="MF head"]; feat -> m3;
  m1 -> ev; m2 -> ev; m3 -> ev; m1 -> mlf [style=dotted]; m2 -> mlf [style=dotted]; m3 -> mlf [style=dotted];
  ev -> mlf [style=dotted, label="test metrics"]; ev -> rep; feat -> srv [style=dashed]; m3 -> srv [style=dashed];
  orch -> ing [style=invis];
}
'''


def render_architecture(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dot = out_dir / "architecture.dot"
    dot.write_text(ARCH_DOT)
    for fmt in ("png", "svg"):
        subprocess.run(["dot", f"-T{fmt}", "-Gdpi=170", str(dot), "-o", str(out_dir / f"architecture.{fmt}")], check=True)


def render_dag_run(run: dict, out: Path) -> None:
    """Draw the executed DAG: node colour = task status, label = duration. Layout = dependency depth."""
    tasks = {t["task"]: t for t in run["tasks"]}
    depth: dict[str, int] = {}
    def d(n):
        if n not in depth:
            deps = [x for x in tasks[n]["deps"] if x in tasks]
            depth[n] = 0 if not deps else 1 + max(d(x) for x in deps)
        return depth[n]
    for n in tasks:
        d(n)
    cols: dict[int, list[str]] = {}
    for n, k in depth.items():
        cols.setdefault(k, []).append(n)
    pos = {n: (k * 2.5, -(i - (len(v) - 1) / 2) * 1.1) for k, v in cols.items() for i, n in enumerate(v)}
    colors = {"success": "#7bc47f", "failed": "#e5736b", "skipped": "#d9d9d9"}
    fig, ax = plt.subplots(figsize=(15, 4.6))
    for n, t in tasks.items():
        for dep in t["deps"]:
            if dep in pos:
                (x0, y0), (x1, y1) = pos[dep], pos[n]
                ax.annotate("", xy=(x1 - 0.95, y1), xytext=(x0 + 0.95, y0),
                            arrowprops=dict(arrowstyle="-|>", color="#667", lw=1.2, shrinkA=0, shrinkB=0))
    for n, t in tasks.items():
        x, y = pos[n]
        ax.add_patch(FancyBboxPatch((x - 0.95, y - 0.36), 1.9, 0.72, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=colors[t["status"]], ec="#334", lw=1.2))
        ax.text(x, y + 0.1, n, ha="center", va="center", fontsize=9.5, fontweight="bold")
        ax.text(x, y - 0.17, f"{t['status']} · {t['duration_s']:.1f}s", ha="center", va="center", fontsize=8.5)
    total = sum(t["duration_s"] for t in tasks.values())
    ok = sum(t["status"] == "success" for t in tasks.values())
    ax.set_title(f"DAG run  |  {ok}/{len(tasks)} tasks succeeded  |  total {total:.1f}s  |  started {run['started']}",
                 fontsize=11, loc="left")
    xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 1.3, max(xs) + 1.3); ax.set_ylim(min(ys) - 0.8, max(ys) + 0.8)
    ax.axis("off")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
