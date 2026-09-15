import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shared.models.metrics import MetricHandler

RUN_DIR = os.environ.get("RUN_DIR")
if not RUN_DIR:
    raise ValueError("RUN_DIR environment variable is not set")

RUN_DIR = Path(RUN_DIR).resolve()
RUN_NAME = RUN_DIR.name

CHECKPOINT_PATH = RUN_DIR / "best.ckpt"
LATENT_PATH = RUN_DIR / "latents_best_ckpt.npz"   
OUT_DIR = RUN_DIR / "s1_latent_space"
TOPK_HISTORY_PATH = PROJECT_ROOT / "logs" / RUN_NAME / "topk_history.csv"

DCI_ALPHA = 0.1
STABLE_THRESHOLD = 0.1
STABLE_TOP_N = 3
DISPLAY_TOP_N = 5

RANKING_COLUMN = "test_r2"
PLOT_TOP_N = 20
SELECTED_COLOR = "#2E8B57"
UNSELECTED_COLOR = "#C0504D"

NAME_PREFIXES = {
    "lh_hippo_": "L hippo ",
    "rh_hippo_": "R hippo ",
    "lh_amyg_": "L amyg ",
    "rh_amyg_": "R amyg ",
    "wmparc_wm-lh-": "L wm ",
    "wmparc_wm-rh-": "R wm ",
    "aseg_Left-": "L ",
    "aseg_Right-": "R ",
    "aseg_": "",
    "lh_": "L ",
    "rh_": "R ",
}


def load_cimlr_topk():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    best_epoch = int(checkpoint["epoch"])

    history = pd.read_csv(TOPK_HISTORY_PATH)
    row = history[history["epoch"] == best_epoch]

    if row.empty:
        raise ValueError(f"No top-k recorded for best epoch {best_epoch}")

    cimlr_topk = [int(x) for x in row.iloc[0]["feature_idx"].split("|")]
    return best_epoch, cimlr_topk, history


def build_stable_dim_table(stable_dims, train_imp, val_imp, feature_names):
    rows = []

    for dim in stable_dims:
        train_order = np.argsort(train_imp[dim])[::-1][:DISPLAY_TOP_N]
        val_order = np.argsort(val_imp[dim])[::-1][:DISPLAY_TOP_N]

        train_features = [f"{feature_names[i]} ({train_imp[dim, i]:.3f})" for i in train_order]
        val_features = [f"{feature_names[i]} ({val_imp[dim, i]:.3f})" for i in val_order]

        train_names = [feature_names[i] for i in train_order]
        val_names = [feature_names[i] for i in val_order]
        overlap = [name for name in train_names if name in val_names]

        rows.append({
            "dim": dim,
            "train_top_features": " | ".join(train_features),
            "val_top_features": " | ".join(val_features),
            "overlap_count": len(overlap),
            "overlap_features": " | ".join(overlap),
        })

    return pd.DataFrame(rows)


def build_feature_ranking(train_dci, val_dci, test_dci, feature_names, cimlr_topk):
    cimlr_set = set(cimlr_topk)
    train_r2 = train_dci["r2_scores"]
    val_r2 = val_dci["r2_scores"]
    test_r2 = test_dci["r2_scores"]

    df = pd.DataFrame({
        "feature_idx": np.arange(len(feature_names)),
        "feature_name": feature_names,
        "train_r2": train_r2,
        "val_r2": val_r2,
        "test_r2": test_r2,
        "mean_r2": (train_r2 + val_r2 + test_r2) / 3,
        "in_cimlr_topk": [i in cimlr_set for i in range(len(feature_names))],
    })

    return df.sort_values(["test_r2", "val_r2"], ascending=False).reset_index(drop=True)


def summarize_convergence(history):
    overlap = history["overlap"].to_numpy()
    top_k = int(history["top_k"].iloc[0])
    last_10 = overlap[-10:]

    return {
        "n_epochs": len(overlap),
        "top_k": top_k,
        "final_overlap": int(overlap[-1]),
        "last_10_mean": float(last_10.mean()),
        "last_10_min": int(last_10.min()),
    }


def shorten_feature_name(name):
    shortened = name

    for prefix, replacement in NAME_PREFIXES.items():
        if shortened.startswith(prefix):
            shortened = replacement + shortened[len(prefix):]
            break

    shortened = re.sub(r"_(volume|area|thickness|meancurv)$", "", shortened)
    shortened = shortened.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", shortened).strip()


def plot_topk_encoding(feature_df, cimlr_topk, out_path):
    ranking = feature_df.copy()
    selection_rank = {index: rank + 1 for rank, index in enumerate(cimlr_topk)}
    ranking["cimlr_rank"] = ranking["feature_idx"].map(selection_rank)

    selected = ranking[ranking["cimlr_rank"].notna()].copy()
    selected["cimlr_rank"] = selected["cimlr_rank"].astype(int)
    n_selected = len(selected)

    selected = selected.sort_values("cimlr_rank").head(PLOT_TOP_N)
    n_shown = len(selected)

    best_encoded = set(ranking.nlargest(n_shown, RANKING_COLUMN)["feature_name"])
    selected["also_best_encoded"] = selected["feature_name"].isin(best_encoded)
    selected["label"] = selected.apply(
        lambda row: f"{row['cimlr_rank']}.  {shorten_feature_name(row['feature_name'])}", axis=1)

    overlap = int(selected["also_best_encoded"].sum())
    pool_mean = ranking[RANKING_COLUMN].mean()

    plot_df = selected.iloc[::-1]

    height = max(4.5, 0.28 * n_shown + 1.8)
    fig, ax = plt.subplots(figsize=(9.5, height))

    colors = [SELECTED_COLOR if flag else UNSELECTED_COLOR for flag in plot_df["also_best_encoded"]]
    positions = np.arange(n_shown)
    ax.barh(positions, plot_df[RANKING_COLUMN], color=colors, height=0.72,
            edgecolor="white", linewidth=0.5)

    for position, value in enumerate(plot_df[RANKING_COLUMN]):
        ax.text(value + 0.006, position, f"{value:.2f}", va="center", fontsize=8, color="#444444")

    ax.axvline(pool_mean, color="#666666", linestyle="--", linewidth=1)
    ax.text(pool_mean, n_shown - 0.15, f"  pool mean {pool_mean:.2f}",
            fontsize=8, color="#666666", va="top")

    ax.set_yticks(positions)
    ax.set_yticklabels(plot_df["label"], fontsize=8)
    ax.set_xlabel("variance explained by the latent")
    ax.set_xlim(0, max(plot_df[RANKING_COLUMN].max(), pool_mean) * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    handles = [plt.Rectangle((0, 0), 1, 1, color=SELECTED_COLOR),
               plt.Rectangle((0, 0), 1, 1, color=UNSELECTED_COLOR)]
    ax.legend(handles, [f"also in the {n_shown} best encoded", "selected only"],
              frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    ax.set_title(
        f"Top {n_shown} features selected by CIMLR, in selection order\n"
        f"{overlap} of {n_shown} are also among the {n_shown} best-encoded features in the pool",
        fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "n_selected": n_selected,
        "n_shown": n_shown,
        "overlap": overlap,
        "pool_mean": pool_mean,
        "shown_mean": float(selected[RANKING_COLUMN].mean()),
    }


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    best_epoch, cimlr_topk, history = load_cimlr_topk()

    data = np.load(LATENT_PATH, allow_pickle=True)
    feature_names = data["feature_names"].astype(str)

    metric = MetricHandler(dci_alpha=DCI_ALPHA)

    train_dci = metric.compute_dci(data["train_mu"], data["train_features"])
    val_dci = metric.compute_dci(data["val_mu"], data["val_features"])
    test_dci = metric.compute_dci(data["test_mu"], data["test_features"])

    train_imp = train_dci["importance_matrix"]
    val_imp = val_dci["importance_matrix"]

    stable_1 = metric.get_stable_dims(
        train_imp, val_imp,
        threshold=STABLE_THRESHOLD, top_n=STABLE_TOP_N, min_overlap=1,
    )
    stable_2 = metric.get_stable_dims(
        train_imp, val_imp,
        threshold=STABLE_THRESHOLD, top_n=STABLE_TOP_N, min_overlap=2,
    )

    stable_df = build_stable_dim_table(stable_1, train_imp, val_imp, feature_names)
    feature_df = build_feature_ranking(train_dci, val_dci, test_dci, feature_names, cimlr_topk)
    convergence = summarize_convergence(history)

    stable_df.to_csv(OUT_DIR / "stable_dims.csv", index=False)
    feature_df.to_csv(OUT_DIR / "feature_ranking.csv", index=False)

    topk_plot_path = OUT_DIR / "topk_encoding.png"
    topk_stats = plot_topk_encoding(feature_df, cimlr_topk, topk_plot_path)

    top20 = feature_df.head(20)
    cimlr_in_top20 = top20[top20["in_cimlr_topk"]]
    cimlr_names = [feature_names[i] for i in cimlr_topk]

    summary = f"""best_epoch: {best_epoch}

    DCI:
    train_D: {train_dci["disentanglement"]:.4f}
    train_C: {train_dci["completeness"]:.4f}
    train_I: {train_dci["informativeness"]:.4f}
    val_D: {val_dci["disentanglement"]:.4f}
    val_C: {val_dci["completeness"]:.4f}
    val_I: {val_dci["informativeness"]:.4f}
    test_D: {test_dci["disentanglement"]:.4f}
    test_C: {test_dci["completeness"]:.4f}
    test_I: {test_dci["informativeness"]:.4f}

    Stable dimensions (top {STABLE_TOP_N}, threshold {STABLE_THRESHOLD}):
    min_overlap=1: {len(stable_1)} | {stable_1}
    min_overlap=2: {len(stable_2)} | {stable_2}

    CIMLR top-k convergence:
    epochs recorded: {convergence["n_epochs"]}
    final overlap: {convergence["final_overlap"]}/{convergence["top_k"]}
    last 10 epochs mean: {convergence["last_10_mean"]:.2f}/{convergence["top_k"]}
    last 10 epochs min: {convergence["last_10_min"]}/{convergence["top_k"]}

    CIMLR top-k at best epoch:
    {chr(10).join("    " + n for n in cimlr_names)}

    CIMLR features in top 20 encoded features: {len(cimlr_in_top20)}/{convergence["top_k"]}

    Top 20 encoded features:
    {top20[["feature_idx", "feature_name", "train_r2", "val_r2", "test_r2", "in_cimlr_topk"]].to_string(index=False)}
    """

    (OUT_DIR / "dci_summary.txt").write_text(summary)

    print(summary)
    print(f"\nSaved: {OUT_DIR / 'stable_dims.csv'}")
    print(f"Saved: {OUT_DIR / 'feature_ranking.csv'}")
    print(f"Saved: {OUT_DIR / 'dci_summary.txt'}")
    print(f"Saved: {topk_plot_path}")
    print(f"selected: {topk_stats['n_selected']}, shown: {topk_stats['n_shown']}")
    print(f"overlap: {topk_stats['overlap']}/{topk_stats['n_shown']}")
    print(f"pool mean {RANKING_COLUMN}: {topk_stats['pool_mean']:.4f}")
    print(f"shown mean {RANKING_COLUMN}: {topk_stats['shown_mean']:.4f}")