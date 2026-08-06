import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from src.configs.config import ConfigReader
from src.shared.models.metrics import MetricHandler

RUN_DIR = os.environ.get("RUN_DIR")
if not RUN_DIR:
    raise ValueError("RUN_DIR environment variable is not set")

RUN_DIR = Path(RUN_DIR).resolve()
RUN_NAME = RUN_DIR.name

CHECKPOINT_PATH = RUN_DIR / "best.ckpt"
LATENT_PATH = RUN_DIR / "post_analysis" / "latents_best_ckpt.npz"
OUT_DIR = RUN_DIR / "post_analysis"
LOG_PATH = PROJECT_ROOT / "logs" / RUN_NAME / "log.txt"
CONFIG_PATH = PROJECT_ROOT / "logs" / RUN_NAME / "config.yml"

DCI_ALPHA = 0.1
STABLE_THRESHOLD = 0.1
STABLE_TOP_N = 3
DISPLAY_TOP_N = 5

config = ConfigReader.merge(CONFIG_PATH)
CIMLR_TOP_K = config.training.top_k


def load_cimlr_topk():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    best_epoch = int(checkpoint["epoch"])
    log_text = LOG_PATH.read_text()

    pattern = rf"\[epoch {best_epoch}\] top{CIMLR_TOP_K} overlap=\d+/{CIMLR_TOP_K} \| \[([^\]]+)\]"
    match = re.search(pattern, log_text)

    if not match:
        raise ValueError(f"Could not find CIMLR top{CIMLR_TOP_K} for best epoch {best_epoch}")

    cimlr_topk = [int(x.strip()) for x in match.group(1).split(",")]
    return best_epoch, cimlr_topk


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


def build_feature_ranking(train_dci, val_dci, feature_names, cimlr_topk):
    cimlr_set = set(cimlr_topk)
    train_r2 = train_dci["r2_scores"]
    val_r2 = val_dci["r2_scores"]

    df = pd.DataFrame({
        "feature_idx": np.arange(len(feature_names)),
        "feature_name": feature_names,
        "train_r2": train_r2,
        "val_r2": val_r2,
        "mean_r2": (train_r2 + val_r2) / 2,
        "in_cimlr_topk": [i in cimlr_set for i in range(len(feature_names))],
    })

    return df.sort_values(["val_r2", "train_r2"], ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    best_epoch, cimlr_topk = load_cimlr_topk()

    data = np.load(LATENT_PATH, allow_pickle=True)
    feature_names = data["feature_names"].astype(str)

    metric = MetricHandler(dci_alpha=DCI_ALPHA)

    train_dci = metric.compute_dci(data["train_mu"], data["train_features"])
    val_dci = metric.compute_dci(data["val_mu"], data["val_features"])

    train_imp = train_dci["importance_matrix"]
    val_imp = val_dci["importance_matrix"]

    stable_dims = metric.get_stable_dims(
        train_imp,
        val_imp,
        threshold=STABLE_THRESHOLD,
        top_n=STABLE_TOP_N,
        min_overlap=1,
    )

    stable_df = build_stable_dim_table(stable_dims, train_imp, val_imp, feature_names)
    feature_df = build_feature_ranking(train_dci, val_dci, feature_names, cimlr_topk)

    stable_df.to_csv(OUT_DIR / "stable_dims.csv", index=False)
    feature_df.to_csv(OUT_DIR / "feature_ranking.csv", index=False)

    top20 = feature_df.head(20)
    cimlr_in_top20 = top20[top20["in_cimlr_topk"]]

    summary = f"""best_epoch: {best_epoch}

    DCI:
    train_D: {train_dci["disentanglement"]:.4f}
    train_C: {train_dci["completeness"]:.4f}
    train_I: {train_dci["informativeness"]:.4f}
    val_D: {val_dci["disentanglement"]:.4f}
    val_C: {val_dci["completeness"]:.4f}
    val_I: {val_dci["informativeness"]:.4f}

    Stable dimensions:
    count: {len(stable_dims)}
    dims: {stable_dims}

    CIMLR:
    top_k: {cimlr_topk}
    CIMLR features in top 20 encoded features: {len(cimlr_in_top20)}/{CIMLR_TOP_K}

    Top 20 encoded features:
    {top20[["feature_idx", "feature_name", "train_r2", "val_r2", "in_cimlr_topk"]].to_string(index=False)}
    """

    (OUT_DIR / "dci_summary.txt").write_text(summary)

    print(summary)
    print(f"\nSaved: {OUT_DIR / 'stable_dims.csv'}")
    print(f"Saved: {OUT_DIR / 'feature_ranking.csv'}")
    print(f"Saved: {OUT_DIR / 'dci_summary.txt'}")