import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from src.configs.config import ConfigReader  # noqa: E402
from src.shared.models.metrics import MetricHandler  # noqa: E402


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
TOP_N_FEATURES_PER_DIM = 10
config = ConfigReader.merge(CONFIG_PATH)
CIMLR_TOP_K = config.training.top_k


def feature_type(name):
    name = name.lower()

    if "meanintensity" in name:
        return "meanintensity"
    if "surfaceholes" in name:
        return "surfaceholes"
    if "thickness" in name:
        return "thickness"
    if "meancurv" in name:
        return "meancurv"
    if "area" in name:
        return "area"
    if "volume" in name or "vol" in name:
        return "volume"
    if "etiv" in name:
        return "global_ratio"

    return "other"

def plot_stable_dim_heatmap(stable_dims, train_imp, feature_names, output_path, top_n=CIMLR_TOP_K):
    if len(stable_dims) == 0:
        return

    stable_imp = train_imp[stable_dims]
    global_importance = stable_imp.sum(axis=0)
    top_indices = np.argsort(global_importance)[::-1][:top_n]

    plot_data = stable_imp[:, top_indices]
    plot_features = feature_names[top_indices]

    plt.figure(figsize=(14, max(4, len(stable_dims) * 0.6)))
    plt.imshow(plot_data, aspect="auto")
    plt.colorbar(label="DCI importance")
    plt.xticks(range(len(plot_features)), plot_features, rotation=90)
    plt.yticks(range(len(stable_dims)), [f"Dim {dim}" for dim in stable_dims])
    plt.xlabel("FreeSurfer features")
    plt.ylabel("Stable latent dimensions")
    plt.title("Stable latent dimensions and anatomical feature importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_global_feature_importance(stable_dims, train_imp, feature_names, output_path, top_n=CIMLR_TOP_K):
    if len(stable_dims) == 0:
        return

    global_importance = train_imp[stable_dims].sum(axis=0)
    top_indices = np.argsort(global_importance)[::-1][:top_n][::-1]

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_indices)), global_importance[top_indices])
    plt.yticks(range(len(top_indices)), feature_names[top_indices])
    plt.xlabel("Total DCI importance across stable dimensions")
    plt.title(f"Top {top_n} anatomical features from stable latent dimensions")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":

    os.makedirs(OUT_DIR, exist_ok=True)

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    best_epoch = int(ckpt["epoch"])

    with open(LOG_PATH, "r") as f:
        log_text = f.read()

    pattern = rf"\[epoch {best_epoch}\] top{CIMLR_TOP_K} overlap=\d+/{CIMLR_TOP_K} \| \[([^\]]+)\]"
    match = re.search(pattern, log_text)

    if not match:
        raise ValueError(f"Could not find CIMLR top{CIMLR_TOP_K} for best epoch {best_epoch}")

    cimlr_topk = [int(x.strip()) for x in match.group(1).split(",")]

    data = np.load(LATENT_PATH, allow_pickle=True)
    
    feature_names = data["feature_names"].astype(str)
    metric = MetricHandler(dci_alpha=DCI_ALPHA)

    train_dci = metric.compute_dci(data["train_mu"], data["train_features"])
    val_dci = metric.compute_dci(data["val_mu"], data["val_features"])

    train_imp = train_dci["importance_matrix"]
    val_imp = val_dci["importance_matrix"]

    stable_dims = metric.get_stable_dims(train_imp, val_imp, threshold=STABLE_THRESHOLD)

    cimlr_set = set(cimlr_topk)

    pd.DataFrame(train_imp, index=[f"z{i}" for i in range(train_imp.shape[0])], columns=feature_names).to_csv(OUT_DIR / "dci_importance_matrix_train.csv")
    pd.DataFrame(val_imp, index=[f"z{i}" for i in range(val_imp.shape[0])], columns=feature_names,).to_csv(OUT_DIR / "dci_importance_matrix_val.csv")

    rows = []

    for dim in stable_dims:
        train_top = int(np.argmax(train_imp[dim]))
        val_top = int(np.argmax(val_imp[dim]))

        rows.append({
            "dim": dim,
            "train_top_feature_idx": train_top,
            "train_top_feature": feature_names[train_top],
            "val_top_feature_idx": val_top,
            "val_top_feature": feature_names[val_top],
            "train_top_importance": float(train_imp[dim, train_top]),
            "val_top_importance": float(val_imp[dim, val_top]),
        })

    pd.DataFrame(rows).to_csv(OUT_DIR / "stable_dim_summary.csv",index=False,)

    rows = []

    for dim in stable_dims:
        order = np.argsort(train_imp[dim])[::-1][:TOP_N_FEATURES_PER_DIM]

        for rank, feat_idx in enumerate(order, start=1):
            rows.append({
                "dim": dim,
                "rank": rank,
                "feature_idx": int(feat_idx),
                "feature_name": feature_names[feat_idx],
                "feature_type": feature_type(feature_names[feat_idx]),
                "importance": float(train_imp[dim, feat_idx]),
                "in_cimlr_topk": int(feat_idx in cimlr_set),
            })

    stable_top_df = pd.DataFrame(rows)

    stable_top_df.to_csv(OUT_DIR / "stable_dim_top_features.csv", index=False)

    if len(stable_dims) > 0:
        stable_feature_importance = train_imp[stable_dims].sum(axis=0)
    else:
        stable_feature_importance = np.zeros(train_imp.shape[1])

    global_df = pd.DataFrame({
        "feature_idx": np.arange(len(feature_names)),
        "feature_name": feature_names,
        "feature_type": [feature_type(x) for x in feature_names],
        "importance_from_stable_dims": (stable_feature_importance),
        "in_cimlr_topk": [i in cimlr_set for i in range(len(feature_names))],
    })

    global_df = global_df.sort_values("importance_from_stable_dims", ascending=False)

    global_df.to_csv(OUT_DIR / "global_feature_importance_from_stable_dims.csv", index=False)

    stable_topk = set(global_df.head(CIMLR_TOP_K)["feature_idx"].astype(int).tolist())

    overlap = sorted(stable_topk & cimlr_set)

    total_importance = global_df["importance_from_stable_dims"].sum()

    cimlr_importance = global_df.loc[global_df["in_cimlr_topk"], "importance_from_stable_dims"].sum()

    weighted_overlap = cimlr_importance / total_importance if total_importance > 0 else 0.0

    type_df = (global_df.groupby("feature_type")["importance_from_stable_dims"].sum().reset_index().sort_values("importance_from_stable_dims", ascending=False))

    total_type_importance = type_df["importance_from_stable_dims"].sum()

    type_df["share"] = type_df["importance_from_stable_dims"]/ total_type_importance if total_type_importance > 0 else 0.0

    type_df.to_csv(OUT_DIR / "feature_type_share.csv", index=False,)

    summary = f"""best_epoch: {best_epoch}
    cimlr_topk: {cimlr_topk}

    === DCI ===
    train_D: {train_dci['disentanglement']:.4f}
    train_C: {train_dci['completeness']:.4f}
    train_I: {train_dci['informativeness']:.4f}
    val_D: {val_dci['disentanglement']:.4f}
    val_C: {val_dci['completeness']:.4f}
    val_I: {val_dci['informativeness']:.4f}

    === Stable dims ===
    count: {len(stable_dims)}
    dims: {stable_dims}

    === Overlap ===
    overlap@{CIMLR_TOP_K}: {len(overlap)}/{CIMLR_TOP_K}
    overlap_features: {overlap}
    weighted_overlap: {weighted_overlap:.4f}
    """

    (OUT_DIR / "summary.txt").write_text(summary)

    print(f"Best epoch: {best_epoch}")
    print(f"CIMLR top{CIMLR_TOP_K}: {cimlr_topk}")
    print(f"Stable dims: {len(stable_dims)}")
    print(f"Overlap@{CIMLR_TOP_K}: {len(overlap)}/{CIMLR_TOP_K}")
    print(f"Weighted overlap: {weighted_overlap:.4f}")
    print(f"Saved to: {OUT_DIR}")

    plot_stable_dim_heatmap(stable_dims, train_imp, feature_names, OUT_DIR / "stable_dim_importance_heatmap.png")
    plot_global_feature_importance(stable_dims, train_imp, feature_names, OUT_DIR / "global_feature_importance.png")