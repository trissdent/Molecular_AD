import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd().parent
SRC_ROOT = Path.cwd()

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from src.shared.models.metrics import MetricHandler


LATENT_PATH = "../output/analyze_dci/latents_best_ckpt.npz"
OUT_DIR = "../output/analyze_dci"

DCI_ALPHA = 0.1
STABLE_THRESHOLD = 0.1
TOP_N_FEATURES_PER_DIM = 10


CIMLR_TOP20 = [
    100, 101, 106, 285, 286, 293, 303, 304, 343, 344,
    351, 352, 354, 359, 362, 364, 369, 370, 372, 377,
]


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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    data = np.load(LATENT_PATH, allow_pickle=True)

    train_mu = data["train_mu"]
    val_mu = data["val_mu"]
    train_features = data["train_features"]
    val_features = data["val_features"]
    feature_names = data["feature_names"].astype(str)

    metric = MetricHandler(dci_alpha=DCI_ALPHA)

    train_dci = metric.compute_dci(train_mu, train_features)
    val_dci = metric.compute_dci(val_mu, val_features)

    train_imp = train_dci["importance_matrix"]
    val_imp = val_dci["importance_matrix"]

    stable_dims = metric.get_stable_dims(
        train_imp,
        val_imp,
        threshold=STABLE_THRESHOLD,
    )

    cimlr_set = set(CIMLR_TOP20)

    # importance matrix
    pd.DataFrame(
        train_imp,
        index=[f"z{i}" for i in range(train_imp.shape[0])],
        columns=feature_names,
    ).to_csv(os.path.join(OUT_DIR, "dci_importance_matrix_train.csv"))

    # stable dim summary
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

    pd.DataFrame(rows).to_csv(
        os.path.join(OUT_DIR, "stable_dim_summary.csv"),
        index=False,
    )

    # stable dim -> top features
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
                "in_cimlr_top20": int(feat_idx in cimlr_set),
            })

    stable_top_df = pd.DataFrame(rows)
    stable_top_df.to_csv(
        os.path.join(OUT_DIR, "stable_dim_top_features.csv"),
        index=False,
    )

    # global feature importance from stable dims
    if len(stable_dims) > 0:
        stable_feature_importance = train_imp[stable_dims].sum(axis=0)
    else:
        stable_feature_importance = np.zeros(train_imp.shape[1])

    global_df = pd.DataFrame({
        "feature_idx": np.arange(len(feature_names)),
        "feature_name": feature_names,
        "feature_type": [feature_type(x) for x in feature_names],
        "importance_from_stable_dims": stable_feature_importance,
        "in_cimlr_top20": [i in cimlr_set for i in range(len(feature_names))],
    })

    global_df = global_df.sort_values(
        "importance_from_stable_dims",
        ascending=False,
    )

    global_df.to_csv(
        os.path.join(OUT_DIR, "global_feature_importance_from_stable_dims.csv"),
        index=False,
    )

    # overlap
    stable_top20 = set(global_df.head(20)["feature_idx"].astype(int).tolist())
    overlap = sorted(list(stable_top20 & cimlr_set))

    total_importance = global_df["importance_from_stable_dims"].sum()
    cimlr_importance = global_df.loc[
        global_df["in_cimlr_top20"],
        "importance_from_stable_dims",
    ].sum()

    weighted_overlap = (
        cimlr_importance / total_importance
        if total_importance > 0 else 0.0
    )

    # feature type share
    type_df = (
        global_df.groupby("feature_type")["importance_from_stable_dims"]
        .sum()
        .reset_index()
        .sort_values("importance_from_stable_dims", ascending=False)
    )
    type_df["share"] = type_df["importance_from_stable_dims"] / type_df["importance_from_stable_dims"].sum()
    type_df.to_csv(os.path.join(OUT_DIR, "feature_type_share.csv"), index=False)

    # summary
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write("=== DCI ===\n")
        f.write(f"train_D: {train_dci['disentanglement']:.4f}\n")
        f.write(f"train_C: {train_dci['completeness']:.4f}\n")
        f.write(f"train_I: {train_dci['informativeness']:.4f}\n")
        f.write(f"val_D:   {val_dci['disentanglement']:.4f}\n")
        f.write(f"val_C:   {val_dci['completeness']:.4f}\n")
        f.write(f"val_I:   {val_dci['informativeness']:.4f}\n\n")

        f.write("=== Stable dims ===\n")
        f.write(f"count: {len(stable_dims)}\n")
        f.write(f"dims: {stable_dims}\n\n")

        f.write("=== Overlap ===\n")
        f.write(f"overlap@20: {len(overlap)}/20\n")
        f.write(f"overlap_features: {overlap}\n")
        f.write(f"weighted_overlap: {weighted_overlap:.4f}\n\n")

        f.write("=== Top stable-dim features ===\n")
        for _, row in global_df.head(20).iterrows():
            f.write(
                f"{int(row['feature_idx'])}: {row['feature_name']} | "
                f"{row['feature_type']} | "
                f"{row['importance_from_stable_dims']:.6f} | "
                f"in_cimlr={row['in_cimlr_top20']}\n"
            )

        f.write("\n=== Feature type share ===\n")
        for _, row in type_df.iterrows():
            f.write(f"{row['feature_type']}: {row['share']:.4f}\n")

    print("Done.")
    print(f"Stable dims: {len(stable_dims)}")
    print(f"Overlap@20: {len(overlap)}/20")
    print(f"Weighted overlap: {weighted_overlap:.4f}")
    print(f"Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()