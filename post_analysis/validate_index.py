import os
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

RUN_DIR = os.environ.get("RUN_DIR")

if not RUN_DIR:
    raise ValueError("RUN_DIR environment variable is not set")

RUN_DIR = Path(RUN_DIR).resolve()

INDEX_PATH = RUN_DIR / "post_analysis" / "disease_index.csv"
DEMOGRAPHICS_PATH = PROJECT_ROOT / "data" / "all_demographics.csv"
OUTPUT_PATH = RUN_DIR / "post_analysis" / "index_validation.csv"

INDEX_COLUMNS = ["l2_norm", "cn_ad_projection", "off_axis_distance"]


def validate_index(df, index_name, split):
    subset = df.dropna(subset=[index_name, "mmse"])

    cn = subset.loc[subset["diagnosis"] == "cn", index_name]
    ad = subset.loc[subset["diagnosis"] == "ad", index_name]

    if len(subset) >= 2 and subset[index_name].nunique() > 1 and subset["mmse"].nunique() > 1:
        pearson_r, pearson_p = pearsonr(subset[index_name], subset["mmse"])
        spearman_r, spearman_p = spearmanr(subset[index_name], subset["mmse"])
    else:
        pearson_r = pearson_p = spearman_r = spearman_p = float("nan")

    return {
        "index_name": index_name,
        "split": split,
        "n": len(subset),
        "cn_mean": cn.mean(),
        "cn_std": cn.std(),
        "ad_mean": ad.mean(),
        "ad_std": ad.std(),
        "pearson_r_mmse": pearson_r,
        "pearson_p_mmse": pearson_p,
        "spearman_r_mmse": spearman_r,
        "spearman_p_mmse": spearman_p,
    }

def plot_distribution(df, index_name, output_path):
    subset = df.dropna(subset=[index_name])

    cn = subset.loc[subset["diagnosis"] == "cn", index_name]
    ad = subset.loc[subset["diagnosis"] == "ad", index_name]

    plt.figure(figsize=(8, 6))
    plt.hist(cn, bins=25, alpha=0.6, label="CN")
    plt.hist(ad, bins=25, alpha=0.6, label="AD")
    plt.xlabel(index_name)
    plt.ylabel("Count")
    plt.title(f"{index_name}: CN vs AD distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_index_vs_mmse(df, index_name, output_path):
    subset = df.dropna(subset=[index_name, "mmse"])

    cn = subset[subset["diagnosis"] == "cn"]
    ad = subset[subset["diagnosis"] == "ad"]

    if len(subset) >= 2 and subset[index_name].nunique() > 1 and subset["mmse"].nunique() > 1:
        spearman_r, spearman_p = spearmanr(subset[index_name], subset["mmse"])
    else:
        spearman_r = spearman_p = float("nan")
        
    plt.figure(figsize=(8, 6))
    plt.scatter(cn[index_name], cn["mmse"], alpha=0.7, label="CN")
    plt.scatter(ad[index_name], ad["mmse"], alpha=0.7, label="AD")
    plt.xlabel(index_name)
    plt.ylabel("MMSE")
    plt.title(f"{index_name} vs MMSE\nSpearman r={spearman_r:.3f}, p={spearman_p:.3g}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    index_df = pd.read_csv(INDEX_PATH)
    demographics = pd.read_csv(DEMOGRAPHICS_PATH)

    index_df["image_id"] = index_df["image_id"].astype(str)
    demographics["image_id"] = demographics["image_id"].astype(str)

    merged = index_df.merge(demographics[["image_id", "mmse"]], on="image_id", how="left")

    rows = []

    for split in ["train", "val", "test", "heldout"]:
        if split == "heldout":
            split_df = merged[merged["split"].isin(["val", "test"])].copy()
        else:
            split_df = merged[merged["split"] == split].copy()

        for index_name in INDEX_COLUMNS:
            rows.append(validate_index(split_df, index_name, split))

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(result.to_string(index=False))

    heldout_df = merged[merged["split"].isin(["val", "test"])].copy()

    for index_name in INDEX_COLUMNS:
        plot_distribution(heldout_df, index_name, RUN_DIR / "post_analysis" / f"{index_name}_distribution.png")
        plot_index_vs_mmse(heldout_df, index_name, RUN_DIR / "post_analysis" / f"{index_name}_vs_mmse.png")