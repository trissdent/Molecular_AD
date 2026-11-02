"""s2: CN-AD disease index from the CALIBER latents (mu), validated on every split.

Index construction and statistics live in index_core.py, shared with baselines.py.
Outputs in <run_dir>/s2_index/: disease_index.csv, index_validation.csv, plots.
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from index_core import (
    AGECORR_INDEX,
    RAW_INDEX,
    SPLITS,
    TARGETS,
    add_trend_line,
    build_index_table,
    drop_excluded,
    paired_subset,
    primary_summary,
    safe_auc,
    safe_spearman,
    validate_all,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RUN_DIR = os.environ.get("RUN_DIR")
if not RUN_DIR:
    raise ValueError("RUN_DIR environment variable is not set")
RUN_DIR = Path(RUN_DIR).resolve()

LATENT_PATH = RUN_DIR / "latents_best_ckpt.npz"
RUN_CONFIG = yaml.safe_load(open(PROJECT_ROOT / "logs" / RUN_DIR.name / "config.yml"))
COHORT_PATH = Path(RUN_CONFIG["data"]["prepared_dir"]) / "cohort.csv"
OUT_DIR = RUN_DIR / "s2_index"
INDEX_PATH = OUT_DIR / "disease_index.csv"
VALIDATION_PATH = OUT_DIR / "index_validation.csv"

CN_COLOR = "#3B7EA1"
AD_COLOR = "#E08A2E"


def latents_by_split(data):
    """CALIBER latents (mu) and visit info per split, from the extract_latents output."""
    reps, meta = {}, {}
    for split in SPLITS:
        reps[split] = data[f"{split}_mu"].astype(float)
        meta[split] = pd.DataFrame({
            "visit_id": data[f"{split}_visit_ids"].astype(str),
            "diagnosis": np.char.lower(data[f"{split}_diagnoses"].astype(str)),
            "age": data[f"{split}_age"].astype(float),
        })
    return reps, meta


def plot_distribution(df, index_name, output_path):
    subset = df.dropna(subset=[index_name])

    cn = subset.loc[subset["diagnosis"] == "cn", index_name]
    ad = subset.loc[subset["diagnosis"] == "ad", index_name]
    auc = safe_auc(subset[index_name].to_numpy(), (subset["diagnosis"] == "ad").to_numpy().astype(int))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(cn, bins=25, alpha=0.7, color=CN_COLOR, label=f"CN (n={len(cn)})")
    ax.hist(ad, bins=25, alpha=0.7, color=AD_COLOR, label=f"AD (n={len(ad)})")
    ax.set_xlabel(index_name)
    ax.set_ylabel("count")
    ax.set_title(f"{index_name}: CN vs AD distribution\nAUC={auc:.3f}", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def scatter_panel(ax, subset, index_name, target, ylabel):
    ad = subset[subset["diagnosis"] == "ad"]
    cn = subset[subset["diagnosis"] == "cn"]

    r_all, p_all = safe_spearman(subset[index_name].to_numpy(), subset[target].to_numpy())
    r_ad, p_ad = safe_spearman(ad[index_name].to_numpy(), ad[target].to_numpy())

    for group, color, label in [(cn, CN_COLOR, "CN"), (ad, AD_COLOR, "AD")]:
        ax.scatter(group[index_name], group[target], s=42, alpha=0.75, color=color,
                   edgecolor="white", linewidth=0.6, label=f"{label} (n={len(group)})")

    add_trend_line(ax, ad[index_name], ad[target], target, AD_COLOR)

    ax.set_xlabel("disease index")
    ax.set_ylabel(ylabel)
    ax.set_title(f"all r={r_all:.3f} (p={p_all:.2g})   |   "
                 f"AD only r={r_ad:.3f} (p={p_ad:.2g}, n={len(ad)})", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)


def plot_index_vs_target(df, target, ylabel, output_path):
    raw_subset = paired_subset(df, RAW_INDEX, target)
    corrected_subset = paired_subset(df, AGECORR_INDEX, target)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    scatter_panel(axes[0], raw_subset, RAW_INDEX, target, ylabel)
    scatter_panel(axes[1], corrected_subset, AGECORR_INDEX, target, ylabel)
    axes[0].set_title("uncorrected\n" + axes[0].get_title(), fontsize=10)
    axes[1].set_title("age-corrected\n" + axes[1].get_title(), fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_index_vs_age(df, output_path):
    subset = df.dropna(subset=["age", RAW_INDEX, AGECORR_INDEX])

    r_raw, _ = safe_spearman(subset["age"].to_numpy(), subset[RAW_INDEX].to_numpy())
    r_corrected, _ = safe_spearman(subset["age"].to_numpy(), subset[AGECORR_INDEX].to_numpy())

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for group, color, label in [("cn", CN_COLOR, "CN"), ("ad", AD_COLOR, "AD")]:
        group_df = subset[subset["diagnosis"] == group]
        ax.scatter(group_df["age"], group_df[AGECORR_INDEX], s=42, alpha=0.75, color=color,
                   edgecolor="white", linewidth=0.6, label=f"{label} (n={len(group_df)})")

    ax.set_xlabel("age")
    ax.set_ylabel("age-corrected index")
    ax.set_title(f"index vs age: {r_raw:.3f} before correction, {r_corrected:.3f} after", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = np.load(LATENT_PATH, allow_pickle=True)
    index_df = build_index_table(*latents_by_split(data))

    cohort = pd.read_csv(COHORT_PATH, dtype={"visit_id": str, "adni_id": str})
    keep = ["visit_id", "adni_id"] + [t for t in TARGETS if t in cohort.columns]
    merged = drop_excluded(index_df.merge(cohort[keep], on="visit_id", how="left"))
    merged.to_csv(INDEX_PATH, index=False)

    validation = validate_all(merged)
    validation.to_csv(VALIDATION_PATH, index=False)

    test_df = merged[merged["split"] == "test"].copy()
    plot_distribution(test_df, RAW_INDEX, OUT_DIR / f"{RAW_INDEX}_distribution.png")
    plot_distribution(test_df, AGECORR_INDEX, OUT_DIR / f"{AGECORR_INDEX}_distribution.png")
    for target in TARGETS:
        if target in test_df.columns:
            ylabel = "MMSE" if target == "mmse" else target.replace("ptau217_", "pTau-217 ")
            plot_index_vs_target(test_df, target, ylabel, OUT_DIR / f"agecorr_index_vs_{target}.png")
    plot_index_vs_age(test_df, OUT_DIR / "agecorr_index_vs_age.png")

    print(primary_summary(validation))
    print(f"Saved: {INDEX_PATH}")
    print(f"Saved: {VALIDATION_PATH}")