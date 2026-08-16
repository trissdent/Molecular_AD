import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
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

INDEX_COLUMNS = ["cn_ad_projection", "l2_norm", "off_axis_distance"]
PRIMARY_INDEX = "cn_ad_projection"
PRIMARY_SPLIT = "test"
TARGETS = ["mmse", "ptau217_alzpath", "ptau217_janssen"]
N_BOOTSTRAP = 1000
RNG = np.random.default_rng(42)


def bootstrap_ci(values, labels, fn):
    n = len(values)
    if n < 10:
        return float("nan"), float("nan")

    stats = []
    for _ in range(N_BOOTSTRAP):
        idx = RNG.integers(0, n, n)
        try:
            stats.append(fn(values[idx], labels[idx]))
        except ValueError:
            continue

    if not stats:
        return float("nan"), float("nan")
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def safe_auc(index_values, is_ad):
    if len(np.unique(is_ad)) < 2:
        return float("nan")
    return roc_auc_score(is_ad, index_values)


def safe_spearman(x, y):
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    r, p = spearmanr(x, y)
    return float(r), float(p)


def cohens_d(cn, ad):
    if len(cn) < 2 or len(ad) < 2:
        return float("nan")
    pooled_var = ((len(cn) - 1) * cn.var(ddof=1) + (len(ad) - 1) * ad.var(ddof=1)) / (len(cn) + len(ad) - 2)
    if pooled_var <= 0:
        return float("nan")
    return float((ad.mean() - cn.mean()) / np.sqrt(pooled_var))


def correlate_target(df, index_name, target):
    if target not in df.columns:
        return {}

    paired = df.dropna(subset=[index_name, target])
    out = {f"n_{target}": len(paired)}

    r, p = safe_spearman(paired[index_name].to_numpy(), paired[target].to_numpy())
    out[f"spearman_r_{target}"] = r
    out[f"spearman_p_{target}"] = p

    lo, hi = bootstrap_ci(
        paired[index_name].to_numpy(),
        paired[target].to_numpy(),
        lambda a, b: safe_spearman(a, b)[0],
    )
    out[f"spearman_ci_lo_{target}"] = lo
    out[f"spearman_ci_hi_{target}"] = hi

    for group in ["ad", "cn"]:
        sub = paired[paired["diagnosis"] == group]
        gr, gp = safe_spearman(sub[index_name].to_numpy(), sub[target].to_numpy())
        out[f"n_{group}_{target}"] = len(sub)
        out[f"spearman_r_{target}_{group}_only"] = gr
        out[f"spearman_p_{target}_{group}_only"] = gp

    return out


def validate_index(df, index_name, split):
    subset = df.dropna(subset=[index_name])

    cn = subset.loc[subset["diagnosis"] == "cn", index_name]
    ad = subset.loc[subset["diagnosis"] == "ad", index_name]

    is_ad = (subset["diagnosis"] == "ad").to_numpy().astype(int)
    values = subset[index_name].to_numpy()

    auc = safe_auc(values, is_ad)
    auc_lo, auc_hi = bootstrap_ci(values, is_ad, safe_auc)

    if len(cn) >= 3 and len(ad) >= 3:
        _, mwu_p = mannwhitneyu(ad, cn, alternative="two-sided")
    else:
        mwu_p = float("nan")

    row = {
        "index_name": index_name,
        "split": split,
        "primary": index_name == PRIMARY_INDEX and split == PRIMARY_SPLIT,
        "n": len(subset),
        "n_cn": len(cn),
        "n_ad": len(ad),
        "cn_mean": cn.mean(),
        "cn_std": cn.std(),
        "ad_mean": ad.mean(),
        "ad_std": ad.std(),
        "cohens_d": cohens_d(cn, ad),
        "auc": auc,
        "auc_ci_lo": auc_lo,
        "auc_ci_hi": auc_hi,
        "mannwhitney_p": mwu_p,
    }

    for target in TARGETS:
        row.update(correlate_target(subset, index_name, target))

    return row


def plot_distribution(df, index_name, output_path):
    subset = df.dropna(subset=[index_name])

    cn = subset.loc[subset["diagnosis"] == "cn", index_name]
    ad = subset.loc[subset["diagnosis"] == "ad", index_name]
    auc = safe_auc(subset[index_name].to_numpy(), (subset["diagnosis"] == "ad").to_numpy().astype(int))

    plt.figure(figsize=(8, 6))
    plt.hist(cn, bins=25, alpha=0.6, label=f"CN (n={len(cn)})")
    plt.hist(ad, bins=25, alpha=0.6, label=f"AD (n={len(ad)})")
    plt.xlabel(index_name)
    plt.ylabel("Count")
    plt.title(f"{index_name}: CN vs AD distribution\nAUC={auc:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_index_vs_target(df, index_name, target, output_path):
    subset = df.dropna(subset=[index_name, target])

    if len(subset) < 3:
        return

    cn = subset[subset["diagnosis"] == "cn"]
    ad = subset[subset["diagnosis"] == "ad"]

    r, p = safe_spearman(subset[index_name].to_numpy(), subset[target].to_numpy())
    ad_r, ad_p = safe_spearman(ad[index_name].to_numpy(), ad[target].to_numpy())

    plt.figure(figsize=(8, 6))
    plt.scatter(cn[index_name], cn[target], alpha=0.7, label=f"CN (n={len(cn)})")
    plt.scatter(ad[index_name], ad[target], alpha=0.7, label=f"AD (n={len(ad)})")
    plt.xlabel(index_name)
    plt.ylabel(target)
    plt.title(
        f"{index_name} vs {target}\n"
        f"all: r={r:.3f}, p={p:.3g} (n={len(subset)}) | "
        f"AD only: r={ad_r:.3f}, p={ad_p:.3g} (n={len(ad)})"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    index_df = pd.read_csv(INDEX_PATH)
    demographics = pd.read_csv(DEMOGRAPHICS_PATH, low_memory=False)

    index_df["image_id"] = index_df["image_id"].astype(str)
    demographics["image_id"] = demographics["image_id"].astype(str)

    keep = ["image_id"] + [t for t in TARGETS if t in demographics.columns]
    merged = index_df.merge(demographics[keep], on="image_id", how="left")
    merged["diagnosis"] = merged["diagnosis"].str.lower()

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

    test_df = merged[merged["split"] == "test"].copy()
    heldout_df = merged[merged["split"].isin(["val", "test"])].copy()

    for index_name in INDEX_COLUMNS:
        plot_distribution(heldout_df, index_name, RUN_DIR / "post_analysis" / f"{index_name}_distribution.png")
        plot_index_vs_target(heldout_df, index_name, "mmse", RUN_DIR / "post_analysis" / f"{index_name}_vs_mmse.png")

    for target in ["ptau217_alzpath", "ptau217_janssen"]:
        plot_index_vs_target(test_df, PRIMARY_INDEX, target, RUN_DIR / "post_analysis" / f"{PRIMARY_INDEX}_vs_{target}.png")

    primary = result[result["primary"]].iloc[0]

    print(f"Saved: {OUTPUT_PATH}")
    print(f"\nPRIMARY: {PRIMARY_INDEX} on {PRIMARY_SPLIT}")
    print(f"  n={int(primary['n'])} (CN={int(primary['n_cn'])}, AD={int(primary['n_ad'])})")
    print(f"  AUC            {primary['auc']:.3f}  [{primary['auc_ci_lo']:.3f}, {primary['auc_ci_hi']:.3f}]")
    print(f"  Cohen's d      {primary['cohens_d']:.3f}")
    print(f"  Mann-Whitney p {primary['mannwhitney_p']:.3g}")

    for target in TARGETS:
        if f"spearman_r_{target}" not in primary:
            continue
        print(f"\n  {target}  (n={int(primary[f'n_{target}'])})")
        print(f"    all      r={primary[f'spearman_r_{target}']:.3f}  "
              f"[{primary[f'spearman_ci_lo_{target}']:.3f}, {primary[f'spearman_ci_hi_{target}']:.3f}]  "
              f"p={primary[f'spearman_p_{target}']:.3g}")
        print(f"    AD only  r={primary[f'spearman_r_{target}_ad_only']:.3f}  "
              f"p={primary[f'spearman_p_{target}_ad_only']:.3g}  (n={int(primary[f'n_ad_{target}'])})")
        print(f"    CN only  r={primary[f'spearman_r_{target}_cn_only']:.3f}  "
              f"p={primary[f'spearman_p_{target}_cn_only']:.3g}  (n={int(primary[f'n_cn_{target}'])})")

    print("\nAll splits (exploratory):")
    cols = ["index_name", "split", "n", "auc", "cohens_d", "spearman_r_mmse", "spearman_r_mmse_ad_only"]
    if "spearman_r_ptau217_alzpath" in result.columns:
        cols += ["n_ptau217_alzpath", "spearman_r_ptau217_alzpath", "spearman_r_ptau217_alzpath_ad_only"]
    print(result[cols].to_string(index=False))