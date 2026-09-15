import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RUN_DIR = os.environ.get("RUN_DIR")

if not RUN_DIR:
    raise ValueError("RUN_DIR environment variable is not set")

RUN_DIR = Path(RUN_DIR).resolve()

LATENT_PATH = RUN_DIR / "latents_best_ckpt.npz"
DEMOGRAPHICS_PATH = PROJECT_ROOT / "data" / "all_demographics.csv"
OUT_DIR = RUN_DIR / "s2_index"
INDEX_PATH = OUT_DIR / "disease_index.csv"
VALIDATION_PATH = OUT_DIR / "index_validation.csv"

SPLITS = ["train", "val", "test"]
RAW_INDEX = "cn_ad_projection"
AGECORR_INDEX = "cn_ad_projection_agecorr"
INDEX_COLUMNS = [RAW_INDEX, AGECORR_INDEX]
PRIMARY_INDEX = AGECORR_INDEX
PRIMARY_SPLIT = "test"

PTAU_TARGETS = ["ptau217_alzpath", "ptau217_janssen"]
TARGETS = ["mmse"] + PTAU_TARGETS
N_OUTLIER_DROP = 1

N_BOOTSTRAP = 1000
RNG = np.random.default_rng(42)

CN_COLOR = "#3B7EA1"
AD_COLOR = "#E08A2E"


def fit_aging_model(train_mu, train_age, train_diagnoses):
    mask = (train_diagnoses == "cn") & np.isfinite(train_age)
    design = np.column_stack([np.ones(mask.sum()), train_age[mask]])
    coefficients, *_ = np.linalg.lstsq(design, train_mu[mask], rcond=None)
    return coefficients


def remove_aging(mu, age, coefficients):
    corrected = mu.copy()
    finite = np.isfinite(age)

    if finite.any():
        design = np.column_stack([np.ones(finite.sum()), age[finite]])
        corrected[finite] = mu[finite] - design @ coefficients

    return corrected


def group_means(train_mu, train_diagnoses):
    return (train_mu[train_diagnoses == "cn"].mean(axis=0),
            train_mu[train_diagnoses == "ad"].mean(axis=0))


def compute_projection(mu, cn_mean, ad_mean):
    direction = ad_mean - cn_mean
    return ((mu - cn_mean) @ direction) / (direction @ direction)


def build_index_table(data):
    train_mu = data["train_mu"].astype(float)
    train_age = data["train_age"].astype(float)
    train_diagnoses = np.char.lower(data["train_diagnoses"].astype(str))

    coefficients = fit_aging_model(train_mu, train_age, train_diagnoses)

    raw_mu = {split: data[f"{split}_mu"].astype(float) for split in SPLITS}
    corrected_mu = {split: remove_aging(raw_mu[split], data[f"{split}_age"].astype(float), coefficients)
                    for split in SPLITS}

    raw_cn_mean, raw_ad_mean = group_means(raw_mu["train"], train_diagnoses)
    corrected_cn_mean, corrected_ad_mean = group_means(corrected_mu["train"], train_diagnoses)

    rows = []
    for split in SPLITS:
        rows.append(pd.DataFrame({
            "split": split,
            "image_id": data[f"{split}_image_ids"].astype(str),
            "diagnosis": np.char.lower(data[f"{split}_diagnoses"].astype(str)),
            "age": data[f"{split}_age"].astype(float),
            RAW_INDEX: compute_projection(raw_mu[split], raw_cn_mean, raw_ad_mean),
            AGECORR_INDEX: compute_projection(corrected_mu[split], corrected_cn_mean, corrected_ad_mean),
        }))

    return pd.concat(rows, ignore_index=True)


def drop_outliers(subset, target):
    if target not in PTAU_TARGETS or N_OUTLIER_DROP <= 0 or len(subset) <= N_OUTLIER_DROP:
        return subset

    order = np.argsort(subset[target].to_numpy())
    return subset.iloc[np.sort(order[:-N_OUTLIER_DROP])]


def paired_subset(df, index_name, target):
    return drop_outliers(df.dropna(subset=[index_name, target]), target)


def bootstrap_ci(values, labels, function):
    n = len(values)
    if n < 10:
        return float("nan"), float("nan")

    stats = []
    for _ in range(N_BOOTSTRAP):
        sample = RNG.integers(0, n, n)
        try:
            stats.append(function(values[sample], labels[sample]))
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

    pooled_variance = ((len(cn) - 1) * cn.var(ddof=1) + (len(ad) - 1) * ad.var(ddof=1)) / (len(cn) + len(ad) - 2)

    if pooled_variance <= 0:
        return float("nan")
    return float((ad.mean() - cn.mean()) / np.sqrt(pooled_variance))


def correlate_target(df, index_name, target):
    if target not in df.columns:
        return {}

    paired = paired_subset(df, index_name, target)
    out = {f"n_{target}": len(paired)}

    r, p = safe_spearman(paired[index_name].to_numpy(), paired[target].to_numpy())
    out[f"spearman_r_{target}"] = r
    out[f"spearman_p_{target}"] = p

    low, high = bootstrap_ci(
        paired[index_name].to_numpy(),
        paired[target].to_numpy(),
        lambda a, b: safe_spearman(a, b)[0],
    )
    out[f"spearman_ci_lo_{target}"] = low
    out[f"spearman_ci_hi_{target}"] = high

    for group in ["ad", "cn"]:
        group_df = paired[paired["diagnosis"] == group]
        group_r, group_p = safe_spearman(group_df[index_name].to_numpy(), group_df[target].to_numpy())
        out[f"n_{group}_{target}"] = len(group_df)
        out[f"spearman_r_{target}_{group}_only"] = group_r
        out[f"spearman_p_{target}_{group}_only"] = group_p

    return out


def validate_index(df, index_name, split):
    subset = df.dropna(subset=[index_name])

    cn = subset.loc[subset["diagnosis"] == "cn", index_name]
    ad = subset.loc[subset["diagnosis"] == "ad", index_name]

    is_ad = (subset["diagnosis"] == "ad").to_numpy().astype(int)
    values = subset[index_name].to_numpy()

    auc = safe_auc(values, is_ad)
    auc_low, auc_high = bootstrap_ci(values, is_ad, safe_auc)

    if len(cn) >= 3 and len(ad) >= 3:
        _, mannwhitney_p = mannwhitneyu(ad, cn, alternative="two-sided")
    else:
        mannwhitney_p = float("nan")

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
        "auc_ci_lo": auc_low,
        "auc_ci_hi": auc_high,
        "mannwhitney_p": mannwhitney_p,
    }

    for target in TARGETS:
        row.update(correlate_target(subset, index_name, target))

    return row


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

    if len(ad) > 4:
        fit = np.polyfit(ad[index_name], ad[target], 1)
        line_x = np.linspace(ad[index_name].min(), ad[index_name].max(), 50)
        ax.plot(line_x, np.polyval(fit, line_x), color=AD_COLOR, linewidth=1.8, alpha=0.9)

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
    index_df = build_index_table(data)

    demographics = pd.read_csv(DEMOGRAPHICS_PATH, low_memory=False)
    demographics["image_id"] = demographics["image_id"].astype(str)

    keep = ["image_id"] + [target for target in TARGETS if target in demographics.columns]
    merged = index_df.merge(demographics[keep], on="image_id", how="left")
    merged.to_csv(INDEX_PATH, index=False)

    rows = []
    for split in SPLITS + ["heldout"]:
        if split == "heldout":
            split_df = merged[merged["split"].isin(["val", "test"])].copy()
        else:
            split_df = merged[merged["split"] == split].copy()

        for index_name in INDEX_COLUMNS:
            rows.append(validate_index(split_df, index_name, split))

    pd.DataFrame(rows).to_csv(VALIDATION_PATH, index=False)

    test_df = merged[merged["split"] == "test"].copy()

    plot_distribution(test_df, RAW_INDEX, OUT_DIR / f"{RAW_INDEX}_distribution.png")

    for target in TARGETS:
        if target not in test_df.columns:
            continue

        ylabel = "MMSE" if target == "mmse" else target.replace("ptau217_", "pTau-217 ")
        plot_index_vs_target(test_df, target, ylabel, OUT_DIR / f"agecorr_index_vs_{target}.png")

    plot_index_vs_age(test_df, OUT_DIR / "agecorr_index_vs_age.png")

    print(f"Saved: {INDEX_PATH}")
    print(f"Saved: {VALIDATION_PATH}")
    print(f"Saved plots to: {OUT_DIR}")