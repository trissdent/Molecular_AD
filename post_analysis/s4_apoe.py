"""APOE e4 and adjusted associations, test split.

Within AD, per target (MMSE, pTau-217 AlzPath, pTau-217 Janssen) and index version:
    simple Spearman (as in s2);
    partial Spearman adjusted for APOE e4 carrier, age and sex: ranks of the index and
        the target, each residualized on the covariates, then correlated (t-test, n-2-k df);
    OLS of the target on index + APOE e4 + age + sex (coefficients and p per term).
Also: APOE-stratified correlations, AUC of index vs index + e4, genotype gradient.
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from i1_index_core import add_trend_line
from scipy.stats import rankdata, spearmanr
from scipy.stats import t as t_dist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RUN_DIR = os.environ.get("RUN_DIR")

if not RUN_DIR:
    raise ValueError("RUN_DIR environment variable is not set")

RUN_DIR = Path(RUN_DIR).resolve()

INDEX_PATH = RUN_DIR / "s2_index" / "disease_index.csv"
RUN_CONFIG = yaml.safe_load(open(PROJECT_ROOT / "logs" / RUN_DIR.name / "config.yml"))
COHORT_PATH = Path(RUN_CONFIG["data"]["prepared_dir"]) / "cohort.csv"
OUT_DIR = RUN_DIR / "s4_apoe"

RAW_INDEX = "cn_ad_projection"
AGECORR_INDEX = "cn_ad_projection_agecorr"
INDEX_COLUMNS = [RAW_INDEX, AGECORR_INDEX]
PLOT_INDEX = AGECORR_INDEX

PTAU_TARGETS = ["ptau217_alzpath", "ptau217_janssen"]
TARGETS = ["mmse"] + PTAU_TARGETS
TARGET_LABELS = {
    "mmse": "MMSE",
    "ptau217_alzpath": "pTau-217 alzpath",
    "ptau217_janssen": "pTau-217 janssen",
}

GENOTYPE_ORDER = ["2/2", "2/3", "3/3", "2/4", "3/4", "4/4"]
COVARIATES = ["e4_carrier", "age", "sex"]      # adjustment set for the within-AD models

N_BOOTSTRAP = 2000
RNG = np.random.default_rng(42)

CARRIER_COLOR = "#C0504D"
NONCARRIER_COLOR = "#3B7EA1"


def count_e4(genotype):
    if not isinstance(genotype, str):
        return np.nan
    return float(genotype.count("4"))


def paired_subset(df, index_name, target):
    return df.dropna(subset=[index_name, target] + COVARIATES)


def safe_spearman(x, y):
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    r, p = spearmanr(x, y)
    return float(r), float(p)


def partial_spearman(subset, index_name, target):
    """Spearman between index and target with the covariates partialled out."""
    values = subset[COVARIATES].astype(float).to_numpy()
    varying = values[:, np.ptp(values, axis=0) > 0]           # a constant covariate adds nothing
    covariates = np.column_stack([np.ones(len(subset)), varying])

    def residual(values):
        ranks = rankdata(values)
        coef, *_ = np.linalg.lstsq(covariates, ranks, rcond=None)
        return ranks - covariates @ coef

    x, y = residual(subset[index_name].to_numpy()), residual(subset[target].to_numpy())
    r = float(np.corrcoef(x, y)[0, 1])
    dof = len(subset) - 2 - (covariates.shape[1] - 1)
    if dof <= 0 or not np.isfinite(r) or abs(r) >= 1:
        return r, float("nan")
    stat = r * np.sqrt(dof / (1 - r ** 2))
    return r, float(2 * t_dist.sf(abs(stat), dof))


def partial_model(subset, index_name, target):
    design = sm.add_constant(subset[[index_name] + COVARIATES].astype(float))
    model = sm.OLS(subset[target].astype(float), design).fit()

    simple_r, simple_p = safe_spearman(
        subset[index_name].to_numpy(), subset[target].to_numpy())
    adjusted_r, adjusted_p = partial_spearman(subset, index_name, target)

    confidence = model.conf_int()

    return {
        "n": len(subset),
        "simple_r": simple_r,
        "simple_p": simple_p,
        "adjusted_r": adjusted_r,
        "adjusted_p": adjusted_p,
        "index_beta": float(model.params[index_name]),
        "index_ci_lo": float(confidence.loc[index_name, 0]),
        "index_ci_hi": float(confidence.loc[index_name, 1]),
        "index_p": float(model.pvalues[index_name]),
        "e4_beta": float(model.params["e4_carrier"]),
        "e4_p": float(model.pvalues["e4_carrier"]),
        "age_beta": float(model.params["age"]),
        "age_p": float(model.pvalues["age"]),
        "sex_beta": float(model.params["sex"]),
        "sex_p": float(model.pvalues["sex"]),
        "r_squared": float(model.rsquared),
    }


def stratified_stats(subset, index_name, target):
    out = {}

    for flag, name in [(0.0, "noncarrier"), (1.0, "carrier")]:
        group = subset[subset["e4_carrier"] == flag]
        r, p = safe_spearman(group[index_name].to_numpy(), group[target].to_numpy())
        out[f"n_{name}"] = len(group)
        out[f"r_{name}"] = r
        out[f"p_{name}"] = p

    return out


def plot_partial(ad_df, index_name, statistics, output_path):
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(6 * len(TARGETS), 5))

    for ax, target in zip(np.atleast_1d(axes), TARGETS):
        subset = paired_subset(ad_df, index_name, target)
        row = statistics[target]

        for flag, name, color in [(0.0, "non-carrier", NONCARRIER_COLOR),
                                  (1.0, "carrier", CARRIER_COLOR)]:
            group = subset[subset["e4_carrier"] == flag]
            ax.scatter(group[index_name], group[target], s=46, alpha=0.8, color=color,
                       edgecolor="white", linewidth=0.6,
                       label=f"e4 {name} (n={len(group)})")

        add_trend_line(ax, subset[index_name], subset[target], target, "#444444")

        ax.set_xlabel("disease index")
        ax.set_ylabel(TARGET_LABELS[target])
        ax.set_title(
            f"{TARGET_LABELS[target]} (n={row['n']})\n"
            f"adjusted r={row['adjusted_r']:.2f} (p={row['adjusted_p']:.2g})   |   "
            f"APOE e4 p={row['e4_p']:.2g}",
            fontsize=10)
        ax.legend(frameon=False, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("AD: index vs outcome, adjusted for APOE e4, age and sex", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_stratified(subset, index_name, target, output_path):
    label = TARGET_LABELS[target]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, sharex=True)

    for ax, flag, name, color in [
        (axes[0], 0.0, "non-carrier", NONCARRIER_COLOR),
        (axes[1], 1.0, "carrier", CARRIER_COLOR),
    ]:
        group = subset[subset["e4_carrier"] == flag]
        r, p = safe_spearman(group[index_name].to_numpy(), group[target].to_numpy())

        ax.scatter(group[index_name], group[target], s=46, alpha=0.8, color=color,
                   edgecolor="white", linewidth=0.6)

        add_trend_line(ax, group[index_name], group[target], target, color)

        ax.set_xlabel("disease index")
        ax.set_ylabel(label)
        ax.set_title(f"APOE e4 {name} (n={len(group)})\nr={r:.3f} (p={p:.2g})", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"AD: index vs {label} by APOE e4 status", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fit_auc(features, labels):
    if len(np.unique(labels)) < 2:
        return float("nan")

    model = LogisticRegression(max_iter=2000)
    model.fit(features, labels)
    return float(roc_auc_score(labels, model.predict_proba(features)[:, 1]))


def compare_auc(subset, index_name):
    labels = (subset["diagnosis"] == "ad").to_numpy().astype(int)
    index_only = subset[[index_name]].to_numpy(dtype=float)
    with_carrier = subset[[index_name, "e4_carrier"]].to_numpy(dtype=float)

    auc_index = fit_auc(index_only, labels)
    auc_combined = fit_auc(with_carrier, labels)

    differences = []
    for _ in range(N_BOOTSTRAP):
        sample = RNG.integers(0, len(labels), len(labels))

        if len(np.unique(labels[sample])) < 2:
            continue

        boot_index = fit_auc(index_only[sample], labels[sample])
        boot_combined = fit_auc(with_carrier[sample], labels[sample])
        differences.append(boot_combined - boot_index)

    differences = np.array(differences)
    proportion_below = float((differences <= 0).mean())

    return {
        "n": len(subset),
        "n_cn": int((labels == 0).sum()),
        "n_ad": int((labels == 1).sum()),
        "auc_index": auc_index,
        "auc_index_and_e4": auc_combined,
        "auc_difference": auc_combined - auc_index,
        "difference_ci_lo": float(np.percentile(differences, 2.5)),
        "difference_ci_hi": float(np.percentile(differences, 97.5)),
        "bootstrap_p": float(2 * min(proportion_below, 1 - proportion_below)),
    }


def plot_auc_comparison(subset, index_name, row, output_path):
    labels = (subset["diagnosis"] == "ad").to_numpy().astype(int)
    index_only = subset[[index_name]].to_numpy(dtype=float)
    with_carrier = subset[[index_name, "e4_carrier"]].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6.5, 6))

    for features, label, color, auc in [
        (index_only, "index", NONCARRIER_COLOR, row["auc_index"]),
        (with_carrier, "index + APOE e4", CARRIER_COLOR, row["auc_index_and_e4"]),
    ]:
        model = LogisticRegression(max_iter=2000)
        model.fit(features, labels)
        false_positive, true_positive, _ = roc_curve(
            labels, model.predict_proba(features)[:, 1])
        ax.plot(false_positive, true_positive, color=color, linewidth=2,
                label=f"{label} (AUC {auc:.3f})")

    ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1)

    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    ax.set_title(
        f"CN vs AD, test set (n={row['n']}: {row['n_cn']} CN, {row['n_ad']} AD)\n"
        f"difference {row['auc_difference']:+.3f} "
        f"[{row['difference_ci_lo']:+.3f}, {row['difference_ci_hi']:+.3f}], "
        f"p={row['bootstrap_p']:.3f}", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_genotype_gradient(table, output_path):
    present = [genotype for genotype in GENOTYPE_ORDER if genotype in table.index]
    ordered = table.loc[present]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    positions = np.arange(len(ordered))
    colors = [CARRIER_COLOR if "4" in genotype else NONCARRIER_COLOR for genotype in ordered.index]
    ax.bar(positions, ordered["ad_fraction"] * 100, color=colors, width=0.65,
           edgecolor="white", linewidth=0.5)

    for position, (fraction, total) in enumerate(zip(ordered["ad_fraction"], ordered["total"])):
        ax.text(position, fraction * 100 + 1.5, f"{fraction * 100:.0f}%\nn={int(total)}",
                ha="center", fontsize=9, color="#444444")

    ax.set_xticks(positions)
    ax.set_xticklabels(ordered.index)
    ax.set_xlabel("APOE genotype")
    ax.set_ylabel("AD (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Proportion diagnosed AD by APOE genotype, full cohort", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    index_df = pd.read_csv(INDEX_PATH, dtype={"visit_id": str})
    cohort = pd.read_csv(COHORT_PATH, dtype={"visit_id": str})

    keep = ["visit_id", "apoe_genotype"] + [c for c in TARGETS + ["age", "gender"]
                                            if c not in index_df.columns]
    merged = index_df.merge(cohort[keep], on="visit_id", how="left")
    merged["sex"] = pd.to_numeric(merged["gender"], errors="coerce")   # two-level code as in cohort

    merged["e4_count"] = merged["apoe_genotype"].map(count_e4)
    merged["e4_carrier"] = (merged["e4_count"] > 0).astype(float)
    merged.loc[merged["e4_count"].isna(), "e4_carrier"] = np.nan

    test_df = merged[merged["split"] == "test"]
    ad_df = test_df[test_df["diagnosis"] == "ad"]

    partial_rows = []
    stratified_rows = []
    auc_rows = []

    for index_name in INDEX_COLUMNS:
        partial_statistics = {}

        for target in TARGETS:
            subset = paired_subset(ad_df, index_name, target)

            if len(subset) < 10:
                continue

            partial_statistics[target] = partial_model(subset, index_name, target)
            partial_rows.append({"index_name": index_name, "target": target,
                                 **partial_statistics[target]})
            stratified_rows.append({"index_name": index_name, "target": target,
                                    **stratified_stats(subset, index_name, target)})

            if index_name == PLOT_INDEX:
                plot_stratified(subset, index_name, target,
                                OUT_DIR / f"apoe_stratified_{target}.png")

        if index_name == PLOT_INDEX and len(partial_statistics) == len(TARGETS):
            plot_partial(ad_df, index_name, partial_statistics,
                         OUT_DIR / "apoe_partial_model.png")

        auc_subset = test_df.dropna(subset=[index_name, "e4_carrier"])
        auc_row = compare_auc(auc_subset, index_name)
        auc_rows.append({"index_name": index_name, **auc_row})

        if index_name == PLOT_INDEX:
            plot_auc_comparison(auc_subset, index_name, auc_row,
                                OUT_DIR / "apoe_auc_comparison.png")

    genotype_table = pd.crosstab(merged["apoe_genotype"], merged["diagnosis"])
    genotype_table["total"] = genotype_table.sum(axis=1)
    genotype_table["ad_fraction"] = genotype_table["ad"] / genotype_table["total"]
    plot_genotype_gradient(genotype_table, OUT_DIR / "apoe_genotype_gradient.png")

    pd.DataFrame(partial_rows).to_csv(OUT_DIR / "apoe_partial_model.csv", index=False)
    pd.DataFrame(stratified_rows).to_csv(OUT_DIR / "apoe_stratified.csv", index=False)
    pd.DataFrame(auc_rows).to_csv(OUT_DIR / "apoe_auc_comparison.csv", index=False)
    genotype_table.to_csv(OUT_DIR / "apoe_genotype_by_diagnosis.csv")

    print(f"Saved tables and plots to: {OUT_DIR}")