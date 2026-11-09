"""External cohort, A to Z: T1 cubes -> latents of each trained run -> ADNI's index -> statistics.

    python post_analysis/e1_external.py --name umc \\
        --csv /mnt/data_lab513/molecular_mri/umc.csv \\
        --t1-root /mnt/data_lab513/molecular_mri/preprocessed \\
        --runs "smri_pw0.5_s*" "smri_pw0.5_2026*"

--name       cohort name, used in every output path
--csv        one row per subject: subject_id, diagnosis (cn/ad), age, gender, mmse, ptau217,
             apoe_genotype (the cleaned UMC table)
--t1-root    <t1-root>/<subject_id>/mri/brain.finalsurfs.mgz
--runs       trained sMRI runs (globs over checkpoints/, quoted); every one is applied
--keep-ptau-outliers   do not apply the log-scale Tukey pTau outlier rule (applied by default,
                       as for ADNI)
--features   (optional, for s1 and s3) the cohort's FreeSurfer feature table: subject_id plus
             the same feature columns as ADNI's features_smri.csv
--etiv       (optional, for s3 head size) csv with subject_id, etiv (FreeSurfer eTIV, mm^3)

Steps
1. Cubes (once, shared by all runs): crop around the brain, resize, min-max to 0-1, with the
   margin and size of the ADNI sMRI cache (read from its cache_info.json). The box comes from the
   T1 alone (ADNI's used T1 or FA; this cohort has no DTI).
2. Per run: the trained encoder (best.ckpt) gives mu per subject, no training.
3. Per run: the index exactly as s2, fitted on that run's ADNI training latents (age trend on
   training CN, CN-mean -> AD-mean direction) and applied to this cohort. A check recomputes the
   ADNI test index and compares it with the run's s2 output.
4. Per run, s2 statistics: AUC (AD vs CN), Cohen's d, Spearman with MMSE and pTau-217 within AD
   (and pTau over all subjects); 95% CIs by bootstrap over subjects.
5. Per run, s4: the within-AD correlations adjusted for APOE e4, age and sex (partial Spearman).
6. Per run, latent plot: PCA fitted on the run's ADNI training latents; ADNI test and this cohort
   drawn together, CN and AD coloured (does the new cohort land where ADNI's does?).
7. Per run, s1 (needs --features): DCI and per-feature R^2 exactly as s1 (compute_dci on one set:
   standardized within the set, Lasso per feature, in-sample R^2), for ADNI test and this cohort.
8. Per run, s3: probes fitted on ADNI train (as s3), scored on this cohort. Spaces: before = PCA of
   the input cubes (2x downsampled, fitted once on ADNI train), after = mu, features (needs
   --features). Targets: diagnosis; age, head size (needs --etiv) cut at ADNI train quantiles; sex
   (the cohort's 1 = male / 0 = female is mapped to ADNI's two codes; ADNI's male code is the one
   with the larger mean eTIV in ADNI train).
9. All runs together (as m1): one row per run, then the mean over runs and seeds with p < 0.05.

Outputs
    <ADNI smri cache parent>/cache_<name>/<subject_id>.npy     cubes (+ cache_info.json)
    checkpoints/<run>/external_<name>/                          per run: latents.npz, index.csv,
                                                                latent_space.png and
        s2_index/   validation.csv + the s2 plots (distributions, index vs MMSE / pTau / age)
        s4_apoe/    apoe_partial_model / stratified / auc_comparison / genotype csv + s4 plots
        s1_dci/     dci.csv, feature_r2.csv, feature_r2.png (needs --features)
        s3_alignment/ alignment.csv + the s3 2x2 plots
    results/external/<name>_runs.csv, <name>_summary.csv        s2 + s4, all runs together
    results/external/<name>_dci.csv, <name>_alignment.csv       s1 and s3, all runs together
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
import yaml
from scipy.stats import rankdata
from scipy.stats import t as t_dist
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "preprocessing"))

from cache_volumes import CROP, brain_box, crop_cube, load, minmax, resize  # noqa: E402
from configs.config import ConfigReader  # noqa: E402
from extract_latents import load_model  # noqa: E402
from shared.models.metrics import MetricHandler  # noqa: E402
from i1_index_core import (  # noqa: E402
    AGECORR_INDEX,
    OUTLIER_IQR_K,
    RAW_INDEX,
    add_trend_line,
    cohens_d,
    compute_projection,
    fit_aging_model,
    group_means,
    remove_aging,
    safe_auc,
    safe_spearman,
    subject_bootstrap_ci,
)

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")   # as s3

CHECKPOINTS = PROJECT_ROOT / "checkpoints"
LOGS = PROJECT_ROOT / "logs"
RESULTS = PROJECT_ROOT / "results" / "external"
INDEXES = [RAW_INDEX, AGECORR_INDEX]
TARGETS = ["mmse", "ptau217"]
COVARIATES = ["e4_carrier", "age", "sex"]       # as s4
DOWNSAMPLE, N_COMPONENTS, N_BINS, CV_FOLDS, SEED = 2, 128, 3, 5, 42   # as s3
CN_COLOR, AD_COLOR = "#3B7EA1", "#E08A2E"                  # as s2
CARRIER_COLOR, NONCARRIER_COLOR = "#C0504D", "#3B7EA1"     # as s4
TARGET_LABELS = {"mmse": "MMSE", "ptau217": "pTau-217 (pg/mL)"}
GENOTYPE_ORDER = ["2/2", "2/3", "3/3", "2/4", "3/4", "4/4"]
N_BOOTSTRAP_AUC = 2000                                     # as s4
SELECTED_COLOR, UNSELECTED_COLOR = "#2E8B57", "#C0504D"    # as s1


# ---------- 1. cohort and cubes ----------

def load_cohort(csv, keep_outliers):
    df = pd.read_csv(csv, dtype={"subject_id": str})
    df["subject_id"] = df["subject_id"].str.strip()
    df["diagnosis"] = df["diagnosis"].str.strip().str.lower()
    df["adni_id"] = df["subject_id"]               # one visit per subject; i1 bootstraps by adni_id
    df["sex"] = pd.to_numeric(df["gender"], errors="coerce")
    df["e4_carrier"] = df["apoe_genotype"].map(
        lambda g: float("4" in g) if isinstance(g, str) else np.nan)
    if not keep_outliers:
        df = drop_ptau_outliers(df)
    return df.reset_index(drop=True)


def drop_ptau_outliers(df):
    """The ADNI rule (i1_index_core): Tukey upper fence on log pTau over all subjects with a value."""
    values = df.loc[df["ptau217"] > 0, ["subject_id", "ptau217"]].dropna()
    if len(values) < 10:
        return df
    log_v = np.log(values["ptau217"])
    q1, q3 = log_v.quantile([0.25, 0.75])
    fence = q3 + OUTLIER_IQR_K * (q3 - q1)
    flagged = values.loc[log_v > fence, "subject_id"].tolist()
    for sid in flagged:
        value = df.loc[df["subject_id"] == sid, "ptau217"].iloc[0]
        print(f"excluded {sid}: ptau217={value:.3f} > {np.exp(fence):.3f}")
    return df[~df["subject_id"].isin(flagged)]


def build_cubes(cohort, t1_root, cache_dir, adni_cache):
    info = json.loads((adni_cache / "cache_info.json").read_text())
    if info.get("crop") != CROP:
        sys.exit(f"ADNI cache was built with crop={info.get('crop')}, this script uses {CROP}")
    margin, target = int(info["margin"]), tuple(info["target_shape"])
    mine = {**info, "box": "t1_only"}

    cache_dir.mkdir(parents=True, exist_ok=True)
    info_path = cache_dir / "cache_info.json"
    if info_path.is_file() and json.loads(info_path.read_text()) != mine:
        sys.exit(f"{cache_dir} was built with other settings; delete it to rebuild")
    info_path.write_text(json.dumps(mine))

    status = {}
    for sid in cohort["subject_id"]:
        out = cache_dir / f"{sid}.npy"
        t1_path = t1_root / sid / "mri" / "brain.finalsurfs.mgz"
        if out.is_file():
            status[sid] = "skipped"
        elif not t1_path.is_file():
            status[sid] = "no_t1"
        else:
            t1 = load(t1_path)
            center, side = brain_box(t1, np.zeros_like(t1), margin)
            np.save(out, minmax(resize(crop_cube(t1, center, side), target)))
            status[sid] = "done"
    counts = pd.Series(status).value_counts().to_dict()
    print(f"cubes in {cache_dir} (margin {margin}, {target}): {counts}")
    missing = [s for s, v in status.items() if v == "no_t1"]
    if missing:
        print(f"WARNING: no {t1_root}/<id>/mri/brain.finalsurfs.mgz for {missing}; left out")
    return cohort[~cohort["subject_id"].isin(missing)].reset_index(drop=True)


# ---------- 2-3. latents and index, per run ----------

@torch.no_grad()
def encode(run, cohort, cache_dir, device):
    config = ConfigReader.merge(str(LOGS / run / "config.yml"))
    if config.data.modality != "smri":
        sys.exit(f"{run}: modality {config.data.modality}; this cohort has T1 only")
    n_features = len(pd.read_csv(CHECKPOINTS / run / "feature_mean.csv"))
    model = load_model(config, n_features, CHECKPOINTS / run / "best.ckpt", device)
    mus = []
    for start in range(0, len(cohort), 8):
        ids = cohort["subject_id"].iloc[start:start + 8]
        x = np.stack([np.load(cache_dir / f"{s}.npy")[None] for s in ids]).astype(np.float32)
        mus.append(model(torch.from_numpy(x).to(device))["mu"].cpu().numpy())
    return np.concatenate(mus).astype(float)


def adni_latents(run):
    data = np.load(CHECKPOINTS / run / "latents_best_ckpt.npz", allow_pickle=True)
    out = {}
    for split in ("train", "test"):
        out[split] = (data[f"{split}_mu"].astype(float),
                      data[f"{split}_age"].astype(float),
                      np.char.lower(data[f"{split}_diagnoses"].astype(str)),
                      data[f"{split}_visit_ids"].astype(str))
    return out


def fit_index(train_mu, train_age, train_dx):
    """The s2 index (i1_index_core.build_index_table), fitted once on ADNI training latents."""
    coefficients = fit_aging_model(train_mu, train_age, train_dx)
    corrected = remove_aging(train_mu, train_age, coefficients)
    raw_means = group_means(train_mu, train_dx)
    cor_means = group_means(corrected, train_dx)

    def apply(mu, age):
        return {RAW_INDEX: compute_projection(mu, *raw_means),
                AGECORR_INDEX: compute_projection(remove_aging(mu, age, coefficients), *cor_means)}
    return apply


def check_against_s2(run, apply, adni):
    """Recompute the ADNI test index with this code and compare with the run's s2 output."""
    s2 = CHECKPOINTS / run / "s2_index" / "disease_index.csv"
    if not s2.is_file():
        return
    mu, age, _, visits = adni["test"]
    mine = pd.DataFrame(apply(mu, age), index=visits)
    ref = pd.read_csv(s2, dtype={"visit_id": str}).query("split == 'test'").set_index("visit_id")
    common = mine.index.intersection(ref.index)
    diff = float(np.abs(mine.loc[common, INDEXES].to_numpy() - ref.loc[common, INDEXES].to_numpy()).max())
    if diff > 1e-6:
        sys.exit(f"{run}: index differs from s2 by {diff:.2e}; something is wrong")


# ---------- 4-5. statistics ----------

def correlation(frame, index_name, target):
    pair = frame.dropna(subset=[index_name, target])
    r, p = safe_spearman(pair[index_name].to_numpy(), pair[target].to_numpy())
    lo, hi = subject_bootstrap_ci(
        pair, lambda f: safe_spearman(f[index_name].to_numpy(), f[target].to_numpy())[0])
    return {"n": len(pair), "r": r, "p": p, "ci_lo": lo, "ci_hi": hi}


def s2_stats(table):
    rows = []
    for index_name in INDEXES:
        is_ad = (table["diagnosis"] == "ad").to_numpy()
        auc_lo, auc_hi = subject_bootstrap_ci(
            table, lambda f: safe_auc(f[index_name].to_numpy(), (f["diagnosis"] == "ad").to_numpy().astype(int)))
        row = {"index_name": index_name, "n": len(table), "n_cn": int((~is_ad).sum()),
               "n_ad": int(is_ad.sum()),
               "auc": safe_auc(table[index_name].to_numpy(), is_ad.astype(int)),
               "auc_ci_lo": auc_lo, "auc_ci_hi": auc_hi,
               "cohens_d": cohens_d(table.loc[~is_ad, index_name], table.loc[is_ad, index_name])}
        ad = table[is_ad]
        for target in TARGETS:
            for key, value in correlation(ad, index_name, target).items():
                row[f"{target}_ad_{key}"] = value
        for key, value in correlation(table, index_name, "ptau217").items():
            row[f"ptau217_all_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def partial_spearman(subset, index_name, target):
    """As s4: Spearman of index and target after removing the covariates from both (ranks)."""
    values = subset[COVARIATES].astype(float).to_numpy()
    covariates = np.column_stack([np.ones(len(subset)), values[:, np.ptp(values, axis=0) > 0]])

    def residual(v):
        ranks = rankdata(v)
        coef, *_ = np.linalg.lstsq(covariates, ranks, rcond=None)
        return ranks - covariates @ coef

    x, y = residual(subset[index_name].to_numpy()), residual(subset[target].to_numpy())
    r = float(np.corrcoef(x, y)[0, 1])
    dof = len(subset) - 2 - (covariates.shape[1] - 1)
    if dof <= 0 or not np.isfinite(r) or abs(r) >= 1:
        return r, float("nan")
    return r, float(2 * t_dist.sf(abs(r * np.sqrt(dof / (1 - r ** 2))), dof))


def trend(ax, x, y, target, color):
    """i1's trend line; pTau gets the log axis as ADNI's pTau columns do."""
    add_trend_line(ax, x, y, "ptau217_alzpath" if target == "ptau217" else target, color)


# ---- s2 plots (as s2_index.py) ----

def plot_distribution(df, index_name, output_path):
    subset = df.dropna(subset=[index_name])
    cn = subset.loc[subset["diagnosis"] == "cn", index_name]
    ad = subset.loc[subset["diagnosis"] == "ad", index_name]
    auc = safe_auc(subset[index_name].to_numpy(), (subset["diagnosis"] == "ad").to_numpy().astype(int))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(cn, bins=15, alpha=0.7, color=CN_COLOR, label=f"CN (n={len(cn)})")
    ax.hist(ad, bins=15, alpha=0.7, color=AD_COLOR, label=f"AD (n={len(ad)})")
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
    trend(ax, ad[index_name], ad[target], target, AD_COLOR)
    ax.set_xlabel("disease index")
    ax.set_ylabel(ylabel)
    ax.set_title(f"all r={r_all:.3f} (p={p_all:.2g})   |   "
                 f"AD only r={r_ad:.3f} (p={p_ad:.2g}, n={len(ad)})", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)


def plot_index_vs_target(df, target, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    scatter_panel(axes[0], df.dropna(subset=[RAW_INDEX, target]), RAW_INDEX, target, TARGET_LABELS[target])
    scatter_panel(axes[1], df.dropna(subset=[AGECORR_INDEX, target]), AGECORR_INDEX, target,
                  TARGET_LABELS[target])
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
        g = subset[subset["diagnosis"] == group]
        ax.scatter(g["age"], g[AGECORR_INDEX], s=42, alpha=0.75, color=color,
                   edgecolor="white", linewidth=0.6, label=f"{label} (n={len(g)})")
    ax.set_xlabel("age")
    ax.set_ylabel("age-corrected index")
    ax.set_title(f"index vs age: {r_raw:.3f} before correction, {r_corrected:.3f} after", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def s2_plots(table, out):
    for index_name in INDEXES:
        plot_distribution(table, index_name, out / f"{index_name}_distribution.png")
    for target in TARGETS:
        plot_index_vs_target(table, target, out / f"agecorr_index_vs_{target}.png")
    plot_index_vs_age(table, out / "agecorr_index_vs_age.png")


# ---- s4 (as s4_apoe.py) ----

def partial_model(subset, index_name, target):
    design = sm.add_constant(subset[[index_name] + COVARIATES].astype(float))
    model = sm.OLS(subset[target].astype(float), design).fit()
    simple_r, simple_p = safe_spearman(subset[index_name].to_numpy(), subset[target].to_numpy())
    adjusted_r, adjusted_p = partial_spearman(subset, index_name, target)
    ci = model.conf_int()
    return {"n": len(subset), "simple_r": simple_r, "simple_p": simple_p,
            "adjusted_r": adjusted_r, "adjusted_p": adjusted_p,
            "index_beta": float(model.params[index_name]),
            "index_ci_lo": float(ci.loc[index_name, 0]), "index_ci_hi": float(ci.loc[index_name, 1]),
            "index_p": float(model.pvalues[index_name]),
            "e4_beta": float(model.params["e4_carrier"]), "e4_p": float(model.pvalues["e4_carrier"]),
            "age_beta": float(model.params["age"]), "age_p": float(model.pvalues["age"]),
            "sex_beta": float(model.params["sex"]), "sex_p": float(model.pvalues["sex"]),
            "r_squared": float(model.rsquared)}


def stratified_stats(subset, index_name, target):
    out = {}
    for flag, name in [(0.0, "noncarrier"), (1.0, "carrier")]:
        group = subset[subset["e4_carrier"] == flag]
        r, p = safe_spearman(group[index_name].to_numpy(), group[target].to_numpy())
        out[f"n_{name}"], out[f"r_{name}"], out[f"p_{name}"] = len(group), r, p
    return out


def plot_partial(ad_df, index_name, statistics, output_path):
    targets = list(statistics)
    fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 5))
    for ax, target in zip(np.atleast_1d(axes), targets):
        subset = ad_df.dropna(subset=[index_name, target])
        row = statistics[target]
        for flag, name, color in [(0.0, "non-carrier", NONCARRIER_COLOR), (1.0, "carrier", CARRIER_COLOR)]:
            g = subset[subset["e4_carrier"] == flag]
            ax.scatter(g[index_name], g[target], s=46, alpha=0.8, color=color,
                       edgecolor="white", linewidth=0.6, label=f"e4 {name} (n={len(g)})")
        trend(ax, subset[index_name], subset[target], target, "#444444")
        ax.set_xlabel("disease index")
        ax.set_ylabel(TARGET_LABELS[target])
        ax.set_title(f"{TARGET_LABELS[target]} (n={row['n']})\n"
                     f"adjusted r={row['adjusted_r']:.2f} (p={row['adjusted_p']:.2g})   |   "
                     f"APOE e4 p={row['e4_p']:.2g}", fontsize=10)
        ax.legend(frameon=False, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("AD: index vs outcome, adjusted for APOE e4, age and sex", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_stratified(subset, index_name, target, output_path):
    label = TARGET_LABELS[target]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, sharex=True)
    for ax, flag, name, color in [(axes[0], 0.0, "non-carrier", NONCARRIER_COLOR),
                                  (axes[1], 1.0, "carrier", CARRIER_COLOR)]:
        g = subset[subset["e4_carrier"] == flag]
        r, p = safe_spearman(g[index_name].to_numpy(), g[target].to_numpy())
        ax.scatter(g[index_name], g[target], s=46, alpha=0.8, color=color, edgecolor="white", linewidth=0.6)
        trend(ax, g[index_name], g[target], target, color)
        ax.set_xlabel("disease index")
        ax.set_ylabel(label)
        ax.set_title(f"APOE e4 {name} (n={len(g)})\nr={r:.3f} (p={p:.2g})", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"AD: index vs {label} by APOE e4 status", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fit_auc(features, labels):
    if len(np.unique(labels)) < 2:
        return float("nan")
    model = LogisticRegression(max_iter=2000).fit(features, labels)
    return float(roc_auc_score(labels, model.predict_proba(features)[:, 1]))


def compare_auc(subset, index_name, rng):
    labels = (subset["diagnosis"] == "ad").to_numpy().astype(int)
    index_only = subset[[index_name]].to_numpy(dtype=float)
    with_carrier = subset[[index_name, "e4_carrier"]].to_numpy(dtype=float)
    auc_index, auc_combined = fit_auc(index_only, labels), fit_auc(with_carrier, labels)
    differences = []
    for _ in range(N_BOOTSTRAP_AUC):
        sample = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[sample])) < 2:
            continue
        differences.append(fit_auc(with_carrier[sample], labels[sample])
                           - fit_auc(index_only[sample], labels[sample]))
    differences = np.array(differences)
    below = float((differences <= 0).mean())
    return {"n": len(subset), "n_cn": int((labels == 0).sum()), "n_ad": int((labels == 1).sum()),
            "auc_index": auc_index, "auc_index_and_e4": auc_combined,
            "auc_difference": auc_combined - auc_index,
            "difference_ci_lo": float(np.percentile(differences, 2.5)),
            "difference_ci_hi": float(np.percentile(differences, 97.5)),
            "bootstrap_p": float(2 * min(below, 1 - below))}


def plot_auc_comparison(subset, index_name, row, output_path):
    labels = (subset["diagnosis"] == "ad").to_numpy().astype(int)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for cols, label, color, auc in [([index_name], "index", NONCARRIER_COLOR, row["auc_index"]),
                                    ([index_name, "e4_carrier"], "index + APOE e4", CARRIER_COLOR,
                                     row["auc_index_and_e4"])]:
        x = subset[cols].to_numpy(dtype=float)
        model = LogisticRegression(max_iter=2000).fit(x, labels)
        fpr, tpr, _ = roc_curve(labels, model.predict_proba(x)[:, 1])
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{label} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    ax.set_title(f"CN vs AD (n={row['n']}: {row['n_cn']} CN, {row['n_ad']} AD)\n"
                 f"difference {row['auc_difference']:+.3f} "
                 f"[{row['difference_ci_lo']:+.3f}, {row['difference_ci_hi']:+.3f}], "
                 f"p={row['bootstrap_p']:.3f}", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_genotype_gradient(table, output_path):
    ordered = table.loc[[g for g in GENOTYPE_ORDER if g in table.index]]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    positions = np.arange(len(ordered))
    colors = [CARRIER_COLOR if "4" in g else NONCARRIER_COLOR for g in ordered.index]
    ax.bar(positions, ordered["ad_fraction"] * 100, color=colors, width=0.65, edgecolor="white", linewidth=0.5)
    for pos, (fraction, total) in enumerate(zip(ordered["ad_fraction"], ordered["total"])):
        ax.text(pos, fraction * 100 + 1.5, f"{fraction * 100:.0f}%\nn={int(total)}",
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


def s4_apoe(table, out):
    """Everything s4 does, on this cohort (all subjects play the role of ADNI's test split)."""
    rng = np.random.default_rng(42)
    ad = table[table["diagnosis"] == "ad"]
    partial_rows, stratified_rows, auc_rows = [], [], []
    for index_name in INDEXES:
        statistics = {}
        for target in TARGETS:
            subset = ad.dropna(subset=[index_name, target] + COVARIATES)
            if len(subset) < 10:                                      # as s4
                print(f"s4: {target} within AD has {len(subset)} subjects with covariates, skipped")
                continue
            statistics[target] = partial_model(subset, index_name, target)
            partial_rows.append({"index_name": index_name, "target": target, **statistics[target]})
            stratified_rows.append({"index_name": index_name, "target": target,
                                    **stratified_stats(subset, index_name, target)})
            if index_name == AGECORR_INDEX:
                plot_stratified(subset, index_name, target, out / f"apoe_stratified_{target}.png")
        if index_name == AGECORR_INDEX and statistics:
            plot_partial(ad, index_name, statistics, out / "apoe_partial_model.png")
        auc_subset = table.dropna(subset=[index_name, "e4_carrier"])
        auc_row = compare_auc(auc_subset, index_name, rng)
        auc_rows.append({"index_name": index_name, **auc_row})
        if index_name == AGECORR_INDEX:
            plot_auc_comparison(auc_subset, index_name, auc_row, out / "apoe_auc_comparison.png")

    genotype = pd.crosstab(table["apoe_genotype"], table["diagnosis"])
    genotype["total"] = genotype.sum(axis=1)
    genotype["ad_fraction"] = genotype.get("ad", 0) / genotype["total"]
    plot_genotype_gradient(genotype, out / "apoe_genotype_gradient.png")

    partial = pd.DataFrame(partial_rows)
    partial.to_csv(out / "apoe_partial_model.csv", index=False)
    pd.DataFrame(stratified_rows).to_csv(out / "apoe_stratified.csv", index=False)
    pd.DataFrame(auc_rows).to_csv(out / "apoe_auc_comparison.csv", index=False)
    genotype.to_csv(out / "apoe_genotype_by_diagnosis.csv")
    return partial


# ---------- 6. latent plot ----------

def plot_latents(run, adni, mu, table, name, path):
    pca = PCA(n_components=2, random_state=0).fit(adni["train"][0])
    test_mu, _, test_dx, _ = adni["test"]
    a, b = pca.transform(test_mu), pca.transform(mu)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for dx, color in (("cn", CN_COLOR), ("ad", AD_COLOR)):
        m = test_dx == dx
        ax.scatter(a[m, 0], a[m, 1], s=14, color=color, alpha=0.3, label=f"ADNI test {dx.upper()}")
        m = (table["diagnosis"] == dx).to_numpy()
        ax.scatter(b[m, 0], b[m, 1], s=46, color=color, edgecolor="black", linewidth=0.8,
                   label=f"{name} {dx.upper()}")
    ax.set_xlabel("PC1 (fitted on ADNI train)")
    ax.set_ylabel("PC2")
    ax.set_title(f"Latent space: ADNI test vs {name}\n{run}", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------- 7. s1: DCI ----------

def load_features(path, cohort, feature_names):
    """Cohort features in ADNI's column order, one row per cohort subject (NaN where missing)."""
    df = pd.read_csv(path, dtype={"subject_id": str})
    df["subject_id"] = df["subject_id"].str.strip()
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        sys.exit(f"{path}: {len(missing)} ADNI features missing, e.g. {missing[:5]}")
    return df.set_index("subject_id").reindex(cohort["subject_id"])[list(feature_names)].to_numpy(float)


def active_topk(run):
    """Feature indices the saved model was trained toward (as m3): the last top-k chosen before
    the best epoch; empty before the warm-up ends."""
    epoch = int(torch.load(CHECKPOINTS / run / "best.ckpt", map_location="cpu", weights_only=False)["epoch"])
    history = LOGS / run / "topk_history.csv"
    if not history.is_file():
        return []
    before = pd.read_csv(history).query("epoch < @epoch").sort_values("epoch")
    return [] if before.empty else [int(i) for i in str(before.iloc[-1]["feature_idx"]).split("|")]


def plot_feature_r2(r2, topk, output_path):
    """Per-feature R^2: ADNI test vs this cohort; CIMLR's active top-k highlighted (as s1's colours)."""
    selected = np.zeros(len(r2), dtype=bool)
    selected[topk] = True
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(r2.loc[~selected, "adni_test_r2"], r2.loc[~selected, "cohort_r2"], s=10, alpha=0.4,
               color=UNSELECTED_COLOR, label="other features")
    ax.scatter(r2.loc[selected, "adni_test_r2"], r2.loc[selected, "cohort_r2"], s=30, alpha=0.9,
               color=SELECTED_COLOR, label=f"CIMLR top-k at best epoch ({selected.sum()})")
    ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1)
    ax.set_xlabel("R² on ADNI test")
    ax.set_ylabel("R² on this cohort")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Feature encoding: ADNI test vs external cohort", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def s1_dci(run, adni_npz, mu, features):
    config = yaml.safe_load((LOGS / run / "config.yml").read_text())
    metric = MetricHandler(dci_alpha=config.get("metrics", {}).get("dci_alpha", 0.1))
    ok = np.isfinite(features).all(axis=1)
    sets = {"adni_test": (adni_npz["test_mu"], adni_npz["test_features"]),
            "cohort": (mu[ok], features[ok])}
    rows, r2 = [], {}
    for name, (z, f) in sets.items():
        dci = metric.compute_dci(z, f)
        rows.append({"set": name, "n": len(z), "D": dci["disentanglement"],
                     "C": dci["completeness"], "I": dci["informativeness"]})
        r2[f"{name}_r2"] = dci["r2_scores"]
    names = adni_npz["feature_names"].astype(str)
    return pd.DataFrame(rows), pd.DataFrame({"feature": names, **r2})


# ---------- 8. s3: decodability ----------

def pooled_cube(path):
    v = np.load(path).astype(np.float32)
    d = DOWNSAMPLE
    s = [n // d * d for n in v.shape]
    v = v[:s[0], :s[1], :s[2]].reshape(s[0] // d, d, s[1] // d, d, s[2] // d, d).mean(axis=(1, 3, 5))
    return v.ravel()


def image_space(adni_train_ids, adni_cache, cohort, cache_dir):
    """'before' space of s3: PCA of the input cubes fitted on ADNI train, applied to the cohort."""
    train = np.stack([pooled_cube(adni_cache / f"{v}.npy") for v in adni_train_ids])
    n = min(N_COMPONENTS, *train.shape)
    pca = PCA(n_components=n, svd_solver="randomized", random_state=SEED).fit(train)
    cohort_x = np.stack([pooled_cube(cache_dir / f"{s}.npy") for s in cohort["subject_id"]])
    print(f"s3 image PCA: {pca.n_components_} components, "
          f"{pca.explained_variance_ratio_.sum():.1%} of ADNI train variance")
    return pca.transform(train), pca.transform(cohort_x)


def bin_labels(train_values, values):
    finite = train_values[np.isfinite(train_values)]
    edges = np.quantile(finite, np.linspace(0, 1, N_BINS + 1)[1:-1])
    out = np.full(len(values), np.nan)
    ok = np.isfinite(values)
    out[ok] = np.digitize(values[ok], edges)
    return out


def decodability(train_x, train_y, eval_x, eval_y):
    """As s3: probe fitted on ADNI train, balanced accuracy on the evaluation subjects."""
    def valid(y):
        return np.isfinite(y) if y.dtype.kind == "f" else np.ones(len(y), dtype=bool)
    tm, em = valid(train_y), valid(eval_y)
    if len(np.unique(train_y[tm])) < 2 or len(np.unique(eval_y[em])) < 2:
        return float("nan")
    scaler = StandardScaler().fit(train_x[tm])
    probe = LogisticRegressionCV(Cs=10, cv=CV_FOLDS, class_weight="balanced",
                                 max_iter=5000, random_state=SEED)
    probe.fit(scaler.transform(train_x[tm]), train_y[tm])
    return float(balanced_accuracy_score(eval_y[em], probe.predict(scaler.transform(eval_x[em]))))


def to_adni_sex(cohort_sex, adni_sex, adni_etiv):
    """The cohort codes sex 1 = male, 0 = female. ADNI's two codes come from the lab database;
    its male code is the one with the larger mean eTIV in ADNI train (men have larger heads)."""
    codes = np.unique(adni_sex[np.isfinite(adni_sex)])
    ok = np.isfinite(adni_sex) & np.isfinite(adni_etiv)
    if len(codes) != 2 or not ok.any():
        print(f"s3: sex skipped, ADNI has codes {codes.tolist()}")
        return None
    means = {c: adni_etiv[ok & (adni_sex == c)].mean() for c in codes}
    male, female = max(means, key=means.get), min(means, key=means.get)
    print(f"s3: ADNI sex codes, male = {male:g}, female = {female:g} (from mean eTIV)")
    out = np.full(len(cohort_sex), np.nan)
    out[cohort_sex == 1] = male
    out[cohort_sex == 0] = female
    return out


def s3_scatter(ax, coords, labels, title, kind):
    if kind == "diagnosis":
        colors = {"CN": "tab:blue", "AD": "tab:red"}
        for value in np.unique(labels):
            m = labels == value
            ax.scatter(coords[m, 0], coords[m, 1], s=30, alpha=0.8, color=colors.get(value, "tab:gray"),
                       label=str(value))
    else:
        finite = np.isfinite(labels)
        for value in np.unique(labels[finite]):
            m = finite & (labels == value)
            label = f"sex {int(value)}" if kind == "sex" else ["low", "middle", "high"][int(value)]
            ax.scatter(coords[m, 0], coords[m, 1], s=30, alpha=0.8, label=label)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)


def s3_plot(coords, scores, diagnoses, nuisance, name, output_path):
    """As s3's alignment plot: 2 x 2, before / after, by diagnosis / by the other target."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    kind = "sex" if name == "sex" else "bins"
    n_classes = len(np.unique(nuisance[np.isfinite(nuisance)]))
    for row, space in enumerate(["before", "after"]):
        s3_scatter(axes[row, 0], coords[space], diagnoses,
                   f"{space} — by diagnosis (bal. acc = {scores[space]['diagnosis']:.3f}, chance 0.50)",
                   "diagnosis")
        s3_scatter(axes[row, 1], coords[space], nuisance,
                   f"{space} — by {name} (bal. acc = {scores[space][name]:.3f}, "
                   f"chance {1 / n_classes:.2f})", kind)
    fig.suptitle(f"Alignment (external cohort), image space → latent: {name} "
                 f"{scores['before'][name]:.3f} → {scores['after'][name]:.3f} | "
                 f"diagnosis {scores['before']['diagnosis']:.3f} → {scores['after']['diagnosis']:.3f}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def s3_alignment(run, adni_npz, mu, before, cohort, features, out):
    spaces = {"before": before, "after": (adni_npz["train_mu"], mu)}
    if features is not None:
        mean = pd.read_csv(CHECKPOINTS / run / "feature_mean.csv", index_col=0).squeeze("columns")
        std = pd.read_csv(CHECKPOINTS / run / "feature_std.csv", index_col=0).squeeze("columns")
        names = adni_npz["feature_names"].astype(str)
        scaled = (features - mean.loc[names].to_numpy()) / std.loc[names].to_numpy()
        ok = np.isfinite(scaled).all(axis=1)
        spaces["features"] = (adni_npz["train_features"], scaled)
        feature_ok = ok
    train_dx = np.char.lower(np.char.strip(adni_npz["train_diagnoses"].astype(str)))
    cohort_dx = cohort["diagnosis"].to_numpy().astype(str)

    targets = {"diagnosis": (train_dx, cohort_dx, 0.5)}
    train_age = adni_npz["train_age"].astype(float)
    targets["age"] = (bin_labels(train_age, train_age),
                      bin_labels(train_age, cohort["age"].to_numpy(float)), 1 / N_BINS)
    if "etiv" in cohort:
        train_etiv = adni_npz["train_etiv"].astype(float)
        targets["head size"] = (bin_labels(train_etiv, train_etiv),
                                bin_labels(train_etiv, cohort["etiv"].to_numpy(float)), 1 / N_BINS)
    train_sex = adni_npz["train_gender"].astype(float)
    cohort_sex = to_adni_sex(cohort["sex"].to_numpy(float), train_sex,
                             adni_npz["train_etiv"].astype(float))
    if cohort_sex is not None:
        targets["sex"] = (train_sex, cohort_sex, 0.5)

    rows = []
    for target, (train_y, eval_y, chance) in targets.items():
        row = {"target": target, "chance": chance}
        for space, (train_x, eval_x) in spaces.items():
            y = eval_y
            if space == "features":                # only subjects with a complete feature row
                eval_x, y = eval_x[feature_ok], eval_y[feature_ok]
            row[space] = decodability(train_x, train_y, eval_x, y)
        rows.append(row)
    result = pd.DataFrame(rows)

    # plots as s3: 2D display per space (PCA fitted on ADNI train), cohort subjects drawn
    coords = {}
    for space in ("before", "after"):
        train_x, eval_x = spaces[space]
        scaler = StandardScaler().fit(train_x)
        pca2 = PCA(n_components=2).fit(scaler.transform(train_x))
        coords[space] = pca2.transform(scaler.transform(eval_x))
    scores = {space: dict(zip(result["target"], result[space])) for space in ("before", "after")}
    for target, (_, eval_y, _) in targets.items():
        if target != "diagnosis":
            s3_plot(coords, scores, np.char.upper(cohort_dx), np.asarray(eval_y, dtype=float), target,
                    out / f"alignment_{target.replace(' ', '_')}.png")
    return result


# ---------- 9. all runs ----------

def summarize(per_run):
    value_cols = [c for c in per_run.columns if c.endswith(("_r", "auc", "cohens_d"))
                  and not c.startswith("ci")]
    p_cols = [c for c in per_run.columns if c.endswith("_p")]
    rows = []
    for index_name, g in per_run.groupby("index_name"):
        row = {"index_name": index_name, "n_runs": len(g)}
        for c in value_cols:
            row[f"{c} mean"], row[f"{c} sd"] = g[c].mean(), g[c].std()
        for c in p_cols:
            row[f"{c}<0.05"] = f"{int((g[c] < 0.05).sum())}/{len(g)}"
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--t1-root", required=True, type=Path)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--keep-ptau-outliers", action="store_true")
    parser.add_argument("--features", type=Path, default=None)
    parser.add_argument("--etiv", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runs = sorted({r.name for p in args.runs for r in CHECKPOINTS.glob(p)
                   if (r / "best.ckpt").is_file()})
    if not runs:
        sys.exit(f"no trained runs match {args.runs}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    first = yaml.safe_load((LOGS / runs[0] / "config.yml").read_text())
    adni_cache = Path(first["data"]["cache_dirs"]["smri"])
    cache_dir = adni_cache.parent / f"cache_{args.name}"

    cohort = load_cohort(args.csv, args.keep_ptau_outliers)
    if args.etiv:
        etiv = pd.read_csv(args.etiv, dtype={"subject_id": str})
        etiv["subject_id"] = etiv["subject_id"].str.strip()
        cohort = cohort.merge(etiv[["subject_id", "etiv"]], on="subject_id", how="left")
    cohort = build_cubes(cohort, args.t1_root, cache_dir, adni_cache)
    print(f"{args.name}: {len(cohort)} subjects "
          f"({(cohort['diagnosis'] == 'cn').sum()} CN, {(cohort['diagnosis'] == 'ad').sum()} AD)")

    first_npz = np.load(CHECKPOINTS / runs[0] / "latents_best_ckpt.npz", allow_pickle=True)
    before = image_space(first_npz["train_visit_ids"].astype(str), adni_cache, cohort, cache_dir)
    features = (load_features(args.features, cohort, first_npz["feature_names"].astype(str))
                if args.features else None)

    per_run, per_run_adj, dci_runs, align_runs = [], [], [], []
    for run in runs:
        out = CHECKPOINTS / run / f"external_{args.name}"
        out.mkdir(exist_ok=True)
        seed = yaml.safe_load((LOGS / run / "config.yml").read_text())["training"]["seed"]

        mu = encode(run, cohort, cache_dir, device)
        adni = adni_latents(run)
        apply = fit_index(*adni["train"][:3])
        check_against_s2(run, apply, adni)

        table = cohort.copy()
        for k, v in apply(mu, cohort["age"].to_numpy(dtype=float)).items():
            table[k] = v
        np.savez_compressed(out / "latents.npz", mu=mu, subject_ids=cohort["subject_id"].to_numpy())
        table.to_csv(out / "index.csv", index=False)

        dirs = {k: out / k for k in ("s1_dci", "s2_index", "s3_alignment", "s4_apoe")}
        for d in dirs.values():
            d.mkdir(exist_ok=True)

        stats = s2_stats(table)
        stats.to_csv(dirs["s2_index"] / "validation.csv", index=False)
        s2_plots(table, dirs["s2_index"])
        adjusted = s4_apoe(table, dirs["s4_apoe"])
        plot_latents(run, adni, mu, table, args.name, out / "latent_space.png")

        npz = np.load(CHECKPOINTS / run / "latents_best_ckpt.npz", allow_pickle=True)
        if features is not None:
            dci, r2 = s1_dci(run, npz, mu, features)
            dci.to_csv(dirs["s1_dci"] / "dci.csv", index=False)
            r2.to_csv(dirs["s1_dci"] / "feature_r2.csv", index=False)
            plot_feature_r2(r2, active_topk(run), dirs["s1_dci"] / "feature_r2.png")
            dci_runs.append(dci.assign(run=run, seed=seed))
        alignment = s3_alignment(run, npz, mu, before, cohort, features, dirs["s3_alignment"])
        alignment.to_csv(dirs["s3_alignment"] / "alignment.csv", index=False)
        align_runs.append(alignment.assign(run=run, seed=seed))

        per_run.append(stats.assign(run=run, seed=seed))
        per_run_adj.append(adjusted.assign(run=run, seed=seed))
        row = stats.set_index("index_name").loc[AGECORR_INDEX]
        print(f"{run} (seed {seed}): AUC {row['auc']:.3f} | MMSE in AD r {row['mmse_ad_r']:.3f} "
              f"(p {row['mmse_ad_p']:.3g}) | pTau in AD r {row['ptau217_ad_r']:.3f} "
              f"(p {row['ptau217_ad_p']:.3g}, n {row['ptau217_ad_n']})")

    per_run = pd.concat(per_run, ignore_index=True)
    per_run_adj = pd.concat(per_run_adj, ignore_index=True)
    adj_wide = per_run_adj.pivot_table(index=["run", "index_name"], columns="target",
                                       values=["adjusted_r", "adjusted_p"])
    adj_wide.columns = [f"{t}_ad_{v.replace('adjusted', 'adj')}" for v, t in adj_wide.columns]
    per_run = per_run.merge(adj_wide.reset_index(), on=["run", "index_name"], how="left")
    summary = summarize(per_run)

    RESULTS.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(RESULTS / f"{args.name}_runs.csv", index=False)
    summary.to_csv(RESULTS / f"{args.name}_summary.csv", index=False)

    pd.set_option("display.width", 220)
    show = ["index_name", "n_runs", "auc mean", "cohens_d mean", "mmse_ad_r mean", "mmse_ad_p<0.05",
            "ptau217_ad_r mean", "ptau217_ad_p<0.05", "mmse_ad_adj_r mean", "mmse_ad_adj_p<0.05",
            "ptau217_ad_adj_r mean", "ptau217_ad_adj_p<0.05"]
    print(f"\n{args.name}: mean over {len(runs)} runs\n"
          + summary[[c for c in show if c in summary.columns]].round(3).to_string(index=False))
    alignment = pd.concat(align_runs, ignore_index=True)
    alignment.to_csv(RESULTS / f"{args.name}_alignment.csv", index=False)
    spaces = [c for c in ("before", "after", "features") if c in alignment.columns]
    print(f"\ns3 decodability on {args.name} (balanced accuracy, probes fitted on ADNI train; "
          f"mean over runs)\n"
          + alignment.groupby("target", sort=False)[["chance"] + spaces].mean().round(3).to_string())
    if dci_runs:
        dci = pd.concat(dci_runs, ignore_index=True)
        dci.to_csv(RESULTS / f"{args.name}_dci.csv", index=False)
        print(f"\ns1 DCI (mean over runs)\n"
              + dci.groupby("set")[["n", "D", "C", "I"]].mean().round(3).to_string())
        print(f"Saved: {RESULTS / f'{args.name}_dci.csv'}")
    print(f"Saved: {RESULTS / f'{args.name}_alignment.csv'}")
    print(f"\nSaved: {RESULTS / f'{args.name}_runs.csv'}")
    print(f"Saved: {RESULTS / f'{args.name}_summary.csv'}")
    print(f"Per run: checkpoints/<run>/external_{args.name}/")