"""Shared steps for building and validating a CN-AD index.

Used by s2_index.py (CALIBER latents) and i2_baselines.py (simple indices), so every
index goes through the identical age correction, projection and statistics.

Index: representation X (n x d) per split
    -> age correction fitted on training CN only (linear in age, per dimension)
    -> projection onto the training CN-mean -> AD-mean direction
       (0 = CN mean, 1 = AD mean; higher = more AD-like)
Statistics: AUC (AD vs CN), Cohen's d, Spearman with MMSE / pTau overall and
within each diagnosis. 95% CIs by bootstrap over subjects (adni_id), so a
subject's visits are resampled together.

Main result, fixed before looking at results:
    age-corrected index, test split, Spearman with MMSE within AD.
"""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score

SPLITS = ["train", "val", "test"]
RAW_INDEX = "cn_ad_projection"
AGECORR_INDEX = "cn_ad_projection_agecorr"
INDEX_COLUMNS = [RAW_INDEX, AGECORR_INDEX]

PTAU_TARGETS = ["ptau217_alzpath", "ptau217_janssen"]
TARGETS = ["mmse"] + PTAU_TARGETS

PRIMARY_INDEX = AGECORR_INDEX
PRIMARY_SPLIT = "test"
PRIMARY_ENDPOINT = "spearman_r_mmse_ad_only"

N_BOOTSTRAP = 1000
SEED = 42

# pTau outliers: Tukey upper fence on log pTau (Q3 + k * IQR), per assay, over all visits with
# that assay. pTau-217 is right-skewed, so the fence is applied on the log scale; on the raw scale
# it would flag the most advanced AD patients, which are the signal. A subject flagged on either
# assay is left out of every evaluation (CALIBER and baselines alike); none of them is in training.
OUTLIER_IQR_K = 1.5


def find_ptau_outliers(df):
    """{adni_id: reason} for subjects with a pTau value above the log-scale Tukey fence."""
    flagged = {}
    for target in PTAU_TARGETS:
        if target not in df.columns:
            continue
        values = df[["adni_id", target]].dropna()
        values = values[values[target] > 0]
        if len(values) < 10:
            continue
        log_v = np.log(values[target])
        q1, q3 = log_v.quantile([0.25, 0.75])
        fence = q3 + OUTLIER_IQR_K * (q3 - q1)
        for adni_id, value in values.loc[log_v > fence].itertuples(index=False, name=None):
            flagged[adni_id] = f"{target}={value:.3f} > {np.exp(fence):.3f}"
    return flagged


def drop_excluded(df):
    """Remove pTau-outlier subjects (all their visits) from an evaluation table."""
    flagged = find_ptau_outliers(df)
    for adni_id, reason in flagged.items():
        print(f"excluded {adni_id}: {reason}")
    return df[~df["adni_id"].isin(flagged)].reset_index(drop=True)


def add_trend_line(ax, x, y, target, color):
    """Least-squares line; for pTau fitted on log values and drawn on a log axis."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) <= 4:
        return
    line_x = np.linspace(x.min(), x.max(), 50)
    if target in PTAU_TARGETS:
        ax.set_yscale("log")
        fit = np.polyfit(x, np.log(y), 1)
        ax.plot(line_x, np.exp(np.polyval(fit, line_x)), color=color, linewidth=1.8, alpha=0.9)
    else:
        fit = np.polyfit(x, y, 1)
        ax.plot(line_x, np.polyval(fit, line_x), color=color, linewidth=1.8, alpha=0.9)


# ---------- building the index ----------

def fit_aging_model(train_x, train_age, train_diagnoses):
    mask = (train_diagnoses == "cn") & np.isfinite(train_age)
    design = np.column_stack([np.ones(mask.sum()), train_age[mask]])
    coefficients, *_ = np.linalg.lstsq(design, train_x[mask], rcond=None)
    return coefficients


def remove_aging(x, age, coefficients):
    corrected = x.copy()
    finite = np.isfinite(age)
    if finite.any():
        design = np.column_stack([np.ones(finite.sum()), age[finite]])
        corrected[finite] = x[finite] - design @ coefficients
    return corrected


def group_means(train_x, train_diagnoses):
    return (train_x[train_diagnoses == "cn"].mean(axis=0),
            train_x[train_diagnoses == "ad"].mean(axis=0))


def compute_projection(x, cn_mean, ad_mean):
    direction = ad_mean - cn_mean
    return ((x - cn_mean) @ direction) / (direction @ direction)


def build_index_table(reps, meta):
    """reps: {split: (n, d) array}; meta: {split: DataFrame with visit_id, diagnosis, age}
    in the same row order. Returns one row per visit with both index versions."""
    train_x = np.asarray(reps["train"], dtype=float)
    train_age = meta["train"]["age"].to_numpy(dtype=float)
    train_dx = meta["train"]["diagnosis"].str.lower().to_numpy()

    coefficients = fit_aging_model(train_x, train_age, train_dx)
    corrected = {s: remove_aging(np.asarray(reps[s], dtype=float),
                                 meta[s]["age"].to_numpy(dtype=float), coefficients)
                 for s in SPLITS}

    raw_cn, raw_ad = group_means(train_x, train_dx)
    cor_cn, cor_ad = group_means(corrected["train"], train_dx)

    rows = []
    for s in SPLITS:
        rows.append(pd.DataFrame({
            "split": s,
            "visit_id": meta[s]["visit_id"].astype(str).to_numpy(),
            "diagnosis": meta[s]["diagnosis"].str.lower().to_numpy(),
            "age": meta[s]["age"].to_numpy(dtype=float),
            RAW_INDEX: compute_projection(np.asarray(reps[s], dtype=float), raw_cn, raw_ad),
            AGECORR_INDEX: compute_projection(corrected[s], cor_cn, cor_ad),
        }))
    return pd.concat(rows, ignore_index=True)


# ---------- statistics ----------

def safe_auc(index_values, is_ad):
    if len(np.unique(is_ad)) < 2:
        return float("nan")
    return float(roc_auc_score(is_ad, index_values))


def safe_spearman(x, y):
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    r, p = spearmanr(x, y)
    return float(r), float(p)


def cohens_d(cn, ad):
    if len(cn) < 2 or len(ad) < 2:
        return float("nan")
    pooled = ((len(cn) - 1) * cn.var(ddof=1) + (len(ad) - 1) * ad.var(ddof=1)) / (len(cn) + len(ad) - 2)
    return float((ad.mean() - cn.mean()) / np.sqrt(pooled)) if pooled > 0 else float("nan")


def subject_bootstrap_ci(df, statistic, n_boot=N_BOOTSTRAP, seed=SEED):
    """95% CI of statistic(df), resampling subjects (adni_id) with all their visits."""
    if len(df) < 10:
        return float("nan"), float("nan")
    groups = df.groupby("adni_id").indices
    subjects = np.array(list(groups))
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        rows = np.concatenate([groups[s] for s in rng.choice(subjects, len(subjects))])
        value = statistic(df.iloc[rows])
        if np.isfinite(value):
            stats.append(value)
    if not stats:
        return float("nan"), float("nan")
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def paired_subset(df, index_name, target):
    return df.dropna(subset=[index_name, target])


def correlate_target(df, index_name, target):
    if target not in df.columns:
        return {}
    out = {}
    paired = paired_subset(df, index_name, target)

    def spearman_of(frame):
        return safe_spearman(frame[index_name].to_numpy(), frame[target].to_numpy())[0]

    for label, frame in [("", paired),
                         ("_ad_only", paired[paired["diagnosis"] == "ad"]),
                         ("_cn_only", paired[paired["diagnosis"] == "cn"])]:
        r, p = safe_spearman(frame[index_name].to_numpy(), frame[target].to_numpy())
        lo, hi = subject_bootstrap_ci(frame, spearman_of)
        out[f"n_{target}{label}"] = len(frame)
        out[f"spearman_r_{target}{label}"] = r
        out[f"spearman_p_{target}{label}"] = p
        out[f"spearman_ci_lo_{target}{label}"] = lo
        out[f"spearman_ci_hi_{target}{label}"] = hi
    return out


def validate_index(df, index_name, split):
    """One row of statistics for one index on one split. df needs adni_id."""
    subset = df.dropna(subset=[index_name])
    cn = subset.loc[subset["diagnosis"] == "cn", index_name]
    ad = subset.loc[subset["diagnosis"] == "ad", index_name]

    def auc_of(frame):
        return safe_auc(frame[index_name].to_numpy(), (frame["diagnosis"] == "ad").to_numpy().astype(int))

    auc_lo, auc_hi = subject_bootstrap_ci(subset, auc_of)
    mw_p = (mannwhitneyu(ad, cn, alternative="two-sided")[1]
            if len(cn) >= 3 and len(ad) >= 3 else float("nan"))

    row = {
        "index_name": index_name,
        "split": split,
        "primary": index_name == PRIMARY_INDEX and split == PRIMARY_SPLIT,
        "n": len(subset),
        "n_subjects": subset["adni_id"].nunique(),
        "n_cn": len(cn),
        "n_ad": len(ad),
        "cn_mean": cn.mean(),
        "cn_std": cn.std(),
        "ad_mean": ad.mean(),
        "ad_std": ad.std(),
        "cohens_d": cohens_d(cn, ad),
        "auc": auc_of(subset),
        "auc_ci_lo": auc_lo,
        "auc_ci_hi": auc_hi,
        "mannwhitney_p": mw_p,
    }
    for target in TARGETS:
        row.update(correlate_target(subset, index_name, target))
    return row


def validate_all(index_table):
    """Statistics for every index version on every split."""
    return pd.DataFrame([validate_index(index_table[index_table["split"] == s], name, s)
                         for s in SPLITS for name in INDEX_COLUMNS])


def primary_summary(validation):
    """The pre-specified main result, as one printable line."""
    row = validation[validation["primary"]].iloc[0]
    return (f"MAIN RESULT ({PRIMARY_INDEX}, {PRIMARY_SPLIT}): MMSE within AD "
            f"r={row[PRIMARY_ENDPOINT]:.3f} "
            f"[{row['spearman_ci_lo_mmse_ad_only']:.3f}, {row['spearman_ci_hi_mmse_ad_only']:.3f}], "
            f"n={int(row['n_mmse_ad_only'])} | AUC={row['auc']:.3f} "
            f"[{row['auc_ci_lo']:.3f}, {row['auc_ci_hi']:.3f}]")