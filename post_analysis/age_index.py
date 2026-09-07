import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

RUN_DIR = Path("/home/minhtri/Molecular_AD/checkpoints/2026-08-16_13-48-33")
DEMO_PATH = Path("/home/minhtri/Molecular_AD/data/all_demographics.csv")

LATENT_PATH = RUN_DIR / "post_analysis" / "latents_best_ckpt.npz"
OUT_PATH = RUN_DIR / "post_analysis" / "disease_index_agecorrected.csv"
CMP_PATH = RUN_DIR / "post_analysis" / "agecorrection_comparison.csv"
FIG_DIR = RUN_DIR / "post_analysis" / "figures_agecorrection"

TARGETS = ["mmse", "ptau217_alzpath", "ptau217_janssen"]
SPLITS = ["train", "val", "test"]

CN_COLOR = "#3B7EA1"
AD_COLOR = "#E08A2E"

data = np.load(LATENT_PATH, allow_pickle=True)


def get(split, key):
    return data[f"{split}_{key}"]


train_mu = get("train", "mu").astype(float)
train_age = get("train", "age").astype(float)
train_diag = np.char.lower(get("train", "diagnoses").astype(str))

fit_mask = (train_diag == "cn") & np.isfinite(train_age)
print(f"aging model fitted on {fit_mask.sum()} train CN subjects")
print(f"  age {train_age[fit_mask].min():.0f}-{train_age[fit_mask].max():.0f}, "
      f"mean {train_age[fit_mask].mean():.1f}")

A = np.column_stack([np.ones(fit_mask.sum()), train_age[fit_mask]])
coef, *_ = np.linalg.lstsq(A, train_mu[fit_mask], rcond=None)

pred_cn = A @ coef
ss_res = ((train_mu[fit_mask] - pred_cn) ** 2).sum(axis=0)
ss_tot = ((train_mu[fit_mask] - train_mu[fit_mask].mean(axis=0)) ** 2).sum(axis=0)
r2 = 1 - ss_res / np.where(ss_tot > 0, ss_tot, 1)
print(f"  age R2 per dim: median {np.median(r2):.3f}, max {r2.max():.3f}, "
      f"dims above 0.10: {(r2 > 0.10).sum()}/{len(r2)}")


def correct(mu, age):
    out = mu.copy()
    ok = np.isfinite(age)
    if ok.any():
        A_ = np.column_stack([np.ones(ok.sum()), age[ok]])
        out[ok] = mu[ok] - A_ @ coef
    return out, ok


def project(mu, cn_mean, ad_mean):
    direction = ad_mean - cn_mean
    return ((mu - cn_mean) @ direction) / (direction @ direction)


mu_orig, mu_corr, age_ok = {}, {}, {}
for split in SPLITS:
    m = get(split, "mu").astype(float)
    a = get(split, "age").astype(float)
    mu_orig[split] = m
    mu_corr[split], age_ok[split] = correct(m, a)


def direction_means(mu_train, diag_train):
    return (mu_train[diag_train == "cn"].mean(axis=0),
            mu_train[diag_train == "ad"].mean(axis=0))


cn_o, ad_o = direction_means(mu_orig["train"], train_diag)
cn_c, ad_c = direction_means(mu_corr["train"], train_diag)

cos = float(np.dot(ad_o - cn_o, ad_c - cn_c) /
            (np.linalg.norm(ad_o - cn_o) * np.linalg.norm(ad_c - cn_c)))
print(f"  cosine(original axis, corrected axis) = {cos:.3f}")

rows = []
for split in SPLITS:
    diag = np.char.lower(get(split, "diagnoses").astype(str))
    rows.append(pd.DataFrame({
        "split": split,
        "image_id": get(split, "image_ids").astype(str),
        "diagnosis": diag,
        "age": get(split, "age").astype(float),
        "age_available": age_ok[split],
        "index_orig": project(mu_orig[split], cn_o, ad_o),
        "index_corr": project(mu_corr[split], cn_c, ad_c),
    }))

df = pd.concat(rows, ignore_index=True)

demo = pd.read_csv(DEMO_PATH, low_memory=False)
demo["image_id"] = demo["image_id"].astype(str)
keep = ["image_id"] + [t for t in TARGETS if t in demo.columns]
df = df.merge(demo[keep], on="image_id", how="left")

os.makedirs(OUT_PATH.parent, exist_ok=True)
df.to_csv(OUT_PATH, index=False)


def auc(sub, col):
    s = sub.dropna(subset=[col])
    y = (s["diagnosis"] == "ad").astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, s[col].to_numpy())


def rho(sub, col, target):
    s = sub.dropna(subset=[col, target])
    if len(s) < 10:
        return np.nan, np.nan
    r, p = spearmanr(s[col], s[target])
    return float(r), float(p)


comparison = []
for split in SPLITS:
    sub = df[df["split"] == split]
    for group in ["all", "cn", "ad"]:
        g = sub if group == "all" else sub[sub["diagnosis"] == group]
        if len(g) < 10:
            continue

        row = {"split": split, "group": group, "n": len(g)}

        if group == "all":
            row["auc_orig"] = auc(g, "index_orig")
            row["auc_corr"] = auc(g, "index_corr")

        r_o, _ = rho(g, "index_orig", "age")
        r_c, _ = rho(g, "index_corr", "age")
        row["age_orig"], row["age_corr"] = r_o, r_c

        for target in TARGETS:
            if target not in g.columns:
                continue
            r_o, p_o = rho(g, "index_orig", target)
            r_c, p_c = rho(g, "index_corr", target)
            if np.isnan(r_o) and np.isnan(r_c):
                continue
            row[f"{target}_orig"] = r_o
            row[f"{target}_corr"] = r_c
            row[f"{target}_p_corr"] = p_c

        comparison.append(row)

cmp_df = pd.DataFrame(comparison)
cmp_df.to_csv(CMP_PATH, index=False)

pd.set_option("display.width", 250)
print("\nORIGINAL vs AGE-CORRECTED")
print(cmp_df.round(4).to_string(index=False))

os.makedirs(FIG_DIR, exist_ok=True)
test = df[df["split"] == "test"]


def scatter(ax, sub, xcol, ycol, title):
    for grp, color, label in [("cn", CN_COLOR, "CN"), ("ad", AD_COLOR, "AD")]:
        g = sub[sub["diagnosis"] == grp].dropna(subset=[xcol, ycol])
        ax.scatter(g[xcol], g[ycol], s=42, alpha=0.75, color=color,
                   edgecolor="white", linewidth=0.6, label=f"{label} (n={len(g)})")
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)


def fit_line(ax, sub, xcol, ycol, color):
    g = sub.dropna(subset=[xcol, ycol])
    if len(g) < 5:
        return
    b = np.polyfit(g[xcol], g[ycol], 1)
    xs = np.linspace(g[xcol].min(), g[xcol].max(), 50)
    ax.plot(xs, np.polyval(b, xs), color=color, linewidth=1.6, alpha=0.9)


for target in ["ptau217_alzpath", "ptau217_janssen", "mmse"]:
    if target not in test.columns:
        continue
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, col, tag in [(axes[0], "index_orig", "uncorrected"),
                         (axes[1], "index_corr", "age-corrected")]:
        ad = test[test["diagnosis"] == "ad"]
        r_ad, p_ad = rho(ad, col, target)
        r_all, p_all = rho(test, col, target)
        scatter(ax, test, col, target,
                f"{tag}\nall r={r_all:.3f} p={p_all:.2g}  |  AD only r={r_ad:.3f} p={p_ad:.2g}")
        fit_line(ax, ad, col, target, AD_COLOR)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"index_vs_{target}_before_after.png", dpi=200)
    plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=False)
for ax, col, tag in [(axes[0], "index_orig", "uncorrected"),
                     (axes[1], "index_corr", "age-corrected")]:
    r_all, p_all = rho(test, col, "age")
    r_cn, _ = rho(test[test["diagnosis"] == "cn"], col, "age")
    r_ad, _ = rho(test[test["diagnosis"] == "ad"], col, "age")
    scatter(ax, test, "age", col,
            f"{tag}\nall r={r_all:.3f}  |  CN r={r_cn:.3f}  |  AD r={r_ad:.3f}")
    fit_line(ax, test, "age", col, "#555555")
fig.tight_layout()
fig.savefig(FIG_DIR / "index_vs_age_before_after.png", dpi=200)
plt.close(fig)

metrics = [
    ("index vs age", "age_orig", "age_corr"),
    ("MMSE (AD only)", "mmse_orig", "mmse_corr"),
    ("pTau alzpath (AD only)", "ptau217_alzpath_orig", "ptau217_alzpath_corr"),
    ("pTau janssen (AD only)", "ptau217_janssen_orig", "ptau217_janssen_corr"),
]
src = {
    "index vs age": cmp_df[(cmp_df.split == "test") & (cmp_df.group == "all")],
    "MMSE (AD only)": cmp_df[(cmp_df.split == "test") & (cmp_df.group == "ad")],
    "pTau alzpath (AD only)": cmp_df[(cmp_df.split == "test") & (cmp_df.group == "ad")],
    "pTau janssen (AD only)": cmp_df[(cmp_df.split == "test") & (cmp_df.group == "ad")],
}

labels, before, after = [], [], []
for name, ocol, ccol in metrics:
    row = src[name]
    if len(row) == 0 or ocol not in row.columns:
        continue
    labels.append(name)
    before.append(float(row[ocol].iloc[0]))
    after.append(float(row[ccol].iloc[0]))

fig, ax = plt.subplots(figsize=(8, 4.4))
y = np.arange(len(labels))
ax.barh(y - 0.19, np.abs(before), height=0.36, color="#B0B7BE", label="uncorrected")
ax.barh(y + 0.19, np.abs(after), height=0.36, color="#3B7EA1", label="age-corrected")
for i, (b, a) in enumerate(zip(before, after)):
    ax.text(abs(b) + 0.01, i - 0.19, f"{b:.3f}", va="center", fontsize=9)
    ax.text(abs(a) + 0.01, i + 0.19, f"{a:.3f}", va="center", fontsize=9)
ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.set_xlabel("|Spearman r|, test split")
ax.invert_yaxis()
ax.legend(frameon=False, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "agecorrection_summary.png", dpi=200)
plt.close(fig)

print(f"\nsaved {OUT_PATH}")
print(f"saved {CMP_PATH}")
print(f"saved figures to {FIG_DIR}")