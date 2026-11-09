"""Collect one setting's seed runs into one table, next to the matching baselines.

    python post_analysis/m1_collect_seeds.py "smri_pw0.5_off*"            (test split)
    python post_analysis/m1_collect_seeds.py "smri_pw0.5_off*" val
    python post_analysis/m1_collect_seeds.py "smri_pw0.5_s*" "smri_pw0.5_2026*"   (several patterns)

PATTERN is a glob over checkpoints/ (quote it); several patterns are merged, and the last
argument may be the split (train / val / test, default test). Reads each run's s2_index/index_validation.csv
and the seed from logs/<run>/config.yml; the baselines come from results/baselines/
baseline_validation.csv (run i2_baselines.py first), chosen by the runs' modality.

Outputs in results/seeds/ (named after the first PATTERN without '*'):
    <name>_<split>.csv           one row per run x index version
    <name>_<split>_summary.csv   mean, SD, min, max over seeds; seeds with p < 0.05
    <name>_test_adjusted.csv     (test only, runs with s4 output) within-AD Spearman before and
                                 after adjusting for APOE e4, age and sex, per run x index x target
"""

import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = PROJECT_ROOT / "checkpoints"
LOGS = PROJECT_ROOT / "logs"
BASELINE_CSV = PROJECT_ROOT / "results" / "baselines" / "baseline_validation.csv"
OUT_DIR = PROJECT_ROOT / "results" / "seeds"

COLS = {
    "n": "n",
    "auc": "AUC",
    "cohens_d": "d",
    "spearman_r_mmse_ad_only": "MMSE in AD r",
    "spearman_p_mmse_ad_only": "MMSE in AD p",
    "spearman_r_ptau217_alzpath": "pTau AlzPath r",
    "spearman_r_ptau217_alzpath_ad_only": "pTau AlzPath in AD r",
    "spearman_p_ptau217_alzpath_ad_only": "pTau AlzPath in AD p",
    "spearman_r_ptau217_janssen": "pTau Janssen r",
    "spearman_r_ptau217_janssen_ad_only": "pTau Janssen in AD r",
    "spearman_p_ptau217_janssen_ad_only": "pTau Janssen in AD p",
}
P_COLS = [c for c in COLS.values() if c.endswith(" p")]
VALUE_COLS = [c for c in COLS.values() if c not in P_COLS and c != "n"]

# baselines that use the same input information as each modality
BASELINES = {
    "smri": ["hippocampus", "pca_smri"],
    "dti": ["pca_dti"],
    "fa_md": ["pca_dti"],
    "both": ["hippocampus", "pca_both"],
    "t1_fa_md": ["hippocampus", "pca_both"],
}


def collect(patterns, split):
    rows = []
    runs = sorted({run for pattern in patterns for run in CHECKPOINTS.glob(pattern)})
    for run in runs:
        csv = run / "s2_index" / "index_validation.csv"
        if not csv.is_file():
            print(f"skip {run.name}: no s2_index/index_validation.csv")
            continue
        config = yaml.safe_load((LOGS / run.name / "config.yml").read_text())
        df = pd.read_csv(csv)
        for _, r in df[df["split"] == split].iterrows():
            rows.append({"run": run.name, "seed": config["training"]["seed"],
                         "modality": config["data"]["modality"], "index": r["index_name"],
                         **{name: r[col] for col, name in COLS.items()}})
    if not rows:
        sys.exit(f"no finished runs match {patterns} in {CHECKPOINTS}")
    return pd.DataFrame(rows).sort_values(["index", "seed"]).reset_index(drop=True)


def summarize(seeds):
    out = []
    for index, g in seeds.groupby("index"):
        row = {"index": index, "n_seeds": len(g), "n": "/".join(map(str, sorted(g["n"].unique())))}
        for c in VALUE_COLS:
            row[f"{c} mean"] = g[c].mean()
            row[f"{c} sd"] = g[c].std()
            row[f"{c} min"] = g[c].min()
            row[f"{c} max"] = g[c].max()
        for c in P_COLS:
            row[c.replace(" p", " p<0.05")] = f"{int((g[c] < 0.05).sum())}/{len(g)}"
        out.append(row)
    return pd.DataFrame(out)


def collect_adjusted(runs):
    """s4's within-AD associations (test split) for the runs that have them."""
    frames = []
    for run, seed in runs:
        csv = CHECKPOINTS / run / "s4_apoe" / "apoe_partial_model.csv"
        if not csv.is_file():
            continue
        df = pd.read_csv(csv)
        if "adjusted_r" not in df.columns:
            print(f"skip {run}: s4 output predates the age/sex adjustment, rerun s4_apoe.py")
            continue
        frames.append(df.assign(run=run, seed=seed)[
            ["run", "seed", "index_name", "target", "n", "simple_r", "simple_p",
             "adjusted_r", "adjusted_p"]])
    return pd.concat(frames, ignore_index=True) if frames else None


def fmt(df):
    """Values to 3 decimals, p-values in scientific notation (so small p does not show as 0)."""
    out = df.copy()
    for c in out.columns:
        if c in P_COLS:
            out[c] = out[c].map(lambda v: f"{v:.1e}")
        elif out[c].dtype.kind == "f":
            out[c] = out[c].round(3)
    return out


def baselines(modalities, split):
    if not BASELINE_CSV.is_file():
        print(f"no {BASELINE_CSV}: run post_analysis/i2_baselines.py for the comparison")
        return None
    names = sorted({b for m in modalities for b in BASELINES.get(m, [])})
    base = pd.read_csv(BASELINE_CSV)
    base = base[(base["split"] == split) & base["baseline"].isin(names)]
    return base.rename(columns=COLS)[["baseline", "index_name", *COLS.values()]]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    args = sys.argv[1:]
    split = args.pop() if args[-1] in ("train", "val", "test") else "test"
    if not args:
        sys.exit(__doc__)

    seeds = collect(args, split)
    summary = summarize(seeds)
    modalities = seeds["modality"].unique().tolist()
    if len(modalities) > 1:
        print(f"WARNING: the pattern matches several modalities: {modalities}")

    pd.set_option("display.width", 200)
    print(f"\n{' '.join(args)} | split {split} | {len(seeds['run'].unique())} runs\n")
    print(fmt(seeds.drop(columns=["modality"])).to_string(index=False))

    short = ["index", "n_seeds", "n"] + [f"{c} mean" for c in VALUE_COLS] \
        + [c.replace(" p", " p<0.05") for c in P_COLS]
    print("\nmean over seeds\n" + summary[short].round(3).to_string(index=False))

    base = baselines(modalities, split)
    if base is not None and len(base):
        print("\nbaselines\n" + fmt(base).to_string(index=False))
        if set(base["n"]) != set(seeds["n"]):
            print(f"WARNING: visit counts differ (runs {sorted(set(seeds['n']))}, "
                  f"baselines {sorted(set(base['n']))}); rerun i2_baselines.py")

    adjusted = None
    if split == "test":
        adjusted = collect_adjusted(seeds[["run", "seed"]].drop_duplicates().itertuples(index=False))
    if adjusted is not None:
        table = adjusted.groupby(["index_name", "target"]).agg(
            n_seeds=("seed", "size"), n=("n", "median"),
            simple_r=("simple_r", "mean"),
            simple_sig=("simple_p", lambda p: f"{int((p < 0.05).sum())}/{len(p)}"),
            adjusted_r=("adjusted_r", "mean"),
            adjusted_sig=("adjusted_p", lambda p: f"{int((p < 0.05).sum())}/{len(p)}"),
        ).reset_index()
        print("\nwithin AD, adjusted for APOE e4 + age + sex (mean over seeds)\n"
              + table.round(3).to_string(index=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = args[0].replace("*", "").rstrip("_") or "runs"
    if adjusted is not None:
        adjusted.to_csv(OUT_DIR / f"{name}_test_adjusted.csv", index=False)
        print(f"Saved: {OUT_DIR / f'{name}_test_adjusted.csv'}")
    seeds.to_csv(OUT_DIR / f"{name}_{split}.csv", index=False)
    summary.to_csv(OUT_DIR / f"{name}_{split}_summary.csv", index=False)
    print(f"\nSaved: {OUT_DIR / f'{name}_{split}.csv'}")
    print(f"Saved: {OUT_DIR / f'{name}_{split}_summary.csv'}")