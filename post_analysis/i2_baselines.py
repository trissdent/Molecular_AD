"""Simple comparison indices, built and validated exactly like the CALIBER index (s2).

    python post_analysis/i2_baselines.py        (PREPARED_DIR=... to override the data folder)

Baselines (all fitted on the training split only):
    hippocampus   left + right aseg hippocampal volume, adjusted for eTIV
                  (linear fit on training CN), as a 1-D representation
    pca_smri      features standardized on train -> PCA fitted on train
    pca_dti       same, DTI features
    pca_both      same, sMRI + DTI features side by side
Each representation then goes through i1_index_core: age correction on training CN,
CN -> AD projection, statistics with subject-level bootstrap CIs.

No neural network is trained here. Outputs in results/baselines/:
    baseline_validation.csv   all statistics, every baseline x index version x split
    baseline_summary.csv      test split, age-corrected index: the comparison table
    baseline_index.csv        every visit's index per baseline (same columns as a run's
                              s2_index/disease_index.csv, plus 'baseline'), for m2_compare_models
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from i1_index_core import (
    AGECORR_INDEX,
    PRIMARY_SPLIT,
    SPLITS,
    TARGETS,
    build_index_table,
    drop_excluded,
    primary_summary,
    validate_all,
)
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPARED_DIR = Path(os.environ.get("PREPARED_DIR", PROJECT_ROOT / "data" / "prepared"))
OUT_DIR = PROJECT_ROOT / "results" / "baselines"

PCA_COMPONENTS = 128        # same size as the CALIBER latent (model.z_dim)
HIPPOCAMPUS = ["aseg_Left-Hippocampus_volume", "aseg_Right-Hippocampus_volume"]


def load_inputs():
    cohort = pd.read_csv(PREPARED_DIR / "cohort.csv", dtype={"visit_id": str, "adni_id": str})
    split = pd.read_csv(PREPARED_DIR / "split.csv", dtype={"visit_id": str})
    etiv = pd.read_csv(PREPARED_DIR / "etiv.csv", dtype={"visit_id": str})
    visits = (split[["visit_id", "split"]]
              .merge(cohort, on="visit_id", how="left")
              .merge(etiv, on="visit_id", how="left"))
    feats = {}
    for m in ("smri", "dti"):
        df = pd.read_csv(PREPARED_DIR / f"features_{m}.csv", dtype={"visit_id": str})
        feats[m] = df.drop(columns=[f"{m}_image_id"]).set_index("visit_id")
    return visits, feats


def split_frames(visits):
    return {s: visits[visits["split"] == s].reset_index(drop=True) for s in SPLITS}


def hippocampus_reps(frames, smri):
    missing = [c for c in HIPPOCAMPUS if c not in smri.columns]
    if missing:
        raise KeyError(f"{missing} not in features_smri.csv (check features_smri_report.csv)")
    volume = {s: smri.loc[f["visit_id"], HIPPOCAMPUS].sum(axis=1).to_numpy(dtype=float)
              for s, f in frames.items()}

    train = frames["train"]
    cn = (train["diagnosis"] == "cn").to_numpy()
    design = np.column_stack([np.ones(cn.sum()), train.loc[cn, "etiv"].to_numpy(dtype=float)])
    coef, *_ = np.linalg.lstsq(design, volume["train"][cn], rcond=None)

    reps = {}
    for s, f in frames.items():
        etiv = f["etiv"].to_numpy(dtype=float)
        reps[s] = (volume[s] - (coef[0] + coef[1] * etiv))[:, None]
    return reps


def pca_reps(frames, features):
    train = features.loc[frames["train"]["visit_id"]].to_numpy(dtype=float)
    mean, std = train.mean(axis=0), train.std(axis=0)
    std[std == 0] = 1.0
    n = min(PCA_COMPONENTS, train.shape[1], train.shape[0])
    pca = PCA(n_components=n, random_state=0).fit((train - mean) / std)
    return {s: pca.transform((features.loc[f["visit_id"]].to_numpy(dtype=float) - mean) / std)
            for s, f in frames.items()}


def evaluate(name, reps, frames):
    meta = {s: f[["visit_id", "diagnosis", "age"]] for s, f in frames.items()}
    table = build_index_table(reps, meta)
    keep = ["visit_id", "adni_id"] + [t for t in TARGETS if t in frames["train"].columns]
    table = drop_excluded(table.merge(pd.concat(frames.values())[keep], on="visit_id", how="left"))
    validation = validate_all(table)
    validation.insert(0, "baseline", name)
    table.insert(0, "baseline", name)
    print(f"{name:12s} {primary_summary(validation)}")
    return validation, table


def summary_table(validation):
    test = validation[(validation["split"] == PRIMARY_SPLIT) & (validation["index_name"] == AGECORR_INDEX)]

    def fmt(r, lo, hi):
        return f"{r:.3f} [{lo:.3f}, {hi:.3f}]" if np.isfinite(r) else "n/a"

    rows = []
    for _, row in test.iterrows():
        out = {"baseline": row["baseline"], "n": int(row["n"]),
               "AUC": fmt(row["auc"], row["auc_ci_lo"], row["auc_ci_hi"])}
        for target in TARGETS:
            key = f"{target}_ad_only"
            out[f"{target} (AD)"] = fmt(row[f"spearman_r_{key}"], row[f"spearman_ci_lo_{key}"],
                                        row[f"spearman_ci_hi_{key}"])
            out[f"n {target} (AD)"] = int(row[f"n_{key}"])
        rows.append(out)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    visits, feats = load_inputs()
    frames = split_frames(visits)
    both = pd.concat([feats["smri"], feats["dti"]], axis=1, join="inner")

    results = [
        evaluate("hippocampus", hippocampus_reps(frames, feats["smri"]), frames),
        evaluate("pca_smri", pca_reps(frames, feats["smri"]), frames),
        evaluate("pca_dti", pca_reps(frames, feats["dti"]), frames),
        evaluate("pca_both", pca_reps(frames, both), frames),
    ]
    validation = pd.concat([v for v, _ in results], ignore_index=True)
    validation.to_csv(OUT_DIR / "baseline_validation.csv", index=False)
    pd.concat([t for _, t in results], ignore_index=True).to_csv(
        OUT_DIR / "baseline_index.csv", index=False)

    summary = summary_table(validation)
    summary.to_csv(OUT_DIR / "baseline_summary.csv", index=False)
    print("\n" + summary.to_string(index=False))
    print(f"\nSaved: {OUT_DIR / 'baseline_validation.csv'}")
    print(f"Saved: {OUT_DIR / 'baseline_summary.csv'}")
    print(f"Saved: {OUT_DIR / 'baseline_index.csv'}")