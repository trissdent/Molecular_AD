import re

import numpy as np
import pandas as pd

IN_CSV = "/home/minhtri/Molecular_AD/data/mri_features.csv"
OUT_DIR = "/home/minhtri/Molecular_AD/data/"
CORR_THRESHOLD = 0.99
ETIV_COL = "aseg_EstimatedTotalIntraCranialVol"

PARSER_ARTIFACT = re.compile(r"_aparc_|Measure:")
RATIO_EXEMPT = re.compile(r"-to-eTIV")

TECHNICAL = [
    "aseg_lhSurfaceHoles",
    "aseg_rhSurfaceHoles",
    "aseg_SurfaceHoles",
    "aseg_MaskVol",
    "aseg_MaskVol-to-eTIV",
    "wmparc_Left-UnsegmentedWhiteMatter_volume",
    "wmparc_Right-UnsegmentedWhiteMatter_volume",
]

AGGREGATE = [
    "aseg_BrainSegVol",
    "aseg_BrainSegVolNotVent",
    "aseg_lhCortexVol_volume",
    "aseg_rhCortexVol_volume",
    "aseg_CortexVol_volume",
    "aseg_lhCerebralWhiteMatterVol_volume",
    "aseg_rhCerebralWhiteMatterVol_volume",
    "aseg_CerebralWhiteMatterVol_volume",
    "aseg_SubCortGrayVol_volume",
    "aseg_TotalGrayVol_volume",
    "aseg_SupraTentorialVol",
    "aseg_SupraTentorialVolNotVent",
    "aseg_SegmentedTotalIntracranialVol_volume",
    "lh_WhiteSurfArea_area",
    "rh_WhiteSurfArea_area",
    "lh_MeanThickness_thickness",
    "rh_MeanThickness_thickness",
    "wmparc_lhCerebralWhiteMatterVol_volume",
    "wmparc_rhCerebralWhiteMatterVol_volume",
    "wmparc_CerebralWhiteMatterVol_volume",
    "wmparc_SegmentedTotalIntracranialVol_volume",
    "lh_hippo_Whole_hippocampus_volume",
    "rh_hippo_Whole_hippocampus_volume",
    "lh_hippo_Whole_hippocampal_head_volume",
    "rh_hippo_Whole_hippocampal_head_volume",
    "lh_hippo_Whole_hippocampal_body_volume",
    "rh_hippo_Whole_hippocampal_body_volume",
    "lh_amyg_Whole_amygdala_volume",
    "rh_amyg_Whole_amygdala_volume",
    "aseg_Left-Hippocampus_volume",
    "aseg_Right-Hippocampus_volume",
    "aseg_Left-Amygdala_volume",
    "aseg_Right-Amygdala_volume",
]


def is_volume(col):
    return (col.endswith("_volume") or "Vol" in col) and not RATIO_EXEMPT.search(col)


def residualize(values, etiv):
    x = np.column_stack([np.ones(len(etiv)), etiv])
    beta, *_ = np.linalg.lstsq(x, values, rcond=None)
    return values - x @ beta


if __name__ == "__main__":
    df = pd.read_csv(IN_CSV)
    df["image_id"] = df["image_id"].astype(str)
    meta = df[["subject_id", "image_id"]].copy()
    etiv = df[ETIV_COL].to_numpy(dtype=float)

    feats = df.drop(columns=["subject_id", "image_id"]).select_dtypes(include=[np.number])
    reason = {}

    for c in feats.columns:
        if feats[c].std() == 0:
            reason[c] = "zero_variance"
        elif c.endswith("_meanintensity"):
            reason[c] = "mean_intensity"
        elif PARSER_ARTIFACT.search(c):
            reason[c] = "parser_artifact"
        elif c in TECHNICAL:
            reason[c] = "technical"
        # elif c in AGGREGATE:
        #     reason[c] = "aggregate"
        elif c == ETIV_COL:
            reason[c] = "etiv_helper"

    kept = [c for c in feats.columns if c not in reason]
    sub = feats[kept]

    corr = sub.corr().abs().to_numpy().copy()
    np.fill_diagonal(corr, 0.0)
    dropped_corr = set()
    for i in range(len(kept)):
        if kept[i] in dropped_corr:
            continue
        for j in range(i + 1, len(kept)):
            if kept[j] in dropped_corr:
                continue
            if corr[i, j] > CORR_THRESHOLD:
                dropped_corr.add(kept[j])
                reason[kept[j]] = f"corr_dup_with:{kept[i]}"

    pool = [c for c in kept if c not in dropped_corr]
    vol_cols = [c for c in pool if is_volume(c)]
    etiv_s = pd.Series(etiv, index=sub.index)

    raw = sub[pool].copy()

    ratio = raw.copy()
    ratio[vol_cols] = ratio[vol_cols].div(etiv, axis=0)

    resid = raw.copy()
    resid[vol_cols] = residualize(resid[vol_cols].to_numpy(dtype=float), etiv)

    for name, frame in [("raw", raw), ("ratio", ratio), ("residual", resid)]:
        out = pd.concat([meta, frame, pd.Series(etiv, name=ETIV_COL)], axis=1)
        out.to_csv(f"{OUT_DIR}/pool_{name}.csv", index=False)

    etiv_r = raw.corrwith(etiv_s).abs()
    report = pd.DataFrame({
        "column": feats.columns,
        "kept": [c in pool for c in feats.columns],
        "drop_reason": [reason.get(c, "") for c in feats.columns],
        "is_volume": [is_volume(c) for c in feats.columns],
        "abs_corr_etiv_raw": [round(float(etiv_r[c]), 4) if c in etiv_r.index else np.nan for c in feats.columns],
    })
    report.to_csv(f"{OUT_DIR}/pool_report.csv", index=False)

    print(f"input columns   {feats.shape[1]}")
    print(f"pool columns    {len(pool)}")
    print(f"volume columns  {len(vol_cols)}\n")
    print(report.loc[report.drop_reason != "", "drop_reason"].str.split(":").str[0].value_counts().to_string())
    print("\nmean |r| with eTIV, volume columns")
    for name, frame in [("raw", raw), ("ratio", ratio), ("residual", resid)]:
        print(f"  {name:9s} {frame[vol_cols].corrwith(etiv_s).abs().mean():.4f}")