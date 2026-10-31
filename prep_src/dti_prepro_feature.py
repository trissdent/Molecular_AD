import numpy as np
import pandas as pd

IN_CSV = "/home/minhtri/Molecular_AD/data/dti_features.csv"
OUT_CSV = "/home/minhtri/Molecular_AD/data/pool_dti.csv"
REPORT_CSV = "/home/minhtri/Molecular_AD/data/pool_dti_report.csv"
CORR_THRESHOLD = 0.99

DICT_CSV = "/home/minhtri/Molecular_AD/data/dti_columns.csv"

COMBINED_NAMES = {
    "WholeSkeleton": "Whole skeleton",
    "IC-R": "Internal capsule R", "IC-L": "Internal capsule L", "IC": "Internal capsule",
    "ALIC": "Anterior limb of internal capsule", "PLIC": "Posterior limb of internal capsule",
    "RLIC": "Retrolenticular part of internal capsule",
    "CR-R": "Corona radiata R", "CR-L": "Corona radiata L", "CR": "Corona radiata",
    "ACR": "Anterior corona radiata", "SCR": "Superior corona radiata", "PCR": "Posterior corona radiata",
    "CC": "Corpus callosum", "CST": "Corticospinal tract", "PTR": "Posterior thalamic radiation",
    "SS": "Sagittal stratum", "EC": "External capsule",
    "CGC": "Cingulum (cingulate gyrus)", "CGH": "Cingulum (hippocampus)",
    "SLF": "Superior longitudinal fasciculus", "SFO": "Superior fronto-occipital fasciculus",
    "IFO": "Inferior fronto-occipital fasciculus", "FXST": "Fornix (cres) / Stria terminalis",
    "UNC": "Uncinate fasciculus",
}


def readable_names(columns):
    d = pd.read_csv(DICT_CSV).set_index("column")
    names = {}
    for c in columns:
        m, roi, kind = d.loc[c, "map"], d.loc[c, "roi"], d.loc[c, "kind"]
        if kind == "tract":
            desc = d.loc[c, "description"].split(": ", 1)[1]
        else:
            desc = COMBINED_NAMES[roi]
        names[c] = f"{m} {desc} ({roi})"
    assert len(set(names.values())) == len(names)
    return names

if __name__ == "__main__":
    df = pd.read_csv(IN_CSV)
    df["image_id"] = df["image_id"].astype(str)
    meta = df[["subject_id", "image_id"]]
    feats = df.drop(columns=["subject_id", "image_id"]).select_dtypes(include=[np.number])

    reason = {c: "zero_variance" for c in feats.columns if feats[c].std() == 0}
    kept = [c for c in feats.columns if c not in reason]

    corr = feats[kept].corr().abs().to_numpy().copy()
    np.fill_diagonal(corr, 0.0)
    for i in range(len(kept)):
        if kept[i] in reason:
            continue
        for j in range(i + 1, len(kept)):
            if kept[j] not in reason and corr[i, j] > CORR_THRESHOLD:
                reason[kept[j]] = f"corr_dup_with:{kept[i]}"

    pool = [c for c in feats.columns if c not in reason]
    pd.concat([meta, feats[pool].rename(columns=readable_names(pool))], axis=1).to_csv(OUT_CSV, index=False)

    pd.DataFrame({
        "column": feats.columns,
        "kept": [c in pool for c in feats.columns],
        "drop_reason": [reason.get(c, "") for c in feats.columns],
    }).to_csv(REPORT_CSV, index=False)

    print(f"input columns  {feats.shape[1]}")
    print(f"pool columns   {len(pool)}")
    for c, r in reason.items():
        print(f"  {c}: {r}")