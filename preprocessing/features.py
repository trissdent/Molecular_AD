"""Clean the feature tables (the prediction targets) for each modality.

Same rules for both:
    1. cohort visits only
    2. drop a column empty in more than features.max_missing of visits
       (visits with gaps are listed, not dropped)
    3. drop zero-variance columns
    4. drop near-duplicates (|r| > features.corr_threshold), keeping the first
Modality rules (read from the extraction's column dictionary):
    sMRI  keep measures volume / area / thickness / meancurv,
          drop aggregates (whole-brain measures, Whole_* structures),
          drop technical regions, keep eTIV out (saved to etiv.csv)
    DTI   keep single tracts only (kind == tract), maps in features.dti.maps,
          kept columns ordered by that maps list
          features.dti.extra_tables {name: {maps, gray_matter}} writes more DTI tables
          (e.g. dti_md -> features_dti_md.csv) with the same rules; gray_matter: true adds
          the gray-matter MD regions from gm_md.py (gm_md.feature_csv), placed after the tracts

Outputs (paths.out_dir):
    features_<mod>.csv          visit_id, <mod>_image_id, features
    features_<mod>_report.csv   every input column: kept or why dropped
    features_<mod>_gaps.csv     visits with empty cells in kept columns
    etiv.csv                    visit_id, etiv
"""

import re

import numpy as np
import pandas as pd
from common import load_config, out_dir, require_file

SMRI_MEASURES = {"volume", "area", "thickness", "meancurv"}
SMRI_TECHNICAL = ("UnsegmentedWhiteMatter",)
ETIV_REGIONS = ("eTIV", "EstimatedTotalIntraCranialVol")

# Whole-brain / whole-hemisphere totals from the '# Measure' header lines. The extraction
# labels some with an empty measure and some as '<name>_volume' (when the parser reports the
# header's first field), so they are matched by region name as well.
WHOLE_BRAIN = re.compile(
    r"^(lh|rh)?("
    r"EstimatedTotalIntraCranialVol|eTIV|SegmentedTotalIntracranial(Vol)?|"
    r"BrainSeg[\w-]*|Cortex(Vol)?|CerebralWhiteMatter(Vol)?|SubCortGray(Vol)?|TotalGray(Vol)?|"
    r"SupraTentorial[\w-]*|Mask[\w-]*|SurfaceHoles|WhiteSurfArea|MeanThickness"
    r")$"
)
DTI_DEFAULT_MAPS = ["FA", "AxD", "RD"]
GM_KINDS = {"cortex", "subcortical"}          # gray-matter MD regions from gm_md.py


# ---------- modality rules: column -> drop reason ('' = keep) ----------

def smri_rules(dictionary):
    reasons = {}
    for row in dictionary.itertuples():
        measure = row.measure if isinstance(row.measure, str) else ""
        region = str(row.region)
        if measure == "" or WHOLE_BRAIN.match(region):
            reasons[row.column] = "aggregate_whole_brain"
        elif region.startswith("Whole_"):
            reasons[row.column] = "aggregate_whole_structure"
        elif measure not in SMRI_MEASURES:
            reasons[row.column] = f"measure_{measure}"
        elif any(t in region for t in SMRI_TECHNICAL):
            reasons[row.column] = "technical"
        else:
            reasons[row.column] = ""
    return reasons


def dti_rules(dictionary, maps):
    reasons = {}
    for row in dictionary.itertuples():
        if row.kind in GM_KINDS:
            reasons[row.column] = ""
        elif row.kind != "tract":
            reasons[row.column] = f"aggregate_{row.kind.replace(' ', '_')}"
        elif row.map not in maps:
            reasons[row.column] = f"map_{row.map}"
        else:
            reasons[row.column] = ""
    return reasons


def dti_names(dictionary):
    """'FA_FX' -> 'FA Fornix (column and body of fornix) (FX)'."""
    names = {}
    for row in dictionary.itertuples():
        if row.kind in GM_KINDS:
            names[row.column] = f"MD gray matter {row.roi}"
            continue
        desc = str(row.description).split(": ", 1)[-1]
        names[row.column] = f"{row.map} {desc} ({row.roi})"
    return names


def find_etiv(dictionary, configured):
    if configured in set(dictionary["column"]):
        return configured
    hits = dictionary[(dictionary["source"] == "aseg") & dictionary["region"].isin(ETIV_REGIONS)]
    if hits.empty:
        raise ValueError("eTIV column not found in the sMRI dictionary")
    return hits["column"].iloc[0]


# ---------- shared cleaning ----------

def drop_correlated(frame, threshold):
    """Greedy, in column order: drop a column if |r| > threshold with an earlier kept one."""
    cols = list(frame.columns)
    corr = frame.corr().abs().to_numpy().copy()
    np.fill_diagonal(corr, 0.0)
    dropped = {}
    for i in range(len(cols)):
        if cols[i] in dropped:
            continue
        for j in range(i + 1, len(cols)):
            if cols[j] not in dropped and corr[i, j] > threshold:
                dropped[cols[j]] = f"corr_dup_with:{cols[i]}"
    return dropped


def clean(name, mod_cfg, cohort, feat_cfg, out, out_name=None, maps=None, gm_cfg=None):
    out_name = out_name or name
    image_col = f"{name}_image_id"
    dictionary = pd.read_csv(require_file(mod_cfg["column_dict_csv"]))
    raw = pd.read_csv(require_file(mod_cfg["feature_csv"]), low_memory=False)
    raw["image_id"] = raw["image_id"].astype(str).str.strip()
    if gm_cfg:
        # gray-matter MD regions (gm_md.py), joined on the DTI image id, after the tracts
        gm = pd.read_csv(require_file(gm_cfg["feature_csv"]), dtype={"dti_image_id": str})
        gm = gm.drop(columns="visit_id").rename(columns={"dti_image_id": "image_id"})
        raw = raw.merge(gm, on="image_id", how="left")
        dictionary = pd.concat([dictionary, pd.read_csv(require_file(gm_cfg["column_dict_csv"]))],
                               ignore_index=True)

    # 1. cohort visits only, keyed by visit_id

    # 1. cohort visits only, keyed by visit_id
    keys = cohort[["visit_id", image_col]].rename(columns={image_col: "image_id"})
    raw = keys.merge(raw, on="image_id", how="inner")
    print(f"[{name}] cohort visits with features: {len(raw)}/{len(cohort)}")

    columns = [c for c in raw.columns if c not in ("visit_id", "image_id", "subject_id")]
    reasons = {c: "no_dictionary_entry" for c in columns if c not in set(dictionary["column"])}

    extra = {}
    if name == "smri":
        rule = smri_rules(dictionary)
        etiv_col = find_etiv(dictionary, feat_cfg.get("smri", {}).get("etiv_col", ""))
        extra["etiv"] = raw[["visit_id", etiv_col]].rename(columns={etiv_col: "etiv"})
        names = {}
    else:
        maps = maps or feat_cfg.get("dti", {}).get("maps", DTI_DEFAULT_MAPS)
        rule = dti_rules(dictionary, maps)
        names = dti_names(dictionary)
    for c in columns:
        if c not in reasons and rule.get(c):
            reasons[c] = rule[c]

    kept = [c for c in columns if c not in reasons]
    if name == "dti":
        # order by the maps list, so near-duplicate removal keeps the earlier maps' columns
        map_of = dict(zip(dictionary["column"], dictionary["map"]))
        kind_of = dict(zip(dictionary["column"], dictionary["kind"]))
        kept.sort(key=lambda c: len(maps) if kind_of[c] in GM_KINDS else maps.index(map_of[c]))
    feats = raw[kept].apply(pd.to_numeric, errors="coerce")

    # 2. mostly-empty columns
    # 2. mostly-empty columns
    max_missing = feat_cfg.get("max_missing", 0.05)
    missing = feats.isna().mean()
    for c in missing[missing > max_missing].index:
        reasons[c] = f"missing_{missing[c]:.1%}"
    if gm_cfg:
        for c in feats.columns:
            if kind_of[c] in GM_KINDS and missing[c] > 0 and c not in reasons:
                reasons[c] = f"gm_not_measured_{missing[c]:.1%}"
    feats = feats[[c for c in feats.columns if c not in reasons]]

    # 3. zero variance
    for c in feats.columns[feats.nunique(dropna=True) <= 1]:
        reasons[c] = "zero_variance"
    feats = feats[[c for c in feats.columns if c not in reasons]]

    # 4. near-duplicates
    reasons.update(drop_correlated(feats, feat_cfg.get("corr_threshold", 0.99)))
    feats = feats[[c for c in feats.columns if c not in reasons]]

    # visits with gaps in what is kept: listed, not dropped
    gaps = feats.isna()
    gap_rows = pd.DataFrame({
        "visit_id": raw["visit_id"],
        "n_missing": gaps.sum(axis=1),
        "columns": gaps.apply(lambda r: " | ".join(r.index[r]), axis=1),
    })
    gap_rows = gap_rows[gap_rows["n_missing"] > 0]

    final = pd.concat([raw[["visit_id", "image_id"]].rename(columns={"image_id": image_col}),
                       feats.rename(columns=names)], axis=1)

    report = pd.DataFrame({
        "column": columns,
        "name": [names.get(c, c) for c in columns],
        "kept": [c not in reasons for c in columns],
        "drop_reason": [reasons.get(c, "") for c in columns],
    })

    # safety net: how strongly each column follows head size (eTIV)
    if "etiv" in extra:
        etiv = pd.to_numeric(extra["etiv"]["etiv"], errors="coerce")
        numeric = raw[columns].apply(pd.to_numeric, errors="coerce")
        report["abs_corr_etiv"] = numeric.corrwith(etiv).abs().round(3).to_numpy()
        top = report[report["kept"]].nlargest(5, "abs_corr_etiv")
        print(f"[{name}] kept features most correlated with eTIV:")
        print("\n".join(f"    {r.abs_corr_etiv:.3f}  {r.column}" for r in top.itertuples()))
        dropped_totals = report.loc[report["drop_reason"] == "aggregate_whole_brain", "column"]
        print(f"[{name}] whole-brain totals dropped ({len(dropped_totals)}): "
              f"{', '.join(dropped_totals.head(40))}")

    final.to_csv(out / f"features_{out_name}.csv", index=False)
    report.to_csv(out / f"features_{out_name}_report.csv", index=False)
    gap_rows.to_csv(out / f"features_{out_name}_gaps.csv", index=False)
    if "etiv" in extra:
        extra["etiv"].to_csv(out / "etiv.csv", index=False)

    counts = report.loc[~report["kept"], "drop_reason"].str.split(":").str[0].value_counts()
    print(f"[{out_name}] input columns {len(columns)} -> kept {int(report['kept'].sum())}")
    print("\n".join(f"    {k}: {v}" for k, v in counts.items()))
    print(f"[{name}] visits with empty cells: {len(gap_rows)}")


if __name__ == "__main__":
    config = load_config()
    out = out_dir(config)
    cohort = pd.read_csv(out / "cohort.csv", dtype=str)
    for name in config["cohort"]["require_modalities"]:
        clean(name, config["modalities"][name], cohort, config.get("features", {}), out)
    for out_name, table in config.get("features", {}).get("dti", {}).get("extra_tables", {}).items():
        clean("dti", config["modalities"]["dti"], cohort, config.get("features", {}), out,
              out_name=out_name, maps=table["maps"],
              gm_cfg=config["gm_md"] if table.get("gray_matter") else None)
    print(f"\nSaved to {out}")