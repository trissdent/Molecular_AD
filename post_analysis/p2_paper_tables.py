import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from i1_index_core import AGECORR_INDEX, PRIMARY_SPLIT, RAW_INDEX, drop_excluded
from p1_paper_figures import readable

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "checkpoints"
LOGS = ROOT / "logs"
RESULTS = ROOT / "results"
OUT = RESULTS / "paper"
DEFAULTS = ROOT / "src" / "configs" / "defaults.yaml"

INDEXES = {"age-corrected": AGECORR_INDEX, "uncorrected": RAW_INDEX}
METRICS = {
    "auc": "AUC",
    "cohens_d": "Cohen's d",
    "spearman_r_mmse": "MMSE, all",
    "spearman_r_mmse_ad_only": "MMSE, AD",
    "spearman_r_ptau217_alzpath": "pTau-217 AlzPath, all",
    "spearman_r_ptau217_alzpath_ad_only": "pTau-217 AlzPath, AD",
    "spearman_r_ptau217_janssen": "pTau-217 Janssen, all",
    "spearman_r_ptau217_janssen_ad_only": "pTau-217 Janssen, AD",
}
CI = {"auc": ("auc_ci_lo", "auc_ci_hi")} | {
    c: (c.replace("spearman_r_", "spearman_ci_lo_"), c.replace("spearman_r_", "spearman_ci_hi_"))
    for c in METRICS if c.startswith("spearman_r_")}
COMPARE = {"auc": "AUC", "mmse_ad": "MMSE, AD", "ptau_alzpath": "pTau-217 AlzPath, AD",
           "ptau_janssen": "pTau-217 Janssen, AD"}
NAMES = {"full": "full", "fixed": "fixed features", "off": "no CIMLR", "hippocampus": "hippocampal volume",
         "pca_smri": "T1 PCA", "pca_dti": "DTI PCA", "t1": "T1 full", "dti": "DTI full"}
PTAU = {"pTau-217 AlzPath": "ptau217_alzpath", "pTau-217 Janssen": "ptau217_janssen", "pTau-217 ELISA": "ptau217"}


def save(table, name):
    OUT.mkdir(parents=True, exist_ok=True)
    table = table.fillna("–")
    table.to_csv(OUT / f"{name}.csv", index=False)
    print(f"\n{name}\n" + table.to_string(index=False))


def stars(p):
    return "" if pd.isna(p) else "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def cell(row, c, ci=True):
    if c not in row or pd.isna(row[c]):
        return "–"
    lo, hi = CI.get(c, (None, None))
    if ci and lo in row and pd.notna(row[lo]):
        return f"{row[c]:.2f} [{row[lo]:.2f}, {row[hi]:.2f}]"
    p = c.replace("spearman_r_", "spearman_p_")
    return f"{row[c]:.2f}" + (stars(row[p]) if c.startswith("spearman_r_") and p in row else "")


def p_text(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


def carrier(series):
    return series.map(lambda g: float("4" in g) if isinstance(g, str) else np.nan)


def describe(df, female, e4):
    age, mmse, known = df["age"].dropna(), df["mmse"].dropna(), e4.dropna()
    col = {
        "Imaging sessions, n": f"{len(df)}",
        "Participants, n": f"{df['participant'].nunique()}",
        "CN / AD, n": f"{(df['diagnosis'] == 'cn').sum()} / {(df['diagnosis'] == 'ad').sum()}",
        "Age, years, mean ± SD": f"{age.mean():.1f} ± {age.std():.1f}",
        "Female, n (%)": f"{int(female.sum())} ({100 * female.mean():.0f})",
        "MMSE, mean ± SD (n)": f"{mmse.mean():.1f} ± {mmse.std():.1f} ({len(mmse)})",
        "APOE ε4 carriers, n (%) (n known)": f"{int(known.sum())} ({100 * known.mean():.0f}) ({len(known)})",
    }
    for label, c in PTAU.items():
        col[f"{label}, n"] = f"{df[c].notna().sum()}" if c in df and df[c].notna().any() else "–"
    return col


def table1(external_csv, t1_root):
    prepared = Path(yaml.safe_load(DEFAULTS.read_text())["data"]["prepared_dir"])
    cohort = pd.read_csv(prepared / "cohort.csv", dtype={"visit_id": str, "adni_id": str})
    split = pd.read_csv(prepared / "split.csv", dtype={"visit_id": str})
    adni = drop_excluded(split[["visit_id", "split"]].merge(cohort, on="visit_id", how="left"))
    adni["participant"], adni["diagnosis"] = adni["adni_id"], adni["diagnosis"].str.lower()
    columns = {}
    for s, label in [("train", "Training"), ("val", "Validation"), ("test", "Test")]:
        part = adni[adni["split"] == s]
        columns[label] = describe(part, part["gender"] == 2, carrier(part["apoe_genotype"]))
    if external_csv and Path(external_csv).is_file():
        from e1_external import drop_ptau_outliers
        ext = pd.read_csv(external_csv, dtype={"subject_id": str})
        ext["subject_id"] = ext["subject_id"].str.strip()
        has_mri = ext["subject_id"].map(lambda s: (Path(t1_root) / s / "mri" / "brain.finalsurfs.mgz").is_file())
        print(f"Vietnamese cohort: {int(has_mri.sum())} of {len(ext)} with MRI kept")
        ext = drop_ptau_outliers(ext[has_mri])
        ext["participant"], ext["diagnosis"] = ext["subject_id"], ext["diagnosis"].str.strip().str.lower()
        female = pd.to_numeric(ext["gender"], errors="coerce") == 0
        columns["Vietnamese cohort"] = describe(ext, female, carrier(ext["apoe_genotype"]))
    save(pd.DataFrame(columns).reset_index().rename(columns={"index": "Characteristic"}), "table1")


def seed_of(run):
    return yaml.safe_load((LOGS / run.name / "config.yml").read_text())["training"]["seed"]


def runs_of(patterns):
    return sorted({p for pattern in patterns for p in CHECKPOINTS.glob(pattern)
                   if (p / "s2_index" / "index_validation.csv").is_file()})


def validation(run):
    v = pd.read_csv(run / "s2_index" / "index_validation.csv")
    return v[v["split"] == PRIMARY_SPLIT].set_index("index_name")


def table2(models, seed):
    base = pd.read_csv(RESULTS / "baselines" / "baseline_validation.csv")
    base = base[base["split"] == PRIMARY_SPLIT]
    rows, tables, seeds = [], {k: [] for k in [*INDEXES, "simple"]}, []
    for label, name in models.items():
        groups = {"full": [f"{name}_s*", f"{name}_20*"], "fixed": [f"{name}_fixed_s*"], "off": [f"{name}_off_s*"]}
        for g, patterns in groups.items():
            rows.append((f"{label} {NAMES[g]}", runs_of(patterns)))
        rows.append((f"{label} PCA", f"pca_{'smri' if label == 'T1' else 'dti'}"))
        if label == "T1":
            rows.append(("Hippocampal volume", "hippocampus"))
    for label, source in rows:
        if isinstance(source, str):
            b = base[base["baseline"] == source].set_index("index_name")
            for k, idx in INDEXES.items():
                tables[k].append({"Model": label, **({METRICS[c]: cell(b.loc[idx], c) for c in METRICS}
                                                      if idx in b.index else {})})
            tables["simple"].append({"Model": label, **({METRICS[c]: cell(b.loc[AGECORR_INDEX], c, ci=False)
                                                         for c in METRICS} if AGECORR_INDEX in b.index else {})})
            continue
        for run in source:
            v = validation(run)
            for k, idx in INDEXES.items():
                seeds.append({"Model": label, "Seed": seed_of(run), "Index": k,
                              **{METRICS[c]: round(float(v.loc[idx, c]), 3) for c in METRICS if c in v}})
        chosen = [r for r in source if seed_of(r) == seed]
        if not chosen:
            print(f"{label}: no seed-{seed} run found")
        for k, idx in INDEXES.items():
            v = validation(chosen[-1]) if chosen else None
            tables[k].append({"Model": label, **({METRICS[c]: cell(v.loc[idx], c) for c in METRICS}
                                                  if chosen else {})})
        v = validation(chosen[-1]) if chosen else None
        tables["simple"].append({"Model": label, **({METRICS[c]: cell(v.loc[AGECORR_INDEX], c, ci=False)
                                                     for c in METRICS} if chosen else {})})
    save(pd.DataFrame(tables["simple"]), "table2")
    save(pd.DataFrame(tables["age-corrected"]), "supp_table2_ci")
    save(pd.DataFrame(tables["uncorrected"]), "supp_table2_uncorrected")
    save(pd.DataFrame(seeds).sort_values(["Model", "Index", "Seed"], kind="stable"), "supp_seeds")


def table3():
    files = [("T1", RESULTS / "compare" / "t1.csv"), ("DTI", RESULTS / "compare" / "dti.csv"),
             ("", RESULTS / "compare" / "t1_vs_dti.csv")]
    for k, idx in INDEXES.items():
        rows, simple = [], []
        for label, path in files:
            if not path.is_file():
                print(f"missing {path}: run m2_compare_models.py first")
                continue
            df = pd.read_csv(path)
            df = df[(df["index_name"] == idx) & (df["model_a"].isin(["full", "t1"]))]
            for (a, b), g in df.groupby(["model_a", "model_b"], sort=False):
                name = f"{label} {NAMES[a]} − {NAMES[b]}".strip()
                row, short = {"Comparison": name}, {"Comparison": name}
                for metric, col in COMPARE.items():
                    m = g[g["metric"] == metric]
                    if len(m):
                        m = m.iloc[0]
                        row[col] = f"{m['difference']:+.2f} [{m['ci_lo']:+.2f}, {m['ci_hi']:+.2f}], {p_text(m['p_holm'])}"
                        short[col] = f"{m['difference']:+.2f}{stars(m['p_holm'])}"
                rows.append(row)
                simple.append(short)
        if k == "age-corrected":
            save(pd.DataFrame(simple), "table3")
            save(pd.DataFrame(rows), "supp_table3_ci")
        else:
            save(pd.DataFrame(rows), "supp_table3_uncorrected")


def supp_features(min_seeds=3):
    parts = []
    for label, name in [("T1", "t1"), ("DTI", "dti")]:
        path = RESULTS / "features" / f"{name}_links.csv"
        if not path.is_file():
            print(f"missing {path}: run m3_features.py with --out {name}")
            continue
        links = pd.read_csv(path)
        part = links[links["seeds_full"] >= min_seeds].sort_values("seeds_full", ascending=False)
        out = pd.DataFrame({"Model": label, "Feature": part["feature"].map(readable),"Seeds selected (full)": part["seeds_full"]})
        if "seeds_fixed" in part:
            out["Seeds selected (fixed)"] = part["seeds_fixed"]
        for c, col in [("r_mmse", "ρ MMSE, AD"), ("r_ptau217_alzpath", "ρ pTau-217 AlzPath, AD"),
                       ("r_ptau217_janssen", "ρ pTau-217 Janssen, AD")]:
            if c in part:
                out[col] = part[c].round(2)
        parts.append(out)
    if parts:
        save(pd.concat(parts, ignore_index=True), "supp_selected_features")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t1", default="smri_pw0.5")
    parser.add_argument("--dti", default="dti_md_gm_noage")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--external-csv", default="/mnt/data_lab513/molecular_mri/umc.csv")
    parser.add_argument("--t1-root", default="/mnt/data_lab513/molecular_mri/preprocessed")
    args = parser.parse_args()
    pd.set_option("display.width", 250)
    table1(args.external_csv, args.t1_root)
    table2({"T1": args.t1, "DTI": args.dti}, args.seed)
    table3()
    supp_features()
    print("\nSaved in results/paper/")