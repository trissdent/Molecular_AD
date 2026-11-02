"""Build the visit-level cohort: one row per visit that has every required modality.

Rule: the server decides which scans exist; the DB only confirms the modality and adds labels.

Outputs (in paths.out_dir):
    cohort.csv          one row per visit
    cohort_report.txt   counts and everything that was dropped, and why
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from common import load_config, out_dir, require_file
from db import db_url, fetch

# ---------- 1. scans on the server ----------

def scans_on_server(scan_root, scan_file):
    """{image_id: subject_folder} for every <scan_root>/<subject>/<image_id>/<scan_file> that exists."""
    found = {}
    for subject in sorted(os.listdir(scan_root)):
        subject_dir = Path(scan_root) / subject
        if subject.startswith(".") or not subject_dir.is_dir():
            continue
        for image_id in sorted(os.listdir(subject_dir)):
            if (subject_dir / image_id / scan_file).is_file():
                found[image_id] = subject
    return found


def feature_ids(feature_csv):
    ids = pd.read_csv(require_file(feature_csv), usecols=["image_id"])["image_id"]
    return set(ids.astype(str).str.strip())


# ---------- 2. confirm in the DB ----------

def visits_in_db(table, image_ids, url):
    """image_id -> visit_id, only for ids that are in this modality's table."""
    rows = fetch(f"SELECT image_id, visit_id FROM {table} WHERE image_id = ANY($1)",
                 (sorted(image_ids),), url)
    return pd.DataFrame(rows, columns=["image_id", "visit_id"]).astype(str)


def modality_table(name, mod, url, report):
    """One row per visit for one modality: visit_id, <name>_image_id."""
    on_disk = scans_on_server(mod["scan_root"], mod["scan_file"])
    with_features = feature_ids(mod["feature_csv"])

    disk_ids = set(on_disk)
    both = disk_ids & with_features
    report.append(f"[{name}] scan on server: {len(disk_ids)} | in feature csv: {len(with_features)} "
                  f"| both: {len(both)}")
    report.list(f"[{name}] scan but no features", disk_ids - with_features)
    report.list(f"[{name}] features but no scan", with_features - disk_ids)

    db = visits_in_db(mod["db_table"], both, url)
    report.append(f"[{name}] confirmed in DB table '{mod['db_table']}': {len(db)}")
    report.list(f"[{name}] on server but not in '{mod['db_table']}' (other modality / unknown)",
                both - set(db["image_id"]))

    dup = db[db["visit_id"].duplicated(keep=False)]
    if len(dup):
        report.list(f"[{name}] visits with >1 image (dropped, check by hand)",
                    sorted(f"{v}: {' '.join(g['image_id'])}" for v, g in dup.groupby("visit_id")))
    db = db[~db["visit_id"].isin(dup["visit_id"])]

    return db.rename(columns={"image_id": f"{name}_image_id"})


# ---------- 3. clinical info ----------

VISIT_QUERY = """
SELECT
    v.id AS visit_id,
    s.adni_id,
    s.gender,
    s.birth_year,
    s.birth_month,
    v.education,
    v.record_date,
    v.diagnosis,
    m.total_score AS mmse
FROM visit v
JOIN subject s ON v.subject_id = s.id
LEFT JOIN mmse m ON m.visit_id = v.id
WHERE v.id = ANY($1);
"""


def visit_info(visit_ids, url):
    rows = fetch(VISIT_QUERY, (sorted(visit_ids),), url)
    df = pd.DataFrame(rows)
    df["visit_id"] = df["visit_id"].astype(str)
    df["adni_id"] = df["adni_id"].astype(str).str.strip()
    df["diagnosis"] = df["diagnosis"].astype(str).str.strip().str.lower()
    # a visit can have several MMSE rows: keep the first non-missing one
    df = df.sort_values("mmse", na_position="last").drop_duplicates("visit_id")
    df["age"] = compute_age(df)
    return df


def compute_age(df):
    """Age in years at the visit, using the birth month when it is known."""
    record = pd.to_datetime(df["record_date"], errors="coerce")
    birth_month = df["birth_month"].astype(float).fillna(record.dt.month)
    age = (record.dt.year - df["birth_year"].astype(float)) + (record.dt.month - birth_month) / 12.0
    return age.round(2)


def apoe_table(apoe_csv):
    apoe = pd.read_csv(require_file(apoe_csv), usecols=["PTID", "GENOTYPE"])
    apoe["adni_id"] = apoe["PTID"].astype(str).str.strip()
    return (apoe.rename(columns={"GENOTYPE": "apoe_genotype"})[["adni_id", "apoe_genotype"]]
                .dropna().drop_duplicates("adni_id"))


def biomarker_table(clinical, report):
    bio = pd.read_csv(require_file(clinical["biomarker_csv"]), low_memory=False)
    bio = bio[bio["ASSAYPLATFORM"] == clinical["biomarker_platform"]]
    bio["visit_id"] = bio["visit_id"].astype(str)

    out = pd.DataFrame({"visit_id": bio["visit_id"].unique()})
    for biomarker, column in clinical["biomarkers"].items():
        sub = bio[bio["PLASMA_BIOMARKER"] == biomarker][["visit_id", "TESTVALUE"]].copy()
        sub[column] = pd.to_numeric(sub["TESTVALUE"], errors="coerce")
        sub = sub.dropna(subset=[column])
        n_dup = int(sub["visit_id"].duplicated().sum())
        if n_dup:
            report.append(f"[{column}] {n_dup} visits with repeated measurements, first kept")
        sub = sub.drop_duplicates("visit_id")[["visit_id", column]]
        out = out.merge(sub, on="visit_id", how="left")
    return out


# ---------- report ----------

class Report(list):
    def list(self, title, items, show=20):
        items = sorted(items)
        self.append(f"{title}: {len(items)}")
        for item in items[:show]:
            self.append(f"    {item}")
        if len(items) > show:
            self.append(f"    ... {len(items) - show} more")


# ---------- main ----------

def build_cohort(config):
    report = Report()
    url = db_url(config["paths"]["db_url_env"])
    required = config["cohort"]["require_modalities"]

    tables = [modality_table(name, config["modalities"][name], url, report) for name in required]
    cohort = tables[0]
    for table in tables[1:]:
        cohort = cohort.merge(table, on="visit_id", how="inner")
    report.append(f"visits with all of {required}: {len(cohort)}")

    info = visit_info(cohort["visit_id"], url)
    cohort = cohort.merge(info, on="visit_id", how="left")

    keep = config["cohort"]["diagnoses"]
    counts = cohort["diagnosis"].value_counts(dropna=False).to_dict()
    report.append(f"diagnosis before filter: {counts}")
    cohort = cohort[cohort["diagnosis"].isin(keep)]
    report.append(f"after keeping {keep}: {len(cohort)} visits, {cohort['adni_id'].nunique()} subjects")

    clinical = config["clinical"]
    cohort = cohort.merge(apoe_table(clinical["apoe_csv"]), on="adni_id", how="left")
    cohort = cohort.merge(biomarker_table(clinical, report), on="visit_id", how="left")

    columns = (["visit_id", "adni_id"] + [f"{m}_image_id" for m in required]
               + ["diagnosis", "age", "gender", "education", "record_date", "mmse", "apoe_genotype"]
               + list(clinical["biomarkers"].values()))
    cohort = cohort[columns].sort_values(["adni_id", "record_date"]).reset_index(drop=True)

    report.append("")
    report.append(f"final: {len(cohort)} visits, {cohort['adni_id'].nunique()} subjects, "
                  f"{cohort['diagnosis'].value_counts().to_dict()}")
    report.append(f"visits per subject: {cohort.groupby('adni_id').size().value_counts().sort_index().to_dict()}")
    for col in ["age", "mmse", "apoe_genotype"] + list(clinical["biomarkers"].values()):
        report.append(f"  {col:18s} present {int(cohort[col].notna().sum())}/{len(cohort)}")

    return cohort, report


if __name__ == "__main__":
    config = load_config()
    cohort, report = build_cohort(config)

    out = out_dir(config)
    cohort.to_csv(out / "cohort.csv", index=False)
    (out / "cohort_report.txt").write_text("\n".join(report) + "\n")

    print("\n".join(report))
    print(f"\nSaved: {out / 'cohort.csv'}")
    print(f"Saved: {out / 'cohort_report.txt'}")