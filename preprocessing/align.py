"""Align each visit's DTI onto the same person's T1 (rigid), for the CNN input image.

Method (validated by the pilot):
    FLIRT 6 dof, normalized mutual information, b0_brain -> brain.finalsurfs,
    starting from the scanner position (-usesqform) and only fine-tuning (-nosearch).
    The same transform is then applied to FA.
The T1 is not moved. Features (FreeSurfer, ENIGMA) are not touched.

Outputs, per visit, in paths.aligned_dir/<visit_id>/:
    dti_to_t1.mat      FLIRT transform
    FA_in_t1.nii.gz    FA on the T1 grid
    qc.json            brain overlap between T1 and aligned b0
Report: paths.out_dir/align_report.csv, with outliers flagged.

Safe to rerun: finished visits are skipped.
"""

import concurrent.futures
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from common import load_config, out_dir

N_JOBS = 8
OUTLIER_MADS = 3        # flag a visit whose brain overlap is this many MADs below the median


def is_real(path):
    """A DataLad pointer is a few bytes of text, not an image."""
    return path.is_file() and path.stat().st_size >= 1024


def flirt(*args):
    subprocess.run(["flirt", *map(str, args)], check=True, capture_output=True)


def dice(a, b):
    total = a.sum() + b.sum()
    return float(2 * (a & b).sum() / total) if total else float("nan")


def align_visit(r, smri_root, dti_root, aligned_dir):
    t1_mgz = smri_root / r.adni_id / r.smri_image_id / "mri/brain.finalsurfs.mgz"
    dti = dti_root / r.adni_id / r.dti_image_id
    b0, fa = dti / "mask/b0_brain.nii.gz", dti / "dtifit/dti_FA.nii.gz"
    out = aligned_dir / r.visit_id
    info = {"visit_id": r.visit_id,
            "dti_type": "pair" if (dti / "input/rev.bval").exists() else "single"}

    if (out / "qc.json").is_file() and (out / "FA_in_t1.nii.gz").is_file():
        return {**info, **json.loads((out / "qc.json").read_text()), "status": "skipped"}

    missing = [p.name for p in (t1_mgz, b0, fa) if not is_real(p)]
    if missing:
        return {**info, "status": "missing_data", "detail": " ".join(missing)}

    out.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            t1_img = nib.load(str(t1_mgz))
            t1 = Path(tmp) / "t1.nii.gz"          # FSL cannot read .mgz
            nib.save(nib.Nifti1Image(np.asarray(t1_img.dataobj), t1_img.affine), str(t1))

            b0_in_t1 = Path(tmp) / "b0_in_t1.nii.gz"
            flirt("-in", b0, "-ref", t1, "-dof", 6, "-cost", "normmi", "-usesqform", "-nosearch",
                  "-omat", out / "dti_to_t1.mat", "-out", b0_in_t1)
            flirt("-in", fa, "-ref", t1, "-applyxfm", "-init", out / "dti_to_t1.mat",
                  "-interp", "trilinear", "-out", out / "FA_in_t1.nii.gz")

            t1_brain = np.asarray(t1_img.dataobj) > 0
            b0_brain = np.asarray(nib.load(str(b0_in_t1)).dataobj) > 0
            qc = {"brain_dice": dice(t1_brain, b0_brain)}
    except subprocess.CalledProcessError as e:
        return {**info, "status": "failed", "detail": e.stderr.decode(errors="ignore")[-300:]}

    (out / "qc.json").write_text(json.dumps(qc))
    return {**info, **qc, "status": "done"}


def flag_outliers(report):
    """Flag visits whose brain overlap is far below the rest (median - 3 MAD)."""
    ok = report["brain_dice"].notna()
    d = report.loc[ok, "brain_dice"]
    median = d.median()
    mad = (d - median).abs().median()
    cutoff = median - OUTLIER_MADS * 1.4826 * mad   # 1.4826: MAD -> standard deviation scale
    report["outlier"] = ok & (report["brain_dice"] < cutoff)
    return median, cutoff


if __name__ == "__main__":
    if shutil.which("flirt") is None:
        sys.exit("flirt not found: load FSL first")

    config = load_config()
    smri_root = Path(config["modalities"]["smri"]["scan_root"])
    dti_root = Path(config["modalities"]["dti"]["scan_root"])
    aligned_dir = Path(config["paths"]["aligned_dir"])
    aligned_dir.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(out_dir(config) / "cohort.csv", dtype=str)
    print(f"{len(cohort)} visits -> {aligned_dir}")

    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=N_JOBS) as pool:
        jobs = [pool.submit(align_visit, r, smri_root, dti_root, aligned_dir)
                for r in cohort.itertuples()]
        for i, job in enumerate(concurrent.futures.as_completed(jobs), 1):
            rows.append(job.result())
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}")

    report = pd.DataFrame(rows)
    for col in ("brain_dice", "detail"):
        if col not in report:
            report[col] = np.nan
    median, cutoff = flag_outliers(report)
    report = report.sort_values(["outlier", "brain_dice"], ascending=[False, True])
    report.to_csv(out_dir(config) / "align_report.csv", index=False)

    print(f"\nstatus: {report['status'].value_counts().to_dict()}")
    print(f"brain overlap: median {median:.3f}, outlier below {cutoff:.3f}")
    print(f"outliers: {int(report['outlier'].sum())}")
    print(report[report["outlier"] | report["status"].isin(["failed", "missing_data"])]
          [["visit_id", "dti_type", "status", "brain_dice", "detail"]].to_string(index=False))
    print(f"\nSaved: {out_dir(config) / 'align_report.csv'}")