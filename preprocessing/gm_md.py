"""Gray-matter mean diffusivity (MD) per FreeSurfer region, for this project's DTI features.

Project-specific step (not part of the lab pipeline). Uses what align.py already made:
MD_in_t1.nii.gz (MD on the T1 grid, FLIRT transform from align.py) and the visit's FreeSurfer
aparc+aseg.mgz (same grid). Standard FreeSurfer commands only:
    mri_binarize   mask of voxels with MD below gm_md.md_ceiling (CSF-like voxels excluded)
    mri_segstats   mean MD per label inside that mask
                   cortex (aparc labels): no erosion, the ribbon is only a few voxels thick
                   subcortical gray matter (aseg labels): eroded by gm_md.seg_erode voxels
The .stats files are read with FreeSurfer's own parser (fsbindings/legacy.py).

Inputs per visit:
    paths.aligned_dir/<visit_id>/MD_in_t1.nii.gz
    <smri scan_root>/<adni_id>/<smri_image_id>/mri/aparc+aseg.mgz
Outputs:
    gm_md.work_dir/<visit_id>/     md_mask.nii.gz, cortex.stats, subcort.stats, commands.log
    gm_md.feature_csv              visit_id, dti_image_id, gmmd_<source>_<region> (mm^2/s)
    gm_md.column_dict_csv          column, map, roi, kind, description, unit
    paths.out_dir/gm_md_report.csv per visit: status, voxels per region

Usage (from preprocessing/):
    python gm_md.py --visits <visit_id>,<visit_id>    a few visits, to check the computation
    python gm_md.py                                   every cohort visit (finished ones are skipped)
"""

import argparse
import concurrent.futures
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from common import load_config, out_dir
from tqdm import tqdm
# aparc cortical labels in aparc+aseg: 1001-1035 (lh), 2001-2035 (rh); 1004/2004 corpus callosum is not cortex
CORTEX_IDS = [i for i in list(range(1001, 1036)) + list(range(2001, 2036)) if i not in (1004, 2004)]


def is_real(path):
    """A DataLad pointer is a few bytes of text, not an image."""
    return path.is_file() and path.stat().st_size >= 1024


def run(cmd, env, log):
    with open(log, "a") as fh:
        fh.write("$ " + " ".join(map(str, cmd)) + "\n")
        fh.flush()
        subprocess.run([str(c) for c in cmd], env=env, check=True, stdout=fh, stderr=subprocess.STDOUT)


def load_fs_parser(freesurfer_home):
    """FreeSurfer's own stats parser, loaded by path (fsbindings refuses import outside fspython)."""
    path = Path(freesurfer_home) / "python/packages/fsbindings/legacy.py"
    spec = importlib.util.spec_from_file_location("fs_legacy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_stats(path, legacy, measure):
    """{region: value} for one measure of a mri_segstats summary file."""
    id_name, values = legacy.AsegStatsParser(str(path)).parse(measure)
    return dict(zip(id_name.values(), values))


def process_visit(r, cfg, smri_root, aligned_dir, work_root, fs_home):
    md = aligned_dir / r.visit_id / "MD_in_t1.nii.gz"
    seg = smri_root / r.adni_id / r.smri_image_id / "mri/aparc+aseg.mgz"
    work = work_root / r.visit_id
    info = {"visit_id": r.visit_id, "dti_image_id": r.dti_image_id, "smri_image_id": r.smri_image_id}

    stats = {"cortex": work / "cortex.stats", "subcort": work / "subcort.stats"}
    if all(p.is_file() for p in stats.values()):
        return {**info, "status": "skipped"}

    missing = [str(p) for p in (md, seg) if not is_real(p)]
    if missing:
        return {**info, "status": "missing_data", "detail": " ".join(missing)}

    work.mkdir(parents=True, exist_ok=True)
    log = work / "commands.log"
    env = {**os.environ, "FREESURFER_HOME": fs_home, "FSLOUTPUTTYPE": "NIFTI"}   # fslmaths writes .nii
    ctab = Path(fs_home) / "FreeSurferColorLUT.txt"
    mask = work / "md_mask.nii"
    md_x1000 = work / "md_x1000.nii"            # 1e-3 mm^2/s: mri_segstats writes 4 decimals; uncompressed = faster
    try:
        run(["fslmaths", md, "-mul", 1000, md_x1000], env, log)
        run(["mri_binarize", "--i", md_x1000, "--min", "1e-9", "--max", cfg["md_ceiling"], "--o", mask], env, log)
        run(["mri_segstats", "--seg", seg, "--ctab", ctab, "--i", md_x1000, "--mask", mask,
             "--id", *CORTEX_IDS, "--sum", stats["cortex"]], env, log)
        run(["mri_segstats", "--seg", seg, "--ctab", ctab, "--i", md_x1000, "--mask", mask,
             "--seg-erode", cfg["seg_erode"], "--id", *cfg["subcortical_ids"],
             "--sum", stats["subcort"]], env, log)
    except subprocess.CalledProcessError:
        for p in stats.values():                       # no half-finished visit counts as done
            p.unlink(missing_ok=True)
        return {**info, "status": "failed", "detail": f"see {log}"}
    return {**info, "status": "done"}


def collect(cohort, work_root, legacy):
    """One row per visit: mean MD per region, plus a report with voxel counts."""
    rows, report = [], []
    for r in cohort.itertuples():
        files = {s: work_root / r.visit_id / f"{s}.stats" for s in ("cortex", "subcort")}
        if not all(p.is_file() for p in files.values()):
            continue
        row = {"visit_id": r.visit_id, "dti_image_id": r.dti_image_id}
        rep = {"visit_id": r.visit_id}
        for source, path in files.items():
            means, counts = read_stats(path, legacy, "mean"), read_stats(path, legacy, "nvoxels")
            for region, value in means.items():
                row[f"gmmd_{source}_{region}"] = value
                rep[f"n_{source}_{region}"] = counts.get(region)
        rows.append(row)
        report.append(rep)
    features = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["visit_id", "dti_image_id"])
    report = pd.DataFrame(report) if report else pd.DataFrame(columns=["visit_id"])
    return features, report


def dictionary(features):
    out = []
    for col in features.columns.drop(["visit_id", "dti_image_id"]):
        _, source, region = col.split("_", 2)
        kind = "cortex" if source == "cortex" else "subcortical"
        where = (f"aparc+aseg {region}" if kind == "cortex"
                 else f"aparc+aseg {region}, eroded")
        out.append({"column": col, "map": "MD", "roi": region, "kind": kind,
                    "description": f"mean diffusivity: {where}, CSF-like voxels excluded",
                    "unit": "1e-3 mm^2/s"})
    return pd.DataFrame(out, columns=["column", "map", "roi", "kind", "description", "unit"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--visits", default="", help="comma-separated visit_ids (check run)")
    args, rest = parser.parse_known_args()

    for tool in ("mri_segstats", "mri_binarize", "fslmaths"):
        if shutil.which(tool) is None:
            sys.exit(f"{tool} not found: load FreeSurfer and FSL first")
    fs_home = os.environ.get("FREESURFER_HOME")
    if not fs_home:
        sys.exit("FREESURFER_HOME is not set")

    config = load_config(rest)
    cfg = config["gm_md"]
    smri_root = Path(config["modalities"]["smri"]["scan_root"])
    aligned_dir = Path(config["paths"]["aligned_dir"])
    work_root = Path(cfg["work_dir"])
    work_root.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(out_dir(config) / "cohort.csv", dtype=str)
    if args.visits:
        cohort = cohort[cohort["visit_id"].isin(args.visits.split(","))]
    print(f"{len(cohort)} visits -> {work_root}")

    status = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg["n_jobs"]) as pool:
        jobs = [pool.submit(process_visit, r, cfg, smri_root, aligned_dir, work_root, fs_home)
                for r in cohort.itertuples()]
        for job in tqdm(concurrent.futures.as_completed(jobs), total=len(jobs), desc="gm_md"):
            status.append(job.result())
    status = pd.DataFrame(status)
    print(f"\nstatus: {status['status'].value_counts().to_dict()}")
    bad = status[~status["status"].isin(["done", "skipped"])]
    if len(bad):
        print(bad.to_string(index=False))

    features, report = collect(cohort, work_root, load_fs_parser(fs_home))
    report = status.merge(report, on="visit_id", how="left")
    suffix = "_check" if args.visits else ""
    feat_csv = Path(cfg["feature_csv"]).with_name(Path(cfg["feature_csv"]).stem + suffix + ".csv")
    features.to_csv(feat_csv, index=False)
    dictionary(features).to_csv(cfg["column_dict_csv"], index=False)
    report.to_csv(out_dir(config) / f"gm_md_report{suffix}.csv", index=False)
    print(f"\n{len(features)} visits x {features.shape[1] - 2} regions")
    print(f"Saved: {feat_csv}\n       {cfg['column_dict_csv']}\n       "
          f"{out_dir(config) / f'gm_md_report{suffix}.csv'}")