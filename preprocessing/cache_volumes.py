"""Build the CNN input cache: one cube per visit, the same box for T1 and FA.

    python cache_volumes.py build

Per visit (as the original sMRI pipeline):
    brain = T1 > 0 or aligned FA > 0 (so neither image is cut)
    cube  = that brain's longest extent + volumes.margin voxels on each side, centered on it
            (zero-padded where it leaves the grid) -> same box applied to T1 and FA
    -> resized to volumes.target_shape. Every brain fills its frame, so head size is not
       visible to the model; it has to use shape (atrophy relative to the head).
    T1: min-max to 0-1 (arbitrary intensity units)   FA: kept as is, clipped to 0-1
    -> <cache_dir>/<visit_id>.npy (float32) for each modality
"""

import concurrent.futures
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from common import load_config, out_dir
from scipy.ndimage import zoom

N_JOBS = 8
CROP = "per_brain"


def paths(r, config):
    t1 = (Path(config["modalities"]["smri"]["scan_root"]) / r.adni_id / r.smri_image_id
          / "mri/brain.finalsurfs.mgz")
    fa = Path(config["paths"]["aligned_dir"]) / r.visit_id / "FA_in_t1.nii.gz"
    return t1, fa


def load(path):
    return np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)


def brain_box(t1, fa, margin):
    """Center and side length of the cube around this visit's brain."""
    idx = np.nonzero((t1 > 0) | (fa > 0))
    lo = np.array([i.min() for i in idx])
    hi = np.array([i.max() for i in idx])
    side = int((hi - lo + 1).max()) + 2 * margin
    return (lo + hi) / 2, side


def crop_cube(volume, center, side):
    """Cube of `side` voxels centered at `center`, zero-padded where it leaves the grid."""
    start = np.round(np.array(center) - side / 2).astype(int)
    out = np.zeros((side, side, side), dtype=volume.dtype)
    src_lo = np.maximum(start, 0)
    src_hi = np.minimum(start + side, volume.shape)
    dst_lo = src_lo - start
    dst_hi = dst_lo + (src_hi - src_lo)
    out[dst_lo[0]:dst_hi[0], dst_lo[1]:dst_hi[1], dst_lo[2]:dst_hi[2]] = \
        volume[src_lo[0]:src_hi[0], src_lo[1]:src_hi[1], src_lo[2]:src_hi[2]]
    return out


def resize(volume, target):
    return zoom(volume, [t / s for t, s in zip(target, volume.shape)], order=1).astype(np.float32)


def minmax(volume):
    lo, hi = volume.min(), volume.max()
    return (volume - lo) / (hi - lo) if hi > lo else np.zeros_like(volume)


def build_visit(r, config, margin, target, smri_cache, dti_cache):
    smri_out, dti_out = smri_cache / f"{r.visit_id}.npy", dti_cache / f"{r.visit_id}.npy"
    if smri_out.is_file() and dti_out.is_file():
        return {"visit_id": r.visit_id, "status": "skipped"}
    t1_path, fa_path = paths(r, config)
    if not fa_path.is_file():
        return {"visit_id": r.visit_id, "status": "no_aligned_fa"}
    t1, fa = load(t1_path), load(fa_path)
    center, side = brain_box(t1, fa, margin)
    np.save(smri_out, minmax(resize(crop_cube(t1, center, side), target)))
    np.save(dti_out, np.clip(resize(crop_cube(fa, center, side), target), 0.0, 1.0))
    return {"visit_id": r.visit_id, "status": "done", "cube_side": side}


def check_cache_info(cache, info):
    """Refuse to mix volumes made with different settings in one cache folder."""
    path = cache / "cache_info.json"
    if path.is_file():
        old = json.loads(path.read_text())
        if old != info:
            sys.exit(f"{cache} was built with {old}, config now says {info}. "
                     f"Delete that folder to rebuild.")
    else:
        path.write_text(json.dumps(info))


if __name__ == "__main__":
    if (sys.argv[1] if len(sys.argv) > 1 else "") != "build":
        sys.exit("usage: python cache_volumes.py build")
    config = load_config(sys.argv[2:])
    vol = config["volumes"]
    margin, target = vol.get("margin", 5), tuple(vol["target_shape"])

    info = {"crop": CROP, "margin": margin, "target_shape": list(target)}
    caches = []
    for name in ("smri", "dti"):
        cache = Path(config["modalities"][name]["cache_dir"])
        cache.mkdir(parents=True, exist_ok=True)
        check_cache_info(cache, info)
        caches.append(cache)

    cohort = pd.read_csv(out_dir(config) / "cohort.csv", dtype=str)
    with concurrent.futures.ThreadPoolExecutor(max_workers=N_JOBS) as pool:
        rows = list(pool.map(lambda r: build_visit(r, config, margin, target, *caches),
                             cohort.itertuples()))

    report = pd.DataFrame(rows)
    report.to_csv(out_dir(config) / "cache_report.csv", index=False)
    print(f"per-brain crop, margin {margin} -> {target}")
    print(report["status"].value_counts().to_string())
    if "cube_side" in report:
        sides = report["cube_side"].dropna()
        print(f"cube side (voxels): min {int(sides.min())}, median {int(sides.median())}, "
              f"max {int(sides.max())}")
    print(f"Saved to {caches[0]} and {caches[1]}")