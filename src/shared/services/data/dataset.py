"""Dataset over cohort visits. Everything it reads is made by preprocessing/:

    <cache_dir>/<visit_id>.npy          96^3 float32 volume per modality (cache_volumes.py)
    <prepared_dir>/features_<mod>.csv   prediction targets per modality (features.py)

modality 'smri' / 'dti' -> 1 channel (T1 / FA) and that modality's features;
modality 'both'         -> 2 channels (T1, FA) and both feature sets side by side;
modality 'fa_md'        -> 2 channels (FA, MD) and the DTI features;
modality 't1_fa_md'     -> 3 channels (T1, FA, MD) and both feature sets side by side.
Channel names are cache_dirs keys; feature names are features_<name>.csv files.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# image channels per modality (cache_dirs keys, stacked in this order)
MODALITIES = {"smri": ["smri"], "dti": ["dti"], "both": ["smri", "dti"],
              "fa_md": ["dti", "dti_md"], "t1_fa_md": ["smri", "dti", "dti_md"]}
# prediction-feature tables per modality (features_<name>.csv)
FEATURE_SETS = {"smri": ["smri"], "dti": ["dti"], "both": ["smri", "dti"],
                "fa_md": ["dti"], "t1_fa_md": ["smri", "dti"]}


def modality_list(modality):
    """Image channels of the modality."""
    if modality not in MODALITIES:
        raise ValueError(f"modality must be one of {list(MODALITIES)}, got {modality!r}")
    return MODALITIES[modality]


def feature_list(modality):
    """Feature tables of the modality."""
    modality_list(modality)                     # same check and error message
    return FEATURE_SETS[modality]


def n_channels(modality):
    return len(modality_list(modality))


def load_split(prepared_dir):
    """{split name: [visit_id, ...]} from preprocessing's split.csv, plus a short checksum of it."""
    path = Path(prepared_dir) / "split.csv"
    split = pd.read_csv(path, dtype={"visit_id": str})
    ids = {name: split.loc[split["split"] == name, "visit_id"].tolist()
           for name in ("train", "val", "test")}
    return ids, hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def load_metadata(prepared_dir):
    """Clinical info per visit (cohort.csv) plus eTIV (etiv.csv), indexed by visit_id."""
    prepared_dir = Path(prepared_dir)
    cohort = pd.read_csv(prepared_dir / "cohort.csv", dtype={"visit_id": str, "adni_id": str})
    etiv = pd.read_csv(prepared_dir / "etiv.csv", dtype={"visit_id": str})
    return cohort.merge(etiv, on="visit_id", how="left").set_index("visit_id")


def load_features(prepared_dir, modality):
    """Features of the chosen modality, indexed by visit_id (sMRI columns first when both)."""
    frames = []
    for m in feature_list(modality):
        df = pd.read_csv(Path(prepared_dir) / f"features_{m}.csv", dtype={"visit_id": str})
        df = df.drop(columns=[f"{m}_image_id"]).set_index("visit_id")
        frames.append(df.apply(pd.to_numeric, errors="coerce"))
    features = pd.concat(frames, axis=1, join="inner")

    duplicated = features.columns[features.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(f"feature names appear in more than one modality: {duplicated[:5]}")
    return features


class VisitDataset(Dataset):

    def __init__(self, visit_ids, modality, prepared_dir, cache_dirs, normalize=False):
        self.modalities = modality_list(modality)
        self.cache_dirs = {m: Path(cache_dirs[m]) for m in self.modalities}
        self.visit_ids = [str(v) for v in visit_ids]

        self.raw_features_df = load_features(prepared_dir, modality)
        self.feature_names = list(self.raw_features_df.columns)
        self.feature_mean = None
        self.feature_std = None
        self.features_df = self.raw_features_df

        missing_features = [v for v in self.visit_ids if v not in self.raw_features_df.index]
        missing_volumes = [f"{m}/{v}" for v in self.visit_ids for m in self.modalities
                           if not (self.cache_dirs[m] / f"{v}.npy").is_file()]
        if missing_features or missing_volumes:
            raise FileNotFoundError(
                f"{len(missing_features)} visits without features, {len(missing_volumes)} volumes "
                f"missing from the cache (e.g. {(missing_features + missing_volumes)[:3]}). "
                f"Run preprocessing first.")

        if normalize:
            train = self.raw_features_df.loc[self.visit_ids]
            self.set_normalization(train.mean(), train.std().replace(0, 1.0))

        print(f"{modality}: {len(self.visit_ids)} visits, {len(self.modalities)} channel(s), "
              f"{len(self.feature_names)} features")

    def set_normalization(self, mean, std):
        """Standardize features with statistics computed on the training visits only."""
        self.feature_mean = mean
        self.feature_std = std
        self.features_df = (self.raw_features_df - mean) / std

    def __len__(self):
        return len(self.visit_ids)

    def __getitem__(self, idx):
        visit_id = self.visit_ids[idx]
        volume = np.stack([np.load(self.cache_dirs[m] / f"{visit_id}.npy") for m in self.modalities])
        features = self.features_df.loc[visit_id].to_numpy(dtype=np.float32)
        return torch.from_numpy(volume.astype(np.float32)), torch.from_numpy(features), visit_id