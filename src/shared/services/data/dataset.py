import os
import numpy as np
import torch
import nibabel as nib
import pandas as pd
from torch.utils.data import Dataset
from typing import Optional
from .transforms import BaseTransformer


class MRIDataset(Dataset):

    def __init__(self, data_dir: str, feature_csv_path: str,
        transform: Optional[BaseTransformer] = None,
        cache_dir: Optional[str] = None, max_samples: Optional[int] = None,
        image_ids: Optional[list] = None, normalize: bool = True,
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.cache_dir = cache_dir
        self.allowed_ids = set(str(i) for i in image_ids) if image_ids is not None else None

        self.features_df, self.feature_names = self._load_features(feature_csv_path)
        self.feature_mean = None
        self.feature_std = None
        if normalize:
            self.set_normalization(self.raw_features_df.mean(),
                                   self.raw_features_df.std().replace(0, 1.0))
        print(f"Loaded {len(self.feature_names)} features for {len(self.features_df)} images")

        self.samples = self._scan_files(max_samples)

    def set_normalization(self, mean, std):
        self.feature_mean = mean
        self.feature_std = std
        self.features_df = (self.raw_features_df - mean) / std


    def _load_features(self, feature_csv_path):
        df = pd.read_csv(feature_csv_path)
        df = df.set_index('image_id')
        df.index = df.index.astype(str)
        df = df.drop(columns=['subject_id'], errors='ignore')
        df = df.select_dtypes(include=[np.number])
        self.raw_features_df = df
        return df, list(df.columns)

    def _scan_files(self, max_samples):
        samples = []
        for subject_id in sorted(os.listdir(self.data_dir)):
            subject_path = os.path.join(self.data_dir, subject_id)
            if not os.path.isdir(subject_path):
                continue

            for image_id in sorted(os.listdir(subject_path)):
                if image_id == "fsaverage":
                    continue
                if self.allowed_ids is not None and image_id not in self.allowed_ids:
                    continue
                mgz_path = os.path.join(subject_path, image_id, "mri", "brain.finalsurfs.mgz")
                if os.path.exists(mgz_path) and image_id in self.features_df.index:
                    samples.append({
                        "mgz_path": mgz_path,
                        "image_id": image_id,
                    })

                if max_samples and len(samples) >= max_samples:
                    return samples

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):  # ty:ignore[invalid-method-override]
        sample = self.samples[idx]
        image_id = sample["image_id"]

        if self.cache_dir:
            npy_path = os.path.join(self.cache_dir, f"{image_id}.npy")

            if os.path.exists(npy_path):
                volume = np.load(npy_path)
            else:
                volume = self._load_mgz(sample["mgz_path"])
                if self.transform:
                    volume = self.transform(volume)
                    os.makedirs(self.cache_dir, exist_ok=True)
                    np.save(npy_path, volume)
        else:
            volume = self._load_mgz(sample["mgz_path"])
            if self.transform:
                volume = self.transform(volume)

        volume = volume[np.newaxis, ...].astype(np.float32)
        volume = torch.from_numpy(volume)

        features = self.features_df.loc[image_id].values.astype(np.float32)
        features = torch.from_numpy(features)

        return volume, features, image_id

    def _load_mgz(self, path):
        img = nib.load(path)
        return img.get_fdata(dtype=np.float32)