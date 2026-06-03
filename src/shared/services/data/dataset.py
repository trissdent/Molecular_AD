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
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.cache_dir = cache_dir

        # Load features
        df = pd.read_csv(feature_csv_path)
        df = df.set_index('image_id')
        df = df.drop(columns=['subject_id'], errors='ignore')
        df = df.select_dtypes(include=[np.number])
        self.features_df = df
        self.feature_names = list(df.columns)
        print(f"Loaded {len(self.feature_names)} features for {len(df)} images")

        self.samples = self._scan_files(max_samples)

    def _scan_files(self, max_samples):
        samples = []
        for subject_id in sorted(os.listdir(self.data_dir)):
            subject_path = os.path.join(self.data_dir, subject_id)
            if not os.path.isdir(subject_path):
                continue

            for image_id in sorted(os.listdir(subject_path)):
                if image_id == "fsaverage":
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

    def __getitem__(self, idx):
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