import os
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path.cwd().parent))

from src.configs.config import ConfigReader
from src.shared.services.data.dataset import MRIDataset
from src.shared.services.data.transforms import MRITransformer
from src.shared.services.models_hub.beta_tc_vae.model import BetaTCVAE


CONFIG_PATH = "configs/defaults.yaml"

RUN_DIR = "../checkpoints/2026-06-24_18-43-35"

CHECKPOINT_PATH = os.path.join(RUN_DIR, "best.ckpt")
SPLIT_PATH = os.path.join(RUN_DIR, "split.json")
FEATURE_MEAN_PATH = os.path.join(RUN_DIR, "feature_mean.csv")
FEATURE_STD_PATH = os.path.join(RUN_DIR, "feature_std.csv")

OUTPUT_PATH = "../output/analyze_dci/latents_best_ckpt.npz"


def load_split():
    with open(SPLIT_PATH, "r") as f:
        return json.load(f)


def make_dataset(config, image_ids, mean, std):
    transform = MRITransformer(
        target_shape=tuple(config.transform.target_shape),
        margin=config.transform.margin,
    )

    dataset = MRIDataset(
        data_dir=config.data.data_dir,
        feature_csv_path=config.data.feature_csv_path,
        transform=transform,
        cache_dir=config.data.cache_dir,
        image_ids=image_ids,
        normalize=False,
    )
    dataset.set_normalization(mean, std)
    return dataset


def load_model(config, num_features, ckpt_path, device):
    model = BetaTCVAE(
        z_dim=config.model.z_dim,
        in_channels=config.model.in_channels,
        num_features=num_features,
        cluster_projection_dim=config.model.cluster_projection_dim,
        input_size=config.transform.target_shape[0],
    )

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            cleaned_state_dict[k.replace("model.", "", 1)] = v
        else:
            cleaned_state_dict[k] = v

    model.load_state_dict(cleaned_state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_split(model, dataset, batch_size, num_workers, device):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    mus = []
    logvars = []
    zs = []
    features = []
    image_ids = []

    for x, feat, iid in loader:
        x = x.to(device)

        out = model(x)

        mus.append(out["mu"].detach().cpu().numpy())
        logvars.append(out["logvar"].detach().cpu().numpy())
        zs.append(out["z"].detach().cpu().numpy())
        features.append(feat.detach().cpu().numpy())
        image_ids.extend(list(iid))

    return {
        "mu": np.concatenate(mus, axis=0),
        "logvar": np.concatenate(logvars, axis=0),
        "z": np.concatenate(zs, axis=0),
        "features": np.concatenate(features, axis=0),
        "image_ids": np.array(image_ids),
    }


def main():
    config = ConfigReader.merge(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split = load_split()

    mean = pd.read_csv(FEATURE_MEAN_PATH, index_col=0).squeeze("columns")
    std = pd.read_csv(FEATURE_STD_PATH, index_col=0).squeeze("columns")

    train_ds = make_dataset(config, split["train"], mean, std)
    val_ds = make_dataset(config, split["val"], mean, std)
    test_ds = make_dataset(config, split["test"], mean, std)

    model = load_model(
        config=config,
        num_features=len(train_ds.feature_names),
        ckpt_path=CHECKPOINT_PATH,
        device=device,
    )

    train = extract_split(model, train_ds, config.data.batch_size, config.data.num_workers, device)
    val = extract_split(model, val_ds, config.data.batch_size, config.data.num_workers, device)
    test = extract_split(model, test_ds, config.data.batch_size, config.data.num_workers, device)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    np.savez_compressed(
        OUTPUT_PATH,
        train_mu=train["mu"],
        train_logvar=train["logvar"],
        train_z=train["z"],
        train_features=train["features"],
        train_image_ids=train["image_ids"],

        val_mu=val["mu"],
        val_logvar=val["logvar"],
        val_z=val["z"],
        val_features=val["features"],
        val_image_ids=val["image_ids"],

        test_mu=test["mu"],
        test_logvar=test["logvar"],
        test_z=test["z"],
        test_features=test["features"],
        test_image_ids=test["image_ids"],

        feature_names=np.array(train_ds.feature_names),
    )

    print(f"Saved: {OUTPUT_PATH}")
    print(f"train: {train['mu'].shape}")
    print(f"val:   {val['mu'].shape}")
    print(f"test:  {test['mu'].shape}")


if __name__ == "__main__":
    main()