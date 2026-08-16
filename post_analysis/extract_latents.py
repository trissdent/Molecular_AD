import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.config import ConfigReader
from src.shared.services.data.dataset import MRIDataset
from src.shared.services.data.transforms import MRITransformer
from src.shared.services.models_hub.beta_tc_vae.model import BetaTCVAE

RUN_DIR = os.environ.get("RUN_DIR")

if not RUN_DIR:
    raise ValueError("RUN_DIR environment variable is not set")

RUN_DIR = Path(RUN_DIR).resolve()
RUN_NAME = RUN_DIR.name

CONFIG_PATH = PROJECT_ROOT / "logs" / RUN_NAME / "config.yml"
CHECKPOINT_NAME = os.environ.get("CHECKPOINT", "best")
CHECKPOINT_PATH = RUN_DIR / f"{CHECKPOINT_NAME}.ckpt"
SPLIT_PATH = RUN_DIR / "split.json"
FEATURE_MEAN_PATH = RUN_DIR / "feature_mean.csv"
FEATURE_STD_PATH = RUN_DIR / "feature_std.csv"

OUTPUT_DIR = RUN_DIR / "post_analysis"
OUTPUT_PATH = OUTPUT_DIR / f"latents_{CHECKPOINT_NAME}_ckpt.npz"
DEMOGRAPHICS_PATH = PROJECT_ROOT / "data" / "all_demographics.csv"

ETIV_COL = "aseg_EstimatedTotalIntraCranialVol"


def build_metadata(demographics_path, feature_csv_path):
    demographics = pd.read_csv(demographics_path)
    demographics["image_id"] = demographics["image_id"].astype(str)

    record_date = pd.to_datetime(demographics["record_date"], errors="coerce")
    demographics["age"] = record_date.dt.year - demographics["birth_year"]

    features = pd.read_csv(feature_csv_path, usecols=["image_id", ETIV_COL])
    features["image_id"] = features["image_id"].astype(str)
    features = features.rename(columns={ETIV_COL: "etiv"})

    merged = demographics.merge(features, on="image_id", how="left")
    return merged.set_index("image_id")


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
    state_dict = ckpt.get("state_dict", ckpt)

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
def extract_split(model, dataset, metadata, batch_size, num_workers, device):
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
        image_ids.extend([str(image_id) for image_id in iid])

    meta = metadata.reindex(image_ids)

    return {
        "mu": np.concatenate(mus, axis=0),
        "logvar": np.concatenate(logvars, axis=0),
        "z": np.concatenate(zs, axis=0),
        "features": np.concatenate(features, axis=0),
        "image_ids": np.array(image_ids),
        "diagnoses": meta["diagnosis"].astype(str).to_numpy(),
        "mmse": meta["mmse"].to_numpy(dtype=float),
        "age": meta["age"].to_numpy(dtype=float),
        "gender": meta["gender"].to_numpy(dtype=float),
        "etiv": meta["etiv"].to_numpy(dtype=float),
    }


if __name__ == "__main__":

    config = ConfigReader.merge(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(SPLIT_PATH, "r") as f:
        split = json.load(f)

    metadata = build_metadata(DEMOGRAPHICS_PATH, config.data.feature_csv_path)

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

    train = extract_split(model, train_ds, metadata, config.data.batch_size, config.data.num_workers, device)
    val = extract_split(model, val_ds, metadata, config.data.batch_size, config.data.num_workers, device)
    test = extract_split(model, test_ds, metadata, config.data.batch_size, config.data.num_workers, device)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    arrays = {"feature_names": np.array(train_ds.feature_names)}
    for name, data in [("train", train), ("val", val), ("test", test)]:
        for key, value in data.items():
            arrays[f"{name}_{key}"] = value

    np.savez_compressed(OUTPUT_PATH, **arrays)

    print(f"Saved: {OUTPUT_PATH}")
    for name, data in [("train", train), ("val", val), ("test", test)]:
        n_mmse = int(np.isfinite(data["mmse"]).sum())
        n_etiv = int(np.isfinite(data["etiv"]).sum())
        print(f"{name}: mu={data['mu'].shape} mmse={n_mmse}/{len(data['mmse'])} etiv={n_etiv}/{len(data['etiv'])}")