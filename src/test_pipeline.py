import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from configs.config import ConfigReader
from shared.services.data.dataset import MRIDataset
from shared.services.data.transforms import MRITransformer
from shared.services.models_hub.beta_tc_vae.model import BetaTCVAE
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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


def run(run_dir=None, checkpoint_name=None):
    run_dir = run_dir or os.environ.get("RUN_DIR")

    if not run_dir:
        raise ValueError("RUN_DIR environment variable is not set")

    run_dir = Path(run_dir).resolve()
    checkpoint_name = checkpoint_name or os.environ.get("CHECKPOINT", "best")

    config_path = PROJECT_ROOT / "logs" / run_dir.name / "config.yml"
    checkpoint_path = run_dir / f"{checkpoint_name}.ckpt"
    split_path = run_dir / "split.json"
    feature_mean_path = run_dir / "feature_mean.csv"
    feature_std_path = run_dir / "feature_std.csv"
    output_path = run_dir /  f"latents_{checkpoint_name}_ckpt.npz"

    config = ConfigReader.merge(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(split_path, "r") as f:
        split = json.load(f)

    metadata = build_metadata(DEMOGRAPHICS_PATH, config.data.feature_csv_path)

    mean = pd.read_csv(feature_mean_path, index_col=0).squeeze("columns")
    std = pd.read_csv(feature_std_path, index_col=0).squeeze("columns")

    train_ds = make_dataset(config, split["train"], mean, std)
    val_ds = make_dataset(config, split["val"], mean, std)
    test_ds = make_dataset(config, split["test"], mean, std)

    model = load_model(
        config=config,
        num_features=len(train_ds.feature_names),
        ckpt_path=checkpoint_path,
        device=device,
    )

    splits = []
    for name, dataset in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        data = extract_split(
            model, dataset, metadata,
            config.data.batch_size, config.data.num_workers, device,
        )
        splits.append((name, data))

    arrays = {"feature_names": np.array(train_ds.feature_names)}
    for name, data in splits:
        for key, value in data.items():
            arrays[f"{name}_{key}"] = value

    np.savez_compressed(output_path, **arrays)

    print(f"Saved: {output_path}")
    for name, data in splits:
        n_mmse = int(np.isfinite(data["mmse"]).sum())
        n_etiv = int(np.isfinite(data["etiv"]).sum())
        print(f"{name}: mu={data['mu'].shape} mmse={n_mmse}/{len(data['mmse'])} etiv={n_etiv}/{len(data['etiv'])}")


if __name__ == "__main__":
    run()