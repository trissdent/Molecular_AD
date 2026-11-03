"""Extract latents for every split from a trained run.

    RUN_DIR=../checkpoints/<run_name> python test_pipeline.py
    (optional: CHECKPOINT=last, default best)

Everything is read from the run itself (its saved config.yml, feature mean/std)
and from preprocessing's outputs (split.csv, cohort.csv, etiv.csv).
Writes <run_dir>/latents_<checkpoint>_ckpt.npz for post_analysis.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from configs.config import ConfigReader
from shared.services.data import VisitDataset, load_metadata, load_split, n_channels
from shared.services.models_hub.beta_tc_vae.model import BetaTCVAE
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_dataset(config, visit_ids, mean, std):
    dataset = VisitDataset(visit_ids, config.data.modality, config.data.prepared_dir,
                           vars(config.data.cache_dirs), normalize=False)
    dataset.set_normalization(mean, std)
    return dataset


def load_model(config, num_features, ckpt_path, device):
    model = BetaTCVAE(
        z_dim=config.model.z_dim,
        in_channels=n_channels(config.data.modality),
        num_features=num_features,
        cluster_projection_dim=config.model.cluster_projection_dim,
        input_size=config.model.input_size,
        salient_dim=int(config.model.salient_dim) if getattr(config.model, "contrastive", False) else 0,
    )

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    cleaned = {k.replace("model.", "", 1) if k.startswith("model.") else k: v
               for k, v in state_dict.items()}

    model.load_state_dict(cleaned, strict=True)
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

    mus, logvars, zs, features, visit_ids = [], [], [], [], []
    for x, feat, vid in loader:
        out = model(x.to(device))
        mus.append(out["mu"].cpu().numpy())
        logvars.append(out["logvar"].cpu().numpy())
        zs.append(out["z"].cpu().numpy())
        features.append(feat.cpu().numpy())
        visit_ids.extend(str(v) for v in vid)

    meta = metadata.reindex(visit_ids)
    return {
        "mu": np.concatenate(mus, axis=0),
        "logvar": np.concatenate(logvars, axis=0),
        "z": np.concatenate(zs, axis=0),
        "features": np.concatenate(features, axis=0),
        "visit_ids": np.array(visit_ids),
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
    output_path = run_dir / f"latents_{checkpoint_name}_ckpt.npz"

    config = ConfigReader.merge(str(config_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split, split_sha = load_split(config.data.prepared_dir)
    trained_sha = getattr(config.data, "split_sha", None)
    if trained_sha != split_sha:
        raise ValueError(f"split.csv changed since this run was trained "
                         f"(trained with {trained_sha}, now {split_sha}).")

    metadata = load_metadata(config.data.prepared_dir)
    mean = pd.read_csv(run_dir / "feature_mean.csv", index_col=0).squeeze("columns")
    std = pd.read_csv(run_dir / "feature_std.csv", index_col=0).squeeze("columns")

    datasets = {name: make_dataset(config, split[name], mean, std)
                for name in ("train", "val", "test")}
    feature_names = datasets["train"].feature_names

    model = load_model(config, len(feature_names), checkpoint_path, device)

    arrays = {"feature_names": np.array(feature_names),
              "modality": np.array(config.data.modality)}
    for name, dataset in datasets.items():
        data = extract_split(model, dataset, metadata,
                             config.data.batch_size, config.data.num_workers, device)
        for key, value in data.items():
            arrays[f"{name}_{key}"] = value
        n_mmse = int(np.isfinite(data["mmse"]).sum())
        n_etiv = int(np.isfinite(data["etiv"]).sum())
        print(f"{name}: mu={data['mu'].shape} mmse={n_mmse}/{len(data['mmse'])} "
              f"etiv={n_etiv}/{len(data['etiv'])}")

    np.savez_compressed(output_path, **arrays)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    run()