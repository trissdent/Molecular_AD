import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from src.configs.config import ConfigReader  # noqa: E402
from src.shared.models.CIMLR import CIMLR  # noqa: E402


RUN_DIR = os.environ.get("RUN_DIR")

if not RUN_DIR:
    raise ValueError("RUN_DIR environment variable is not set")

RUN_DIR = Path(RUN_DIR).resolve()
RUN_NAME = RUN_DIR.name

CONFIG_PATH = PROJECT_ROOT / "logs" / RUN_NAME / "config.yml"
CHECKPOINT_PATH = RUN_DIR / "best.ckpt"
LATENT_PATH = RUN_DIR / "post_analysis" / "latents_best_ckpt.npz"
OUT_DIR = RUN_DIR / "post_analysis"
LOG_PATH = PROJECT_ROOT / "logs" / RUN_NAME / "log.txt"


def get_cluster_count_at_epoch(log_path, best_epoch, initial_n_clusters):
    with open(log_path, "r") as f:
        lines = f.readlines()

    n_clusters = initial_n_clusters

    for line in lines:
        cluster_match = re.search(r"\[ClusterCount\] c: \d+ → (\d+)", line)
        if cluster_match:
            n_clusters = int(cluster_match.group(1))

        epoch_match = re.search(r"\[epoch (\d+)\]", line)
        if epoch_match and int(epoch_match.group(1)) >= best_epoch:
            break

    return n_clusters


def cluster_mu(mu, n_clusters):
    mu = mu.astype("float64")
    n = mu.shape[0]

    if n < max(n_clusters + 2, 4):
        raise ValueError(f"Not enough samples for clustering: {n}")

    k = min(10, n - 2)

    _, latent_factor, _, _ = CIMLR([mu], n_clusters, k=k)
    latent_factor = np.real(latent_factor)

    gmm = GaussianMixture(n_components=n_clusters, covariance_type="diag", reg_covar=1e-4, random_state=42)
    gmm.fit(latent_factor)

    return gmm.predict_proba(latent_factor).argmax(axis=1)


def plot_latent_map(mu, diagnoses, clusters, split_name, output_path):
    coords = PCA(n_components=2).fit_transform(mu)
    markers = ["o", "^", "s", "D", "P", "X"]

    plt.figure(figsize=(9, 7))

    for diagnosis in np.unique(diagnoses):
        for cluster in np.unique(clusters):
            mask = (diagnoses == diagnosis) & (clusters == cluster)

            if not np.any(mask):
                continue

            plt.scatter(coords[mask, 0], coords[mask, 1], marker=markers[int(cluster) % len(markers)], alpha=0.7, label=f"{diagnosis} | Cluster {cluster}")

    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(f"Latent map: {split_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    config = ConfigReader.merge(CONFIG_PATH)

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    best_epoch = int(ckpt["epoch"])

    n_clusters = get_cluster_count_at_epoch(LOG_PATH, best_epoch, config.loss.n_clusters)

    data = np.load(LATENT_PATH, allow_pickle=True)

    train_diagnoses = np.char.upper(np.char.strip(data["train_diagnoses"].astype(str)))
    val_diagnoses = np.char.upper(np.char.strip(data["val_diagnoses"].astype(str)))
    test_diagnoses = np.char.upper(np.char.strip(data["test_diagnoses"].astype(str)))

    splits = {
        "train": (data["train_mu"], train_diagnoses),
        "val": (data["val_mu"], val_diagnoses),
        "test": (data["test_mu"], test_diagnoses),
        "heldout": (
            np.concatenate([data["val_mu"], data["test_mu"]], axis=0),
            np.concatenate([val_diagnoses, test_diagnoses]),
        ),
    }

    print(f"Best epoch: {best_epoch}")
    print(f"Cluster count at best epoch: {n_clusters}")

    for split_name, (mu, diagnoses) in splits.items():
        clusters = cluster_mu(mu, n_clusters)
        output_path = OUT_DIR / f"latent_map_{split_name}.png"

        plot_latent_map(mu, diagnoses, clusters, split_name, output_path)

        print(f"{split_name}: cluster sizes = {np.bincount(clusters, minlength=n_clusters).tolist()}")

    print(f"Saved latent maps to: {OUT_DIR}")