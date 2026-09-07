import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from src.configs.config import ConfigReader
from src.shared.models.CIMLR import CIMLR

RUN_DIR = os.environ.get("RUN_DIR")

if not RUN_DIR:
    raise ValueError("RUN_DIR environment variable is not set")

RUN_DIR = Path(RUN_DIR).resolve()
RUN_NAME = RUN_DIR.name

CONFIG_PATH = PROJECT_ROOT / "logs" / RUN_NAME / "config.yml"
LATENT_PATH = RUN_DIR / "post_analysis" / "latents_best_ckpt.npz"
OUT_DIR = RUN_DIR / "post_analysis"

N_BINS = 3
CV_FOLDS = 5
DISPLAY_NAMES = {"etiv": "head size", "age": "age", "gender": "sex"}
BIN_NAMES = {3: ["small", "medium", "large"]}

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


def decodability(space, labels):
    mask = np.isfinite(labels) if labels.dtype.kind == "f" else np.ones(len(labels), dtype=bool)
    x = space[mask]
    y = labels[mask]

    if len(np.unique(y)) < 2 or len(y) < CV_FOLDS * 2:
        return float("nan")

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    x = StandardScaler().fit_transform(x)
    return float(cross_val_score(model, x, y, cv=CV_FOLDS, scoring="accuracy").mean())


def bin_continuous(values, n_bins=N_BINS):
    out = np.full(len(values), np.nan)
    finite = np.isfinite(values)

    if finite.sum() < n_bins:
        return out

    edges = np.quantile(values[finite], np.linspace(0, 1, n_bins + 1)[1:-1])
    out[finite] = np.digitize(values[finite], edges)
    return out


def plot_latent_map(mu, diagnoses, clusters, split_name, output_path):
    coords = PCA(n_components=2).fit_transform(mu)
    markers = ["o", "^", "s", "D", "P", "X"]
    colors = {"CN": "tab:blue", "AD": "tab:red"}

    plt.figure(figsize=(9, 7))

    for diagnosis in np.unique(diagnoses):
        for cluster in np.unique(clusters):
            mask = (diagnoses == diagnosis) & (clusters == cluster)

            if not np.any(mask):
                continue

            plt.scatter(
                coords[mask, 0], coords[mask, 1],
                marker=markers[int(cluster) % len(markers)],
                color=colors.get(diagnosis, "tab:gray"),
                alpha=0.7,
                label=f"{diagnosis} | cluster {cluster}",
            )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"Latent map: {split_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def scatter_panel(ax, coords, labels, title, is_diagnosis):
    if is_diagnosis:
        colors = {"CN": "tab:blue", "AD": "tab:red"}
        for value in np.unique(labels):
            mask = labels == value
            ax.scatter(coords[mask, 0], coords[mask, 1], s=14, alpha=0.7,
                       color=colors.get(value, "tab:gray"), label=str(value))
    else:
        finite = np.isfinite(labels)
        names = BIN_NAMES.get(N_BINS)
        for value in np.unique(labels[finite]):
            mask = finite & (labels == value)
            label = names[int(value)] if names else f"bin {int(value)}"
            ax.scatter(coords[mask, 0], coords[mask, 1], s=14, alpha=0.7, label=label)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)


def plot_alignment(before, after, diagnoses, nuisance, nuisance_name, output_path):
    coords_before = PCA(n_components=2).fit_transform(before)
    coords_after = PCA(n_components=2).fit_transform(after)

    diag_before = decodability(before, diagnoses)
    diag_after = decodability(after, diagnoses)
    nuis_before = decodability(before, nuisance)
    nuis_after = decodability(after, nuisance)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    scatter_panel(axes[0, 0], coords_before, diagnoses,
                  f"before — by diagnosis (acc={diag_before:.3f})", True)
    scatter_panel(axes[0, 1], coords_before, nuisance,
                  f"before — by {nuisance_name} (acc={nuis_before:.3f})", False)
    scatter_panel(axes[1, 0], coords_after, diagnoses,
                  f"after — by diagnosis (acc={diag_after:.3f})", True)
    scatter_panel(axes[1, 1], coords_after, nuisance,
                  f"after — by {nuisance_name} (acc={nuis_after:.3f})", False)

    fig.suptitle(
        f"Alignment: {nuisance_name} decodability {nuis_before:.3f} → {nuis_after:.3f} | "
        f"diagnosis {diag_before:.3f} → {diag_after:.3f}"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "nuisance": nuisance_name,
        "diagnosis_before": diag_before,
        "diagnosis_after": diag_after,
        "nuisance_before": nuis_before,
        "nuisance_after": nuis_after,
        "nuisance_drop": nuis_before - nuis_after,
        "diagnosis_change": diag_after - diag_before,
    }


if __name__ == "__main__":
    config = ConfigReader.merge(CONFIG_PATH)
    n_clusters = config.loss.n_clusters

    data = np.load(LATENT_PATH, allow_pickle=True)

    def heldout(key):
        return np.concatenate([data[f"val_{key}"], data[f"test_{key}"]], axis=0)

    mu = heldout("mu")
    features = heldout("features")
    diagnoses = np.char.upper(np.char.strip(heldout("diagnoses").astype(str)))

    clusters = cluster_mu(mu, n_clusters)
    plot_latent_map(mu, diagnoses, clusters, "heldout", OUT_DIR / "latent_map_heldout.png")
    print(f"cluster sizes = {np.bincount(clusters, minlength=n_clusters).tolist()}")

    rows = []
    for name in ["etiv", "age", "gender"]:
        values = heldout(name).astype(float)

        if not np.isfinite(values).any():
            print(f"skipped {name}: no values")
            continue

        labels = values if name == "gender" else bin_continuous(values)
        rows.append(plot_alignment(features, mu, diagnoses, labels,
                                   DISPLAY_NAMES.get(name, name),
                                   OUT_DIR / f"alignment_{name}.png"))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "alignment_decodability.csv", index=False)

    print(summary.to_string(index=False))
    print(f"\nSaved plots to: {OUT_DIR}")