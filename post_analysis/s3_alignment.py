import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

RUN_DIR = os.environ.get("RUN_DIR")

if not RUN_DIR:
    raise ValueError("RUN_DIR environment variable is not set")

RUN_DIR = Path(RUN_DIR).resolve()

LATENT_PATH = RUN_DIR / "latents_best_ckpt.npz"   
OUT_DIR = RUN_DIR / "s3_alignment"              
DECODABILITY_PATH = OUT_DIR / "alignment_decodability.csv"

N_BINS = 3
CV_FOLDS = 5
NUISANCES = ["etiv", "age", "gender"]
DISPLAY_NAMES = {"etiv": "head size", "age": "age", "gender": "sex"}
BIN_NAMES = {3: ["small", "medium", "large"]}


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
    binned = np.full(len(values), np.nan)
    finite = np.isfinite(values)

    if finite.sum() < n_bins:
        return binned

    edges = np.quantile(values[finite], np.linspace(0, 1, n_bins + 1)[1:-1])
    binned[finite] = np.digitize(values[finite], edges)
    return binned


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


def plot_alignment(before, after, diagnoses, nuisance, nuisance_name, split_name, output_path):
    coords_before = PCA(n_components=2).fit_transform(before)
    coords_after = PCA(n_components=2).fit_transform(after)

    diagnosis_before = decodability(before, diagnoses)
    diagnosis_after = decodability(after, diagnoses)
    nuisance_before = decodability(before, nuisance)
    nuisance_after = decodability(after, nuisance)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    scatter_panel(axes[0, 0], coords_before, diagnoses,
                  f"before — by diagnosis (acc={diagnosis_before:.3f})", True)
    scatter_panel(axes[0, 1], coords_before, nuisance,
                  f"before — by {nuisance_name} (acc={nuisance_before:.3f})", False)
    scatter_panel(axes[1, 0], coords_after, diagnoses,
                  f"after — by diagnosis (acc={diagnosis_after:.3f})", True)
    scatter_panel(axes[1, 1], coords_after, nuisance,
                  f"after — by {nuisance_name} (acc={nuisance_after:.3f})", False)

    fig.suptitle(
        f"Alignment ({split_name}): {nuisance_name} decodability "
        f"{nuisance_before:.3f} → {nuisance_after:.3f} | "
        f"diagnosis {diagnosis_before:.3f} → {diagnosis_after:.3f}"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "split": split_name,
        "nuisance": nuisance_name,
        "diagnosis_before": diagnosis_before,
        "diagnosis_after": diagnosis_after,
        "nuisance_before": nuisance_before,
        "nuisance_after": nuisance_after,
        "nuisance_drop": nuisance_before - nuisance_after,
        "diagnosis_change": diagnosis_after - diagnosis_before,
    }


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = np.load(LATENT_PATH, allow_pickle=True)

    def collect(splits, key):
        return np.concatenate([data[f"{split}_{key}"] for split in splits], axis=0)

    rows = []
    for split_name, splits in [("heldout", ["val", "test"]), ("test", ["test"])]:
        mu = collect(splits, "mu")
        features = collect(splits, "features")
        diagnoses = np.char.upper(np.char.strip(collect(splits, "diagnoses").astype(str)))

        for nuisance in NUISANCES:
            values = collect(splits, nuisance).astype(float)

            if not np.isfinite(values).any():
                continue

            labels = values if nuisance == "gender" else bin_continuous(values)
            rows.append(plot_alignment(
                features, mu, diagnoses, labels,
                DISPLAY_NAMES.get(nuisance, nuisance), split_name,
                OUT_DIR / f"alignment_{nuisance}_{split_name}.png",
            ))

    pd.DataFrame(rows).to_csv(DECODABILITY_PATH, index=False)

    print(f"Saved: {DECODABILITY_PATH}")
    print(f"Saved plots to: {OUT_DIR}")