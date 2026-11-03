"""Alignment: what the encoder keeps and what it removes, compared with the original images.

Spaces (one row per visit):
    before    PCA of the input volumes (2x downsampled, all channels), N_COMPONENTS
              components, fitted on train
    after     latent mean (mu) from the best checkpoint
    features  FreeSurfer / DTI feature table (reference only: CSV, not plotted)

Decodability of diagnosis, head size (eTIV), age and sex in each space:
    logistic regression, regularization chosen by CV on train, fitted on train,
    scored on held-out visits (balanced accuracy; chance = 1 / number of classes).
    Head size and age are cut into N_BINS bins at train quantiles.

Outputs (RUN_DIR/s3_alignment):
    alignment_<nuisance>_<split>.png   2 x 2: before / after, by diagnosis / by nuisance
    alignment_decodability.csv
"""

import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

RUN_DIR = os.environ.get("RUN_DIR")

if not RUN_DIR:
    raise ValueError("RUN_DIR environment variable is not set")

RUN_DIR = Path(RUN_DIR).resolve()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_CONFIG = yaml.safe_load(open(PROJECT_ROOT / "logs" / RUN_DIR.name / "config.yml"))

LATENT_PATH = RUN_DIR / "latents_best_ckpt.npz"
OUT_DIR = RUN_DIR / "s3_alignment"
DECODABILITY_PATH = OUT_DIR / "alignment_decodability.csv"

MODALITY_CHANNELS = {"smri": ["smri"], "dti": ["dti"], "both": ["smri", "dti"]}
DOWNSAMPLE = 2
N_COMPONENTS = 128
N_BINS = 3
CV_FOLDS = 5
SEED = 42
NUISANCES = ["etiv", "age", "gender"]
DISPLAY_NAMES = {"etiv": "head size", "age": "age", "gender": "sex"}
BIN_NAMES = ["low", "middle", "high"]
EVAL_SETS = [("heldout", ["val", "test"]), ("test", ["test"])]


def load_volume(visit_id, channels, cache_dirs):
    """All channels of one visit, each 2x average-pooled, flattened into one vector."""
    parts = []
    for channel in channels:
        v = np.load(Path(cache_dirs[channel]) / f"{visit_id}.npy").astype(np.float32)
        d = DOWNSAMPLE
        s = [n // d * d for n in v.shape]
        v = v[:s[0], :s[1], :s[2]]
        v = v.reshape(s[0] // d, d, s[1] // d, d, s[2] // d, d).mean(axis=(1, 3, 5))
        parts.append(v.ravel())
    return np.concatenate(parts)


def image_space(visit_ids_by_split):
    """PCA of the input volumes, fitted on train, applied to every split."""
    channels = MODALITY_CHANNELS[RUN_CONFIG["data"]["modality"]]
    cache_dirs = RUN_CONFIG["data"]["cache_dirs"]
    volumes = {split: np.stack([load_volume(v, channels, cache_dirs) for v in ids])
               for split, ids in visit_ids_by_split.items()}
    pca = PCA(n_components=N_COMPONENTS, svd_solver="randomized", random_state=SEED)
    pca.fit(volumes["train"])
    print(f"image PCA: {N_COMPONENTS} components, "
          f"{pca.explained_variance_ratio_.sum():.1%} of train variance")
    return {split: pca.transform(x) for split, x in volumes.items()}


def bin_edges(train_values):
    finite = train_values[np.isfinite(train_values)]
    return np.quantile(finite, np.linspace(0, 1, N_BINS + 1)[1:-1])


def to_labels(values, edges):
    labels = np.full(len(values), np.nan)
    finite = np.isfinite(values)
    labels[finite] = np.digitize(values[finite], edges)
    return labels


def decodability(train_x, train_y, eval_x, eval_y):
    """Probe fitted on train, scored on the evaluation visits (balanced accuracy)."""
    def valid(y):
        return np.isfinite(y) if y.dtype.kind == "f" else np.ones(len(y), dtype=bool)

    tm, em = valid(train_y), valid(eval_y)
    if len(np.unique(train_y[tm])) < 2 or len(np.unique(eval_y[em])) < 2:
        return float("nan")

    scaler = StandardScaler().fit(train_x[tm])
    probe = LogisticRegressionCV(Cs=10, cv=CV_FOLDS, class_weight="balanced",
                                 max_iter=5000, random_state=SEED)
    probe.fit(scaler.transform(train_x[tm]), train_y[tm])
    pred = probe.predict(scaler.transform(eval_x[em]))
    return float(balanced_accuracy_score(eval_y[em], pred))


def scatter_panel(ax, coords, labels, title, kind):
    if kind == "diagnosis":
        colors = {"CN": "tab:blue", "AD": "tab:red"}
        for value in np.unique(labels):
            mask = labels == value
            ax.scatter(coords[mask, 0], coords[mask, 1], s=14, alpha=0.7,
                       color=colors.get(value, "tab:gray"), label=str(value))
    else:
        finite = np.isfinite(labels)
        for value in np.unique(labels[finite]):
            mask = finite & (labels == value)
            label = f"sex {int(value)}" if kind == "sex" else BIN_NAMES[int(value)]
            ax.scatter(coords[mask, 0], coords[mask, 1], s=14, alpha=0.7, label=label)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)


def plot_alignment(coords, scores, diagnoses, nuisance, nuisance_name, split_name, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    kind = "sex" if nuisance_name == "sex" else "bins"
    n_classes = len(np.unique(nuisance[np.isfinite(nuisance)]))

    for row, space in enumerate(["before", "after"]):
        scatter_panel(axes[row, 0], coords[space], diagnoses,
                      f"{space} — by diagnosis (bal. acc = {scores[space]['diagnosis']:.3f}, "
                      f"chance 0.50)", "diagnosis")
        scatter_panel(axes[row, 1], coords[space], nuisance,
                      f"{space} — by {nuisance_name} (bal. acc = {scores[space]['nuisance']:.3f}, "
                      f"chance {1 / n_classes:.2f})", kind)

    fig.suptitle(
        f"Alignment ({split_name}), image space → latent: {nuisance_name} "
        f"{scores['before']['nuisance']:.3f} → {scores['after']['nuisance']:.3f} | "
        f"diagnosis {scores['before']['diagnosis']:.3f} → {scores['after']['diagnosis']:.3f}"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(LATENT_PATH, allow_pickle=True)
    splits = ["train", "val", "test"]

    spaces = {
        "before": image_space({s: data[f"{s}_visit_ids"] for s in splits}),
        "after": {s: data[f"{s}_mu"] for s in splits},
        "features": {s: data[f"{s}_features"] for s in splits},
    }
    diagnoses = {s: np.char.upper(np.char.strip(data[f"{s}_diagnoses"].astype(str))) for s in splits}

    nuisance_labels = {}
    for nuisance in NUISANCES:
        values = {s: data[f"{s}_{nuisance}"].astype(float) for s in splits}
        if nuisance == "gender":
            nuisance_labels[nuisance] = values
        else:
            edges = bin_edges(values["train"])
            nuisance_labels[nuisance] = {s: to_labels(v, edges) for s, v in values.items()}

    # 2D display coordinates: PCA fitted on train, per space
    display = {}
    for space in ("before", "after"):
        scaler = StandardScaler().fit(spaces[space]["train"])
        pca2 = PCA(n_components=2).fit(scaler.transform(spaces[space]["train"]))
        display[space] = {s: pca2.transform(scaler.transform(x)) for s, x in spaces[space].items()}

    def cat(d, eval_splits):
        return np.concatenate([d[s] for s in eval_splits], axis=0)

    rows = []
    for split_name, eval_splits in EVAL_SETS:
        dx = cat(diagnoses, eval_splits)
        diag_scores = {space: decodability(spaces[space]["train"], diagnoses["train"],
                                           cat(spaces[space], eval_splits), dx)
                       for space in spaces}

        for nuisance in NUISANCES:
            labels = nuisance_labels[nuisance]
            if not np.isfinite(cat(labels, eval_splits)).any():
                continue
            name = DISPLAY_NAMES[nuisance]
            ny = cat(labels, eval_splits)
            nuis_scores = {space: decodability(spaces[space]["train"], labels["train"],
                                               cat(spaces[space], eval_splits), ny)
                           for space in spaces}
            scores = {space: {"diagnosis": diag_scores[space], "nuisance": nuis_scores[space]}
                      for space in spaces}

            plot_alignment({space: cat(display[space], eval_splits) for space in display},
                           scores, cat(diagnoses, eval_splits), ny, name, split_name,
                           OUT_DIR / f"alignment_{nuisance}_{split_name}.png")

            n_classes = len(np.unique(ny[np.isfinite(ny)]))
            rows.append({
                "split": split_name,
                "nuisance": name,
                "diagnosis_before": diag_scores["before"],
                "diagnosis_after": diag_scores["after"],
                "nuisance_before": nuis_scores["before"],
                "nuisance_after": nuis_scores["after"],
                "nuisance_drop": nuis_scores["before"] - nuis_scores["after"],
                "diagnosis_change": diag_scores["after"] - diag_scores["before"],
                "diagnosis_chance": 0.5,
                "nuisance_chance": 1 / n_classes,
                "diagnosis_features": diag_scores["features"],
                "nuisance_features": nuis_scores["features"],
            })

    pd.DataFrame(rows).to_csv(DECODABILITY_PATH, index=False)

    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(f"Saved: {DECODABILITY_PATH}")
    print(f"Saved plots to: {OUT_DIR}")