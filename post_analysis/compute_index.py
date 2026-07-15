import os
from pathlib import Path

import numpy as np
import pandas as pd


RUN_DIR = os.environ.get("RUN_DIR")

if not RUN_DIR:
    raise ValueError("RUN_DIR environment variable is not set")

RUN_DIR = Path(RUN_DIR).resolve()

LATENT_PATH = RUN_DIR / "post_analysis" / "latents_best_ckpt.npz"
OUTPUT_PATH = RUN_DIR / "post_analysis" / "disease_index.csv"


def compute_projection(mu, cn_mean, ad_mean):
    direction = ad_mean - cn_mean
    return ((mu - cn_mean) @ direction) / (direction @ direction)


def compute_off_axis_distance(mu, projection, cn_mean, ad_mean):
    direction = ad_mean - cn_mean
    projected_points = cn_mean + projection[:, None] * direction
    return np.linalg.norm(mu - projected_points, axis=1)


def build_split_df(split, image_ids, diagnoses, mu, cn_mean, ad_mean):
    l2_norm = np.linalg.norm(mu, axis=1)
    projection = compute_projection(mu, cn_mean, ad_mean)
    off_axis_distance = compute_off_axis_distance(mu, projection, cn_mean, ad_mean)

    return pd.DataFrame({
        "split": split,
        "image_id": image_ids.astype(str),
        "diagnosis": diagnoses.astype(str),
        "l2_norm": l2_norm,
        "cn_ad_projection": projection,
        "off_axis_distance": off_axis_distance,
    })


if __name__ == "__main__":
    data = np.load(LATENT_PATH, allow_pickle=True)

    train_mu = data["train_mu"]
    train_diagnoses = data["train_diagnoses"].astype(str)

    cn_mean = train_mu[train_diagnoses == "cn"].mean(axis=0)
    ad_mean = train_mu[train_diagnoses == "ad"].mean(axis=0)

    train_df = build_split_df(
        "train",
        data["train_image_ids"],
        data["train_diagnoses"],
        data["train_mu"],
        cn_mean,
        ad_mean,
    )

    val_df = build_split_df(
        "val",
        data["val_image_ids"],
        data["val_diagnoses"],
        data["val_mu"],
        cn_mean,
        ad_mean,
    )

    test_df = build_split_df(
        "test",
        data["test_image_ids"],
        data["test_diagnoses"],
        data["test_mu"],
        cn_mean,
        ad_mean,
    )

    result = pd.concat([train_df, val_df, test_df], ignore_index=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Train CN mean projection: {train_df.loc[train_df['diagnosis'] == 'cn', 'cn_ad_projection'].mean():.4f}")
    print(f"Train AD mean projection: {train_df.loc[train_df['diagnosis'] == 'ad', 'cn_ad_projection'].mean():.4f}")