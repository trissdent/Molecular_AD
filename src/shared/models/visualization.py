import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def _read_metrics(log_dir):
    log_path = Path(log_dir) / "metrics.csv"
    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        return None

    df = pd.read_csv(log_path)

    if "epoch" not in df.columns:
        print(f"No epoch column found in: {log_path}")
        return None

    # Lightning logs train/val on different rows.
    # Group by epoch and keep last non-null value for each metric.
    df = df.groupby("epoch", as_index=False).last()
    return df


def _plot_pair(df, train_col, val_col, save_path, title, ylabel):
    if train_col not in df.columns and val_col not in df.columns:
        print(f"Metric not found: {train_col}, {val_col}")
        return

    plt.figure(figsize=(8, 5))

    if train_col in df.columns:
        s = df[["epoch", train_col]].dropna()
        if len(s) > 0:
            plt.plot(s["epoch"], s[train_col], label=train_col)

    if val_col in df.columns:
        s = df[["epoch", val_col]].dropna()
        if len(s) > 0:
            plt.plot(s["epoch"], s[val_col], label=val_col)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved: {save_path}")


def plot_training_curves(log_dir="./logs/experiment/version_0/", save_dir="./"):
    df = _read_metrics(log_dir)
    if df is None:
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    pairs = [
        ("train_loss", "val_loss", "loss_curve.png", "Total Loss", "Loss"),
        ("train_recon", "val_recon", "recon_curve.png", "Reconstruction Loss", "Recon"),
        ("train_kl", "val_kl", "kl_curve.png", "Weighted KL Loss", "KL loss"),
        ("train_dim_kl", "val_dim_kl", "dim_kl_curve.png", "Raw KL", "Raw KL"),
        ("train_pred", "val_pred", "pred_curve.png", "Prediction Loss", "Prediction"),
        ("train_D", "val_D", "dci_D_curve.png", "DCI Disentanglement", "D"),
        ("train_I", "val_I", "dci_I_curve.png", "DCI Informativeness", "I"),
        ("train_C", "val_C", "dci_C_curve.png", "DCI Completeness", "C"),
    ]

    for train_col, val_col, fname, title, ylabel in pairs:
        _plot_pair(
            df=df,
            train_col=train_col,
            val_col=val_col,
            save_path=save_dir / fname,
            title=title,
            ylabel=ylabel,
        )

    if "stable_dims" in df.columns:
        s = df[["epoch", "stable_dims"]].dropna()
        if len(s) > 0:
            plt.figure(figsize=(8, 5))
            plt.plot(s["epoch"], s["stable_dims"], label="stable_dims")
            plt.title("Stable Dimensions")
            plt.xlabel("Epoch")
            plt.ylabel("Count")
            plt.legend()
            plt.tight_layout()
            plt.savefig(save_dir / "stable_dims_curve.png", dpi=200)
            plt.close()
            print(f"Saved: {save_dir / 'stable_dims_curve.png'}")


def plot_metric(log_dir="./logs/experiment/version_0/", metric_name="loss", save_dir="./"):
    df = _read_metrics(log_dir)
    if df is None:
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    _plot_pair(
        df=df,
        train_col=f"train_{metric_name}",
        val_col=f"val_{metric_name}",
        save_path=save_dir / f"{metric_name}_curve.png",
        title=f"{metric_name.upper()} Curve",
        ylabel=metric_name,
    )