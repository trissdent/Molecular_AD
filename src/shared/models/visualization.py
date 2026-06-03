# visualization.py
# Plotting utilities.
# Modify for your task.

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_training_curves(log_dir="./logs/experiment/version_0/", save_dir="./"):
    """
    Plot training curves from CSVLogger.
    """
    # Read logs
    log_path = Path(log_dir) / "metrics.csv"
    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        return
    
    logs = pd.read_csv(log_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Loss curve
    if "train_loss" in logs.columns and "val_loss" in logs.columns:
        plt.figure()
        logs[["train_loss", "val_loss"]].dropna().plot()
        plt.title("Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(["Train", "Validation"])
        plt.savefig(save_dir / "loss_curve.png")
        plt.close()
        print(f"Saved: {save_dir / 'loss_curve.png'}")
    
    # Accuracy curve
    if "train_acc" in logs.columns and "val_acc" in logs.columns:
        plt.figure()
        logs[["train_acc", "val_acc"]].dropna().plot()
        plt.title("Accuracy Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(["Train", "Validation"])
        plt.savefig(save_dir / "acc_curve.png")
        plt.close()
        print(f"Saved: {save_dir / 'acc_curve.png'}")


def plot_metric(log_dir="./logs/experiment/version_0/", metric_name="loss", save_dir="./"):
    """
    Plot specific metric.
    metric_name: "loss", "acc", "f1", "dice", etc.
    """
    log_path = Path(log_dir) / "metrics.csv"
    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        return
    
    logs = pd.read_csv(log_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    train_col = f"train_{metric_name}"
    val_col = f"val_{metric_name}"
    
    if train_col in logs.columns and val_col in logs.columns:
        plt.figure()
        logs[[train_col, val_col]].dropna().plot()
        plt.title(f"{metric_name.upper()} Curve")
        plt.xlabel("Epoch")
        plt.ylabel(metric_name.capitalize())
        plt.legend(["Train", "Validation"])
        plt.savefig(save_dir / f"{metric_name}_curve.png")
        plt.close()
        print(f"Saved: {save_dir / f'{metric_name}_curve.png'}")
    else:
        print(f"Metric not found: {train_col}, {val_col}")