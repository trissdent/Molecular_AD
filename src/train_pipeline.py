"""Train one experiment.

    python train_pipeline.py --exp configs/smri.yaml   # or dti.yaml / both.yaml

Reads cohort split, features and volume caches made by preprocessing/.
The merged config (defaults + override) is saved in the run's log folder, so
test_pipeline and post_analysis need only the run name afterwards.
"""

import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch
from configs.config import ConfigReader
from shared.models.logger import ExperimentLogger
from shared.models.loss_function import LossHandler
from shared.models.metrics import MetricHandler
from shared.models.optimization import OptimizerHandler
from shared.models.trainer import Trainer
from shared.models.visualization import plot_training_curves
from shared.services.data import VisitDataset, load_metadata, load_split, n_channels
from shared.services.models_hub.beta_tc_vae.model import BetaTCVAE
from torch.utils.data import DataLoader


def git_state():
    """Commit of the code this run used, and whether there were uncommitted changes."""
    here = Path(__file__).resolve().parent

    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=here, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = git("rev-parse", "HEAD")
    changes = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "uncommitted_changes": bool(changes) if changes is not None else None}


def run(config_path="./configs/defaults.yaml", experiment_path=None, run_name_file=None):
    config_dict = ConfigReader.merge_dict(config_path, experiment_path)
    split_ids, split_sha = load_split(config_dict["data"]["prepared_dir"])
    config_dict["data"]["split_sha"] = split_sha          # test_pipeline checks this
    config_dict["data"]["experiment_file"] = experiment_path
    config_dict["git"] = git_state()                       # which code version produced this run
    config = ConfigReader(config_dict)
    modality = config.data.modality
    pl.seed_everything(config.training.seed, workers=True)

    # run name = override file name + time, e.g. dti_seed1_2026-11-02_10-00-00
    tag = Path(experiment_path).stem if experiment_path else "defaults"
    run_name = f"{tag}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    logger = ExperimentLogger(log_dir=config.training.log_dir, experiment_name=run_name)
    if run_name_file:       # lets train_and_evaluate.sh know exactly which run this is
        Path(run_name_file).write_text(run_name)
    logger.log_config_dict(config_dict)
    run_name = logger.get_run_name()
    run_ckpt_dir = config.training.checkpoint_dir + run_name + "/"
    os.makedirs(run_ckpt_dir, exist_ok=True)

    counts = {name: len(ids) for name, ids in split_ids.items()}
    logger.log_message(f"Training started | modality={modality} | seed={config.training.seed} | "
                       f"split={counts} sha={split_sha}")
    print(f"Run {run_name} | modality {modality} | seed {config.training.seed} | "
          f"split {counts} (sha {split_sha})")

    def make_ds(ids):
        return VisitDataset(ids, modality, config.data.prepared_dir,
                            vars(config.data.cache_dirs), normalize=False,
                            feature_tables=getattr(getattr(config.data, "feature_tables", None), "__dict__", None))

    train_dataset = make_ds(split_ids["train"])
    val_dataset = make_ds(split_ids["val"])

    # feature normalization from the training visits only
    train_raw = train_dataset.raw_features_df.loc[split_ids["train"]]
    mean = train_raw.mean()
    std = train_raw.std().replace(0, 1.0)
    for ds in (train_dataset, val_dataset):
        ds.set_normalization(mean, std)
    mean.to_csv(run_ckpt_dir + "feature_mean.csv")
    std.to_csv(run_ckpt_dir + "feature_std.csv")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = BetaTCVAE(
        z_dim=config.model.z_dim,
        in_channels=n_channels(modality),
        num_features=len(train_dataset.feature_names),
        cluster_projection_dim=config.model.cluster_projection_dim,
        input_size=config.model.input_size,
    )
    model.summary()
    logger.log_model_info(model)

    loss_handler = LossHandler(
        loss_type=config.loss.type,
        recon_weight=config.loss.recon_weight,
        kl_weight=config.loss.kl_weight,
        prediction_weight=config.loss.prediction_weight,
        cluster_weight=config.loss.cluster_weight,
        tc_weight=config.loss.tc_weight,
        queue_size=config.loss.queue_size,
        queue_warmup=config.loss.queue_warmup,
        n_clusters=config.loss.n_clusters,
        recon_scale=config.loss.recon_scale,
        pred_scale=config.loss.pred_scale,
        cluster_scale=config.loss.cluster_scale,
        tc_scale=config.loss.tc_scale,
        dataset_size=len(train_dataset),
        exp_logger=logger,
    )

    metric_handler = MetricHandler(
        dci_alpha=config.metrics.dci_alpha,
    )

    optimizer_handler = OptimizerHandler(
        optimizer_type=config.training.optimizer,
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
        scheduler_type=config.training.scheduler,
        T_max=config.training.max_epochs,
    )

    # age + CN flag per training visit, used only when CIMLR clusters age-adjusted latents
    cluster_age_adjust = bool(getattr(config.training, "cluster_age_adjust", False))
    cluster_covariates = None
    if cluster_age_adjust:
        meta = load_metadata(config.data.prepared_dir).loc[split_ids["train"]]
        cluster_covariates = {v: (float(r.age), r.diagnosis == "cn") for v, r in meta.iterrows()}
        logger.log_message("CIMLR clusters age-adjusted latents (trend fitted on train CN)")

    trainer = Trainer(
        max_epochs=config.training.max_epochs,
        checkpoint_dir=run_ckpt_dir,
        experiment_dir=logger.get_experiment_dir(),
    )

    trainer.train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_handler=loss_handler,
        metric_handler=metric_handler,
        optimizer_handler=optimizer_handler,
        dci_every_n_epochs=config.training.dci_every_n_epochs,
        exp_logger=logger,
        feature_names=train_dataset.feature_names,
        top_k=config.training.top_k,
        estimate_c_every=config.training.estimate_c_every,
        estimate_c_warmup=config.training.estimate_c_warmup,
        estimate_c_until=config.training.estimate_c_until,
        cimlr_mode=config.training.cimlr_mode,
        checkpoint_monitor=config.training.checkpoint_monitor,
        cluster_age_adjust=cluster_age_adjust,
        cluster_covariates=cluster_covariates,
    )

    model.save(run_ckpt_dir + "model_weights.pt")
    logger.log_message("Training completed")

    plot_training_curves(
        log_dir=logger.get_experiment_dir(),
        save_dir=logger.get_experiment_dir(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/defaults.yaml")
    parser.add_argument("--exp", default=None, help="override file, e.g. configs/dti.yaml")
    parser.add_argument("--run-name-file", default=None, help="write the run name to this file")
    args = parser.parse_args()
    run(config_path=args.config, experiment_path=args.exp, run_name_file=args.run_name_file)