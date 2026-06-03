import torch
from torch.utils.data import DataLoader
from configs.config import ConfigReader
from shared.models.loss_function import LossHandler
from shared.models.metrics import MetricHandler
from shared.models.optimization import OptimizerHandler
from shared.models.trainer import Trainer
from shared.models.logger import ExperimentLogger
from shared.models.visualization import plot_training_curves
from shared.services.data.dataset import MRIDataset
from shared.services.data.transforms import MRITransformer
from shared.services.models_hub.beta_tc_vae.model import BetaTCVAE


def run(config_path="./configs/defaults.yaml", experiment_path=None):
    config = ConfigReader.merge(config_path, experiment_path)

    logger = ExperimentLogger(log_dir=config.training.log_dir)
    logger.log_config(config_path)
    logger.log_message("Training started")

    transform = MRITransformer(
        target_shape=tuple(config.transform.target_shape),
        margin=config.transform.margin,
    )

    train_dataset = MRIDataset(
        data_dir=config.data.data_dir,
        feature_csv_path=config.data.feature_csv_path,
        transform=transform,
        cache_dir=config.data.cache_dir,
    )

    n = len(train_dataset)
    n_train = int(0.8 * n)
    n_val = n - n_train
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [n_train, n_val], generator=generator,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Train: {len(train_subset)}, Val: {len(val_subset)}")
    print(f"Features: {len(train_dataset.feature_names)}")

    # Model
    model = BetaTCVAE(
        z_dim=config.model.z_dim,
        in_channels=config.model.in_channels,
        num_features=len(train_dataset.feature_names),
        cluster_projection_dim=config.model.cluster_projection_dim,
    )
    model.summary()
    logger.log_model_info(model)

    # Handlers
    loss_handler = LossHandler(
        loss_type=config.loss.type,
        alpha=config.loss.alpha,
        beta=config.loss.beta,
        gamma=config.loss.gamma,
        recon_weight=config.loss.recon_weight,
        prediction_weight=config.loss.prediction_weight,
        cluster_weight=config.loss.cluster_weight,
        n_clusters=config.loss.n_clusters,
    )

    metric_handler = MetricHandler(
        dci_alpha=config.metrics.dci_alpha,
    )

    optimizer_handler = OptimizerHandler(
        optimizer_type=config.training.optimizer,
        lr=config.training.lr,
    )

    # Train
    trainer = Trainer(
        max_epochs=config.training.max_epochs,
        checkpoint_dir=config.training.checkpoint_dir,
        experiment_dir=logger.get_experiment_dir(),
    )

    lightning_model = trainer.train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_handler=loss_handler,
        metric_handler=metric_handler,
        optimizer_handler=optimizer_handler,
        dci_every_n_epochs=config.training.dci_every_n_epochs,
    )

    model.save(config.training.checkpoint_dir + "model_weights.pt")
    logger.log_message("Training completed")

    plot_training_curves(
        log_dir=logger.get_experiment_dir(),
        save_dir=logger.get_experiment_dir(),
    )


if __name__ == "__main__":
    run(config_path="./configs/defaults.yaml")