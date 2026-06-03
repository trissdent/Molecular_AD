# Testing pipeline.
# Modify for your task.

import torch
from torch.utils.data import DataLoader

from configs import ConfigReader
from shared.models import LossHandler, MetricHandler, OptimizerHandler, Trainer, LightningModel
from shared.services.data import BaseDataset, Transformer
from shared.services.models_hub import UNet


def run(config_path="./configs/defaults.yml", experiment_path=None):
    # Load config
    config = ConfigReader.merge(config_path, experiment_path)
    
    # Transform
    transform = Transformer(
        target_size=tuple(config.transform.target_size),
        do_augmentation=False
    )
    
    # Dataset
    test_dataset = BaseDataset(data_dir=config.data.test_dir, transform=transform)
    
    # DataLoader
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    # Model
    model = UNet(
        n_channels=config.model.n_channels,
        n_classes=config.model.n_classes
    )
    
    # Handlers
    loss_handler = LossHandler(loss_type=config.loss.type)
    metric_handler = MetricHandler(
        task=config.metrics.task,
        num_classes=config.metrics.num_classes
    )
    optimizer_handler = OptimizerHandler(
        optimizer_type=config.training.optimizer,
        lr=config.training.lr
    )
    
    # Load checkpoint
    checkpoint_path = config.training.checkpoint_dir + "last.ckpt"
    lightning_model = LightningModel.load_from_checkpoint(
        checkpoint_path,
        model=model,
        loss_handler=loss_handler,
        metric_handler=metric_handler,
        optimizer_handler=optimizer_handler
    )
    
    # Test
    trainer = Trainer()
    trainer.test(lightning_model, test_loader)


if __name__ == "__main__":
    run(config_path="./configs/defaults.yml")