import torch, os, json
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
import pandas as pd
from sklearn.model_selection import train_test_split

def run(config_path="./configs/defaults.yaml", experiment_path=None):
    config = ConfigReader.merge(config_path, experiment_path)

    logger = ExperimentLogger(log_dir=config.training.log_dir)
    logger.log_config(config_path)
    logger.log_message("Training started")
    run_name = logger.get_run_name()
    run_ckpt_dir = config.training.checkpoint_dir + run_name + "/"
    os.makedirs(run_ckpt_dir, exist_ok=True)

    transform = MRITransformer(
        target_shape=tuple(config.transform.target_shape),
        margin=config.transform.margin,
    )

    demo = pd.read_csv("/home/minhtri/Molecular_AD/data/all_demographics.csv")
    demo["image_id"] = demo["image_id"].astype(str)
    demo = demo.dropna(subset=["diagnosis"])
    label_lookup = dict(zip(demo["image_id"], demo["diagnosis"].str.lower()))

    feat_df = pd.read_csv(config.data.feature_csv_path)
    feat_df["image_id"] = feat_df["image_id"].astype(str)
    all_ids = feat_df["image_id"].tolist()

    ids, labels = [], []
    for iid in all_ids:
        if iid in label_lookup:
            ids.append(iid)
            labels.append(label_lookup[iid])

    split_path = run_ckpt_dir + "split.json"
    if os.path.exists(split_path):
        with open(split_path) as f:
            split = json.load(f)
        train_ids, val_ids, test_ids = split["train"], split["val"], split["test"]
        print(f"Loaded existing split: {split_path}")
    else:
        train_ids, temp_ids, train_lab, temp_lab = train_test_split(
            ids, labels, train_size=768, stratify=labels, random_state=42,
        )
        val_ids, test_ids = train_test_split(
            temp_ids, test_size=0.50, stratify=temp_lab, random_state=42,
        )
        split = {"train": train_ids, "val": val_ids, "test": test_ids}
        os.makedirs(config.training.checkpoint_dir, exist_ok=True)
        with open(split_path, "w") as f:
            json.dump(split, f, indent=2)

    print(f"Train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")

    def make_ds(id_list):
        return MRIDataset(
            data_dir=config.data.data_dir,
            feature_csv_path=config.data.feature_csv_path,
            transform=transform,
            cache_dir=config.data.cache_dir,
            image_ids=id_list,
            normalize=False,
        )

    train_dataset = make_ds(train_ids)
    val_dataset   = make_ds(val_ids)
    test_dataset  = make_ds(test_ids)

    train_raw = train_dataset.raw_features_df.loc[
        [s["image_id"] for s in train_dataset.samples]
    ]
    mean = train_raw.mean()
    std = train_raw.std().replace(0, 1.0)
    for ds in (train_dataset, val_dataset, test_dataset):
        ds.set_normalization(mean, std)

    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    mean.to_csv(run_ckpt_dir + "feature_mean.csv")
    std.to_csv(run_ckpt_dir + "feature_std.csv")

    train_subset = train_dataset
    val_subset = val_dataset

    train_loader = DataLoader(
        train_subset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=torch.cuda.is_available()
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

    model = BetaTCVAE(
        z_dim=config.model.z_dim,
        in_channels=config.model.in_channels,
        num_features=len(train_dataset.feature_names),
        cluster_projection_dim=config.model.cluster_projection_dim,
        input_size=config.transform.target_shape[0],
    )
    model.summary()
    logger.log_model_info(model)

    loss_handler = LossHandler(
        loss_type=config.loss.type,
        recon_weight=config.loss.recon_weight,
        kl_weight=config.loss.kl_weight,
        prediction_weight=config.loss.prediction_weight,
        cluster_weight=config.loss.cluster_weight,
        n_clusters=config.loss.n_clusters,
        dataset_size=len(train_subset),
        exp_logger=logger,
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
        
    )

    model.save(run_ckpt_dir + "model_weights.pt")
    logger.log_message("Training completed")

    plot_training_curves(
        log_dir=logger.get_experiment_dir(),
        save_dir=logger.get_experiment_dir(),
    )


if __name__ == "__main__":
    run(config_path="./configs/defaults.yaml")
