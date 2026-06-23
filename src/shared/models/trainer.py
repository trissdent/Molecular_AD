import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import torch
import numpy as np
from shared.models.CIMLR import CIMLR_Feature_Ranking


class LightningModel(pl.LightningModule):

    def __init__(self, model, loss_handler, metric_handler, optimizer_handler, 
                dci_every_n_epochs=5, exp_logger=None):
        super().__init__()
        self.model = model
        self.loss_handler = loss_handler
        self.metric_handler = metric_handler
        self.optimizer_handler = optimizer_handler
        self.dci_every_n_epochs = dci_every_n_epochs
        self.exp_logger = exp_logger

        self.train_z = []
        self.train_features = []
        self.train_image_ids = []
        self.val_z = []
        self.val_features = []

    def training_step(self, batch, batch_idx):
        volume, features, image_id = batch
        bs = volume.size(0)
        model_output = self.model(volume)
        model_output["image_ids"] = image_id
        self.loss_handler.loss_fn.training = True
        total_loss, loss_dict = self.loss_handler(model_output, volume, features)

        self.log("train_loss", loss_dict["total_loss"], prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
        self.log("train_recon", loss_dict["recon_loss"], prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
        self.log("train_kl", loss_dict["kl_loss"], prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
        self.log("train_tc", loss_dict["tc"], prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
        self.log("train_pred", loss_dict["pred_loss"], prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
        self.log("train_cluster", loss_dict["cluster_loss"], prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)

        self.train_z.append(model_output["z"].detach().cpu())
        self.train_features.append(features.detach().cpu())
        ids = image_id if isinstance(image_id, (list, tuple)) else list(image_id)
        self.train_image_ids.extend(ids)

        return total_loss

    def validation_step(self, batch, batch_idx):
        volume, features, image_id = batch
        bs = volume.size(0)
        model_output = self.model(volume)
        model_output["image_ids"] = image_id
        self.loss_handler.loss_fn.training = False
        total_loss, loss_dict = self.loss_handler(model_output, volume, features, compute_cluster=False)

        self.log("val_loss", loss_dict["total_loss"], prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
        self.log("val_recon", loss_dict["recon_loss"], prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
        self.log("val_kl", loss_dict["kl_loss"], prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
        self.log("val_tc", loss_dict["tc"], prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
        self.log("val_pred", loss_dict["pred_loss"], prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)

        self.val_z.append(model_output["z"].detach().cpu())
        self.val_features.append(features.detach().cpu())

        return total_loss


    def on_train_epoch_end(self):
        if self.exp_logger is None:
            return
        metrics = self.trainer.callback_metrics
        parts = []
        for key in sorted(metrics.keys()):
            try:
                parts.append(f"{key}={float(metrics[key]):.4f}")
            except (TypeError, ValueError):
                continue
        if parts:
            self.exp_logger.log_message("=== METRICS ===")
            self.exp_logger.log_message(f"[epoch {self.current_epoch}] " + " | ".join(parts))

    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % self.dci_every_n_epochs == 0 and len(self.train_z) > 0 and len(self.val_z) > 0:
            print("Evalute DCI", self.current_epoch)
            train_z = torch.cat(self.train_z, dim=0)
            train_features = torch.cat(self.train_features, dim=0)
            val_z = torch.cat(self.val_z, dim=0)
            val_features = torch.cat(self.val_features, dim=0)

            idx = getattr(self.loss_handler.loss_fn, "active_feature_idx", None)
            if idx is not None:
                train_features = train_features[:, idx]
                val_features = val_features[:, idx]

            train_dci = self.metric_handler.compute_dci(train_z, train_features)
            val_dci = self.metric_handler.compute_dci(val_z, val_features)

            stable_dims = self.metric_handler.get_stable_dims(
                train_dci["importance_matrix"],
                val_dci["importance_matrix"],
            )

            self.log("train_D", train_dci["disentanglement"], prog_bar=True)
            self.log("train_I", train_dci["informativeness"], prog_bar=True)
            self.log("train_C", train_dci["completeness"], prog_bar=True)
            self.log("val_D", val_dci["disentanglement"], prog_bar=True)
            self.log("val_I", val_dci["informativeness"], prog_bar=True)
            self.log("val_C", val_dci["completeness"], prog_bar=True)
            self.log("stable_dims", float(len(stable_dims)), prog_bar=True)

            cur = set(int(d) for d in stable_dims)
            prev = getattr(self, "prev_stable_dims", None)
            overlap = len(cur & prev) if prev is not None else 0
            denom = len(prev) if prev else len(cur)
            self.prev_stable_dims = cur

            if self.exp_logger:
                self.exp_logger.log_message("=== DCI ===")
                self.exp_logger.log_message(
                    f"[epoch {self.current_epoch}] "
                    f"train D={train_dci['disentanglement']:.4f} I={train_dci['informativeness']:.4f} | "
                    f"val D={val_dci['disentanglement']:.4f} I={val_dci['informativeness']:.4f} | "
                    f"stable_dims={len(stable_dims)} overlap={overlap}/{denom} | {sorted(cur)}"
                )

        if len(self.train_z) > 0 and hasattr(self.loss_handler, 'loss_fn') and \
                hasattr(self.loss_handler.loss_fn, 'update_cluster_cache'):
            z_all = torch.cat(self.train_z, dim=0).cpu().numpy().astype('float64')
            estimate_c = (self.current_epoch + 1) % 5 == 0
            self.loss_handler.loss_fn.update_cluster_cache(
                self.train_image_ids, z_all, estimate_c=estimate_c
            )

        WARMUP = 5
        S = getattr(self.loss_handler.loss_fn, "last_S", None)
        if (self.current_epoch + 1) > WARMUP and S is not None and len(self.train_features) > 0:
            X = torch.cat(self.train_features, dim=0).cpu().numpy()
            top_idx, _ = CIMLR_Feature_Ranking(S, X)
            top20 = top_idx[:20].tolist()
            prev = getattr(self, "prev_top20", None)
            overlap = len(set(top20) & set(prev)) if prev else 0
            self.prev_top20 = top20
            self.loss_handler.loss_fn.active_feature_idx = top20
            if self.exp_logger:
                self.exp_logger.log_message("=== TOP20 ===")
                self.exp_logger.log_message(
                    f"[epoch {self.current_epoch}] top20 overlap={overlap}/20 | {sorted(top20)}"
                )

        self.train_z = []
        self.train_features = []
        self.train_image_ids = []
        self.val_z = []
        self.val_features = []

    def configure_optimizers(self):
        optimizer = self.optimizer_handler.get_optimizer(self.parameters())
        scheduler = self.optimizer_handler.get_scheduler(optimizer)

        if scheduler is None:
            return optimizer
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class Trainer:

    def __init__(self, max_epochs=100, checkpoint_dir="./checkpoints/", experiment_dir=None):
        self.max_epochs = max_epochs
        self.checkpoint_dir = checkpoint_dir
        self.experiment_dir = experiment_dir

    def train(self, model, train_loader, val_loader, loss_handler, metric_handler, optimizer_handler, 
                dci_every_n_epochs=5, exp_logger=None):
        lightning_model = LightningModel(
            model=model,
            loss_handler=loss_handler,
            metric_handler=metric_handler,
            optimizer_handler=optimizer_handler,
            dci_every_n_epochs=dci_every_n_epochs,
            exp_logger=exp_logger,
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_dir,
            filename="best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=False,
        )

        if self.experiment_dir:
            logger = CSVLogger(save_dir=self.experiment_dir, name="", version="")
        else:
            logger = CSVLogger(save_dir="./logs/", name="experiment")

        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator="auto",
            devices=1,
            callbacks=[checkpoint_callback],
            logger=logger,
            log_every_n_steps=5,
        )

        trainer.fit(lightning_model, train_loader, val_loader)

        return lightning_model

    def test(self, lightning_model, test_loader):
        trainer = pl.Trainer(accelerator="auto", devices=1)
        return trainer.test(lightning_model, test_loader)