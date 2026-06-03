import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import torch


class LightningModel(pl.LightningModule):

    def __init__(self, model, loss_handler, metric_handler, optimizer_handler, dci_every_n_epochs=5):
        super().__init__()
        self.model = model
        self.loss_handler = loss_handler
        self.metric_handler = metric_handler
        self.optimizer_handler = optimizer_handler
        self.dci_every_n_epochs = dci_every_n_epochs


        self.train_z = []
        self.train_features = []
        self.val_z = []
        self.val_features = []

    def training_step(self, batch, batch_idx):
        volume, features, image_id = batch
        model_output = self.model(volume)
        total_loss, loss_dict = self.loss_handler(model_output, volume, features)

        self.log("train_loss", loss_dict["total_loss"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_recon", loss_dict["recon_loss"], on_step=False, on_epoch=True)
        self.log("train_kl", loss_dict["kl_loss"], on_step=False, on_epoch=True)
        self.log("train_tc", loss_dict["tc"], on_step=False, on_epoch=True)
        self.log("train_pred", loss_dict["pred_loss"], on_step=False, on_epoch=True)
        self.log("train_cluster", loss_dict["cluster_loss"], on_step=False, on_epoch=True)

        self.train_z.append(model_output["z"].detach().cpu())
        self.train_features.append(features.detach().cpu())

        return total_loss

    def validation_step(self, batch, batch_idx):
        volume, features, image_id = batch
        model_output = self.model(volume)
        total_loss, loss_dict = self.loss_handler(model_output, volume, features)

        self.log("val_loss", loss_dict["total_loss"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_recon", loss_dict["recon_loss"], on_step=False, on_epoch=True)
        self.log("val_kl", loss_dict["kl_loss"], on_step=False, on_epoch=True)
        self.log("val_tc", loss_dict["tc"], on_step=False, on_epoch=True)
        self.log("val_pred", loss_dict["pred_loss"], on_step=False, on_epoch=True)
        self.log("val_cluster", loss_dict["cluster_loss"], on_step=False, on_epoch=True)

        # Collect for DCI
        self.val_z.append(model_output["z"].detach().cpu())
        self.val_features.append(features.detach().cpu())

        return total_loss
    
    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % self.dci_every_n_epochs == 0 and len(self.train_z) > 0 and len(self.val_z) > 0:

            train_z = torch.cat(self.train_z, dim=0)
            train_features = torch.cat(self.train_features, dim=0)
            train_dci = self.metric_handler.compute_dci(train_z, train_features)

            val_z = torch.cat(self.val_z, dim=0)
            val_features = torch.cat(self.val_features, dim=0)
            val_dci = self.metric_handler.compute_dci(val_z, val_features)

            stable_dims = self.metric_handler.get_stable_dims(
                train_dci["importance_matrix"],
                val_dci["importance_matrix"],
            )

            self.log("train_D", train_dci["disentanglement"])
            self.log("train_I", train_dci["informativeness"])
            self.log("train_C", train_dci["completeness"])
            self.log("val_D", val_dci["disentanglement"], prog_bar=True)
            self.log("val_I", val_dci["informativeness"], prog_bar=True)
            self.log("val_C", val_dci["completeness"])
            self.log("stable_dims", float(len(stable_dims)), prog_bar=True)

            print(f"\n[Epoch {self.current_epoch}] "
                f"Train DCI — D: {train_dci['disentanglement']:.4f}, I: {train_dci['informativeness']:.4f} | "
                f"Val DCI — D: {val_dci['disentanglement']:.4f}, I: {val_dci['informativeness']:.4f} | "
                f"Stable dims: {len(stable_dims)}")

        self.train_z = []
        self.train_features = []
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

    def train(self, model, train_loader, val_loader, loss_handler, metric_handler, optimizer_handler, dci_every_n_epochs=5):
        lightning_model = LightningModel(
            model=model,
            loss_handler=loss_handler,
            metric_handler=metric_handler,
            optimizer_handler=optimizer_handler,
            dci_every_n_epochs=dci_every_n_epochs,
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_dir,
            filename="{epoch:02d}-{val_loss:.3f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
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
        )

        trainer.fit(lightning_model, train_loader, val_loader)

        return lightning_model

    def test(self, lightning_model, test_loader):
        trainer = pl.Trainer(accelerator="auto", devices=1)
        return trainer.test(lightning_model, test_loader)