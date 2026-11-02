import os
import time

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from shared.models.CIMLR import CIMLR, CIMLR_Feature_Ranking

# cimlr_mode:
#   full   CIMLR on the latents every epoch: updates the cluster targets and re-picks the top-k features
#   off    no CIMLR in training: no cluster loss, prediction on all features
#          (a top-k is still computed from the latents and logged, for post-analysis only)
#   fixed  cluster targets updated every epoch as in full; top-k picked once from the training
#          features (after warmup) and kept fixed
CIMLR_MODES = ("full", "off", "fixed")
TOPK_WARMUP_EPOCHS = 5


class LightningModel(pl.LightningModule):

    def __init__(
        self, model, loss_handler, metric_handler, optimizer_handler,
        dci_every_n_epochs=5, exp_logger=None, feature_names=None, top_k=20,
        estimate_c_every=5, estimate_c_warmup=5, estimate_c_until=40, cimlr_mode="full",
        cluster_age_adjust=False, cluster_covariates=None,
    ):
        super().__init__()
        if cimlr_mode not in CIMLR_MODES:
            raise ValueError(f"cimlr_mode must be one of {CIMLR_MODES}, got {cimlr_mode!r}")
        if cluster_age_adjust and not cluster_covariates:
            raise ValueError("cluster_age_adjust needs cluster_covariates {visit_id: (age, is_cn)}")
        self.cimlr_mode = cimlr_mode
        self.fixed_topk = None
        # age adjustment of what CIMLR sees (the latent itself is never changed)
        self.cluster_age_adjust = cluster_age_adjust
        self.cluster_covariates = cluster_covariates or {}
        self.model = model
        self.loss_handler = loss_handler
        self.metric_handler = metric_handler
        self.optimizer_handler = optimizer_handler
        self.dci_every_n_epochs = dci_every_n_epochs
        self.exp_logger = exp_logger
        self.feature_names = feature_names
        self.top_k = top_k
        self.topk_history = []        
        self.train_z = []
        self.train_features = []
        self.train_image_ids = []
        self.val_z = []
        self.val_features = []
        self.estimate_c_every = estimate_c_every
        self.estimate_c_warmup = estimate_c_warmup
        self.estimate_c_until = estimate_c_until

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
        self.log("train_dim_kl", loss_dict["dim_kl"], prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
        self.log("train_tc", loss_dict["tc"], prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
        self.log("train_pred", loss_dict["pred_loss"], prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
        self.log("train_pred_all", loss_dict["pred_loss_all"], on_step=False, on_epoch=True, batch_size=bs)
        self.log("train_cluster", loss_dict["cluster_loss"], prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
        self.log("train_cluster_pw_std", loss_dict["cluster_pw_std"], on_step=False, on_epoch=True, batch_size=bs)
        self.train_z.append(model_output["mu"].detach().cpu())
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
        self.log("val_dim_kl", loss_dict["dim_kl"], prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
        self.log("val_tc", loss_dict["tc"], prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
        self.log("val_pred", loss_dict["pred_loss"], prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
        self.log("val_pred_all", loss_dict["pred_loss_all"], prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)

        self.val_z.append(model_output["mu"].detach().cpu())
        self.val_features.append(features.detach().cpu())

        return total_loss


    def on_train_epoch_start(self):
        self._epoch_start = time.time()

    def on_train_epoch_end(self):
        # epoch time (training + validation + CIMLR) and a rough estimate of the time left
        took = time.time() - getattr(self, "_epoch_start", time.time())
        self._epoch_times = getattr(self, "_epoch_times", []) + [took]
        recent = self._epoch_times[-3:]
        left = sum(recent) / len(recent) * (self.trainer.max_epochs - self.current_epoch - 1)
        timing = f"[epoch {self.current_epoch}] took {took / 60:.1f} min | ~{left / 3600:.1f} h left"
        print(timing)

        if self.exp_logger is None:
            return
        self.exp_logger.log_message(timing)
        metrics = self.trainer.callback_metrics
        parts = []
        for key in sorted(metrics.keys()):
            try:
                parts.append(f"{key}={float(metrics[key]):.4f}")
            except (TypeError, ValueError):
                continue
        if parts:
            self.exp_logger.log_message(f"[epoch {self.current_epoch}] " + " | ".join(parts))

    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % self.dci_every_n_epochs == 0 and len(self.train_z) > 0 and len(self.val_z) > 0:
            print("Evalute DCI", self.current_epoch)
            train_z = torch.cat(self.train_z, dim=0)
            train_features = torch.cat(self.train_features, dim=0)
            val_z = torch.cat(self.val_z, dim=0)
            val_features = torch.cat(self.val_features, dim=0)

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

            cur = {int(d) for d in stable_dims}
            prev = getattr(self, "prev_stable_dims", None)
            overlap = len(cur & prev) if prev is not None else 0
            denom = len(prev) if prev else len(cur)
            self.prev_stable_dims = cur

            if self.exp_logger:
                self.exp_logger.log_message(
                    f"[epoch {self.current_epoch}] "
                    f"train D={train_dci['disentanglement']:.4f} I={train_dci['informativeness']:.4f} | "
                    f"val D={val_dci['disentanglement']:.4f} I={val_dci['informativeness']:.4f} | "
                    f"stable_dims={len(stable_dims)} overlap={overlap}/{denom} | {sorted(cur)}"
                )

        loss_fn = self.loss_handler.loss_fn
        topk = None
        if len(self.train_z) > 0 and hasattr(loss_fn, "update_cluster_cache"):
            z_all = torch.cat(self.train_z, dim=0).cpu().numpy().astype('float64')
            if self.cluster_age_adjust:
                z_all = self._age_adjusted(z_all, self.train_image_ids)
            epoch_num = self.current_epoch + 1

            if self.cimlr_mode != "off":
                estimate_c = (
                    epoch_num >= self.estimate_c_warmup
                    and epoch_num <= self.estimate_c_until
                    and self.estimate_c_every > 0
                    and epoch_num % self.estimate_c_every == 0
                )
                loss_fn.update_cluster_cache(self.train_image_ids, z_all, estimate_c=estimate_c)

            if epoch_num > TOPK_WARMUP_EPOCHS and len(self.train_features) > 0:
                X = torch.cat(self.train_features, dim=0).cpu().numpy().astype('float64')
                topk = self._select_topk(loss_fn, z_all, X)

        if topk is not None:
            prev = getattr(self, "prev_topk", None)
            overlap = len(set(topk) & set(prev)) if prev else 0
            self.prev_topk = topk
            if self.cimlr_mode != "off":
                loss_fn.active_feature_idx = topk

            names = [self.feature_names[i] for i in topk] if self.feature_names is not None else []
            self.topk_history.append({
                "epoch": self.current_epoch,
                "cimlr_mode": self.cimlr_mode,
                "overlap": overlap,
                "top_k": self.top_k,
                "n_clusters": self.loss_handler.loss_fn.n_clusters,
                "feature_idx": "|".join(str(i) for i in topk),
                "feature_names": "|".join(names),
            })

            if self.exp_logger:
                self.exp_logger.log_message(
                    f"[epoch {self.current_epoch}] top{self.top_k} "
                    f"overlap={overlap}/{self.top_k} | {sorted(topk)}"
                )
                if self.feature_names is not None:
                    names = [self.feature_names[i] for i in sorted(topk)]
                    for j in range(0, len(names), 5):
                        self.exp_logger.log_message("  " + ", ".join(names[j:j+5]))

        self.train_z = []
        self.train_features = []
        self.train_image_ids = []
        self.val_z = []
        self.val_features = []

    def _age_adjusted(self, z, visit_ids):
        """Copy of the latents with the normal-aging trend removed, for CIMLR only.
        Per latent dimension: z = a + b * age, fitted on this epoch's CN training visits;
        every visit with a known age gets z - (a + b * age)."""
        cov = [self.cluster_covariates.get(str(v), (np.nan, False)) for v in visit_ids]
        age = np.array([c[0] for c in cov], dtype=float)
        is_cn = np.array([c[1] for c in cov], dtype=bool)
        known = np.isfinite(age)
        fit = is_cn & known
        if fit.sum() < 10:
            return z
        design = np.column_stack([np.ones(fit.sum()), age[fit]])
        coef, *_ = np.linalg.lstsq(design, z[fit], rcond=None)
        adjusted = z.copy()
        adjusted[known] = z[known] - np.column_stack([np.ones(known.sum()), age[known]]) @ coef

        if self.exp_logger:
            def mean_abs_r(m):
                a = age[known] - age[known].mean()
                x = m[known] - m[known].mean(axis=0)
                denom = np.sqrt((a ** 2).sum() * (x ** 2).sum(axis=0)) + 1e-12
                return float(np.abs(a @ x / denom).mean())
            self.exp_logger.log_message(
                f"[epoch {self.current_epoch}] age-adjusted latents for CIMLR: "
                f"mean |r(dim, age)| {mean_abs_r(z):.3f} -> {mean_abs_r(adjusted):.3f}")
        return adjusted

    def _rank_features(self, data, X, n_clusters):
        """Top-k feature indices by CIMLR feature ranking; the similarity is learned on `data`."""
        k = min(10, data.shape[0] - 2)
        S, _, _, _ = CIMLR([data], n_clusters, k=k)
        top_idx, _ = CIMLR_Feature_Ranking(np.real(S), X)
        return top_idx[:self.top_k].tolist()

    def _select_topk(self, loss_fn, z_all, X):
        """The top-k for this epoch according to cimlr_mode (None if not available)."""
        try:
            if self.cimlr_mode == "full":
                S = getattr(loss_fn, "last_S", None)
                if S is None:
                    return None
                top_idx, _ = CIMLR_Feature_Ranking(S, X)
                return top_idx[:self.top_k].tolist()
            if self.cimlr_mode == "off":
                # logged for post-analysis only, never used in the loss
                return self._rank_features(z_all, X, loss_fn.n_clusters)
            if self.fixed_topk is None:
                self.fixed_topk = self._rank_features(X, X, loss_fn.n_clusters)
                if self.exp_logger:
                    self.exp_logger.log_message(
                        f"[fixed] top{self.top_k} picked once from the training features")
            return self.fixed_topk
        except Exception as e:
            if self.exp_logger:
                self.exp_logger.log_message(f"[topk] CIMLR ranking failed, kept previous: {e}")
            return getattr(self, "prev_topk", None)

    def configure_optimizers(self):
        optimizer = self.optimizer_handler.get_optimizer(self.parameters())
        scheduler = self.optimizer_handler.get_scheduler(optimizer)

        if scheduler is None:
            return optimizer
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_train_end(self):
        if self.topk_history and self.exp_logger:
            import pandas as pd
            pd.DataFrame(self.topk_history).to_csv(
                self.exp_logger.experiment_dir / "topk_history.csv", index=False
            )

class Trainer:

    def __init__(self, max_epochs=100, checkpoint_dir="./checkpoints/", experiment_dir=None):
        self.max_epochs = max_epochs
        self.checkpoint_dir = checkpoint_dir
        self.experiment_dir = experiment_dir

    def train(
        self, model, train_loader, val_loader, loss_handler, metric_handler,
        optimizer_handler, dci_every_n_epochs=5, exp_logger=None,
        feature_names=None, top_k=20,
        estimate_c_every=5, estimate_c_warmup=5, estimate_c_until=40,
        cimlr_mode="full", checkpoint_monitor="val_pred",
        cluster_age_adjust=False, cluster_covariates=None,
    ):
        lightning_model = LightningModel(
            model=model,
            loss_handler=loss_handler,
            metric_handler=metric_handler,
            optimizer_handler=optimizer_handler,
            dci_every_n_epochs=dci_every_n_epochs,
            exp_logger=exp_logger,
            feature_names=feature_names,
            top_k=top_k,
            estimate_c_every=estimate_c_every,
            estimate_c_warmup=estimate_c_warmup,
            estimate_c_until=estimate_c_until,
            cimlr_mode=cimlr_mode,
            cluster_age_adjust=cluster_age_adjust,
            cluster_covariates=cluster_covariates,
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_dir,
            filename="best",
            monitor=checkpoint_monitor,
            mode="min",
            save_top_k=1,
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
            log_every_n_steps=5,
        )

        trainer.fit(lightning_model, train_loader, val_loader)

        return lightning_model

    def test(self, lightning_model, test_loader):
        trainer = pl.Trainer(accelerator="auto", devices=1)
        return trainer.test(lightning_model, test_loader)