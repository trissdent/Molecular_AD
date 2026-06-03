import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.mixture import GaussianMixture


class LossHandler:

    def __init__(self, loss_type="cross_entropy", class_weights=None, **kwargs):
        self.loss_type = loss_type
        
        if loss_type == "cross_entropy":
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        elif loss_type == "bce":
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "beta_tc_vae":
            self.loss_fn = BetaTCVAELoss(**kwargs)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def __call__(self, pred, target, extra=None):
        if self.loss_type == "beta_tc_vae":
            return self.loss_fn(pred, target, extra)
        return self.loss_fn(pred, target)


class BetaTCVAELoss:

    def __init__(self, alpha=1.0, beta=6.0, gamma=1.0,
                 recon_weight=1.0, prediction_weight=1.0, cluster_weight=0.5,
                 n_clusters=2):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.recon_weight = recon_weight
        self.prediction_weight = prediction_weight
        self.cluster_weight = cluster_weight
        self.n_clusters = n_clusters

    def _recon_loss(self, recon, x):
        return F.l1_loss(recon, x, reduction='mean')

    def _tc_decomposition(self, z, mu, logvar):
        batch_size = z.size(0)

        log_qz_given_x = self._log_gaussian(z, mu, logvar).sum(dim=1)
        log_qz = self._log_qz_minibatch(z, mu, logvar, batch_size)
        log_qz_product = self._log_qz_product_minibatch(z, mu, logvar, batch_size)
        log_pz = self._log_gaussian(z, torch.zeros_like(z), torch.zeros_like(z)).sum(dim=1)

        mi = (log_qz_given_x - log_qz).mean()
        tc = (log_qz - log_qz_product).mean()
        dim_kl = (log_qz_product - log_pz).mean()

        return mi, tc, dim_kl

    def _log_gaussian(self, z, mu, logvar):
        return -0.5 * (np.log(2 * np.pi) + logvar + (z - mu).pow(2) / logvar.exp())

    def _log_qz_minibatch(self, z, mu, logvar, batch_size):
        z_expand = z.unsqueeze(1)
        mu_expand = mu.unsqueeze(0)
        logvar_expand = logvar.unsqueeze(0)
        log_qz_ij = self._log_gaussian(z_expand, mu_expand, logvar_expand)
        return torch.logsumexp(log_qz_ij.sum(dim=2), dim=1) - np.log(batch_size)

    def _log_qz_product_minibatch(self, z, mu, logvar, batch_size):
        z_expand = z.unsqueeze(1)
        mu_expand = mu.unsqueeze(0)
        logvar_expand = logvar.unsqueeze(0)
        log_qz_ij = self._log_gaussian(z_expand, mu_expand, logvar_expand)
        return (torch.logsumexp(log_qz_ij, dim=1) - np.log(batch_size)).sum(dim=1)

    def _prediction_loss(self, feature_pred, feature_target):
        return F.mse_loss(feature_pred, feature_target, reduction='mean')

    def _cluster_loss(self, cluster_pairwise, z):
        z_np = z.detach().cpu().numpy()
        gmm = GaussianMixture(n_components=self.n_clusters, covariance_type='full', random_state=42)
        gmm.fit(z_np)
        probs = gmm.predict_proba(z_np)
        probs = torch.from_numpy(probs).float().to(z.device)
        gmm_pairwise = torch.matmul(probs, probs.t())
        return F.binary_cross_entropy_with_logits(cluster_pairwise, gmm_pairwise)

    def __call__(self, model_output, x, feature_target):
        recon = model_output["recon"]
        mu = model_output["mu"]
        logvar = model_output["logvar"]
        z = model_output["z"]
        feature_pred = model_output["feature_pred"]
        cluster_pairwise = model_output["cluster_pairwise"]

        recon_loss = self._recon_loss(recon, x)
        mi, tc, dim_kl = self._tc_decomposition(z, mu, logvar)
        kl_loss = self.alpha * mi + self.beta * tc + self.gamma * dim_kl
        pred_loss = self._prediction_loss(feature_pred, feature_target)
        cluster_loss = self._cluster_loss(cluster_pairwise, z)

        total_loss = (self.recon_weight * recon_loss
                      + kl_loss
                      + self.prediction_weight * pred_loss
                      + self.cluster_weight * cluster_loss)

        loss_dict = {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "mi": mi,
            "tc": tc,
            "dim_kl": dim_kl,
            "kl_loss": kl_loss,
            "pred_loss": pred_loss,
            "cluster_loss": cluster_loss,
        }

        return total_loss, loss_dict