import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.mixture import GaussianMixture
from shared.models.CIMLR import CIMLR, Estimate_Number_of_Clusters_CIMLR

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
        return self.loss_fn(pred, target)  # ty:ignore[missing-argument]


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

        # Cache: image_id (str/int) → prob tensor (n_clusters,)
        # Populated once per epoch in on_train_epoch_end via update_cluster_cache().
        # Empty on epoch 0 → cluster_loss returns 0 silently (correct lag behaviour).
        self.cluster_probs_cache: dict = {}

    # ------------------------------------------------------------------
    # Reconstruction
    # ------------------------------------------------------------------
    def _recon_loss(self, recon, x):
        return F.l1_loss(recon, x, reduction='mean')

    # ------------------------------------------------------------------
    # Beta-TC decomposition
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Feature prediction
    # ------------------------------------------------------------------
    def _prediction_loss(self, feature_pred, feature_target):
        idx = getattr(self, "active_feature_idx", None)
        if idx is not None:
            feature_pred = feature_pred[:, idx]
            feature_target = feature_target[:, idx]
        return F.mse_loss(feature_pred, feature_target, reduction='mean')

    # ------------------------------------------------------------------
    # Cluster loss — cache lookup (no CIMLR per batch)
    # ------------------------------------------------------------------
    def _cluster_loss(self, cluster_pairwise, image_ids):
        """
        Look up per-sample cluster probs from the cache built at the end of
        the previous epoch, then compute a pairwise BCE loss.

        Returns 0 on epoch 0 (cache is empty) — this is intentional lag.
        """
        if not self.cluster_probs_cache:
            # Epoch 0: no cache yet, skip silently
            return torch.tensor(0.0, device=cluster_pairwise.device)

        probs_list = []
        for iid in image_ids:
            key = iid if isinstance(iid, str) else int(iid)
            if key not in self.cluster_probs_cache:
                # Sample missing from cache (shouldn't happen after epoch 0)
                return torch.tensor(0.0, device=cluster_pairwise.device)
            probs_list.append(self.cluster_probs_cache[key])

        # probs: (B, n_clusters) — soft cluster assignments
        probs = torch.stack(probs_list).to(cluster_pairwise.device)
        # pairwise target: (B, B) — how similar are two samples' cluster memberships
        target = torch.matmul(probs, probs.t())
        return F.binary_cross_entropy_with_logits(cluster_pairwise, target)

    # ------------------------------------------------------------------
    # Called once per epoch end from trainer.on_train_epoch_end()
    # ------------------------------------------------------------------
    def update_cluster_cache(self, image_ids: list, z_np: np.ndarray, estimate_c: bool = False):
        """
        Run CIMLR + GMM on the full-epoch accumulated Z and store soft cluster
        probabilities keyed by image_id. Also stores CIMLR's similarity S for
        reuse by the top-20 feature ranking.

        Args:
            image_ids: list of image IDs in the same order as z_np rows.
            z_np:      (N, z_dim) float64 numpy array of all train Z this epoch.
        """
        N = z_np.shape[0]
        if N < max(self.n_clusters + 2, 4):
            # Dataset too small to cluster — keep old cache
            return

        try:
            k = min(10, N - 2)
            if estimate_c:
                
                candidates = np.array([2, 3, 4, 5, 6])
                K1, K2 = Estimate_Number_of_Clusters_CIMLR([z_np], candidates)
                best_c = int(candidates[np.argmin(K1)])
                if best_c != self.n_clusters:
                    print(f"[ClusterCount] c: {self.n_clusters} → {best_c}")
                    self.n_clusters = best_c
            _, S, _, _, _, _, _, LF = CIMLR([z_np], self.n_clusters, k=k)
            LF = np.real(LF)
            self.last_S = np.real(S)   # save similarity for top-20 ranking reuse

            gmm = GaussianMixture(
                n_components=self.n_clusters,
                covariance_type='diag',
                reg_covar=1e-4,
                random_state=42,
            )
            gmm.fit(LF)
            probs = gmm.predict_proba(LF)  # (N, n_clusters)

            # Update cache
            self.cluster_probs_cache = {}
            for i, iid in enumerate(image_ids):
                key = iid if isinstance(iid, str) else int(iid)
                self.cluster_probs_cache[key] = torch.from_numpy(
                    probs[i].astype(np.float32)
                )
            print(f"[ClusterCache] Updated {N} samples → "
                  f"cluster sizes: {np.bincount(probs.argmax(1)).tolist()}")

        except Exception as e:
            print(f"[ClusterCache] CIMLR/GMM failed, keeping old cache: {e}")

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------
    def __call__(self, model_output, x, feature_target):
        recon = model_output["recon"]
        mu = model_output["mu"]
        logvar = model_output["logvar"]
        z = model_output["z"]
        feature_pred = model_output["feature_pred"]
        cluster_pairwise = model_output["cluster_pairwise"]
        image_ids = model_output.get("image_ids", [])

        recon_loss = self._recon_loss(recon, x)
        mi, tc, dim_kl = self._tc_decomposition(z, mu, logvar)
        kl_loss = self.alpha * mi + self.beta * tc + self.gamma * dim_kl
        pred_loss = self._prediction_loss(feature_pred, feature_target)
        cluster_loss = self._cluster_loss(cluster_pairwise, image_ids)

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