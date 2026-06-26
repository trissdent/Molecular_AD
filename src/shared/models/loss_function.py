import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
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

    def __call__(self, pred, target, extra=None, compute_cluster=True):
        if self.loss_type == "beta_tc_vae":
            return self.loss_fn(pred, target, extra, compute_cluster=compute_cluster)
        return self.loss_fn(pred, target)  # ty:ignore[missing-argument]


class BetaTCVAELoss:

    def __init__(self, recon_weight=1.0, kl_weight=1e-6,
                prediction_weight=1.0, cluster_weight=0.5,
                n_clusters=2, dataset_size=None, exp_logger=None):
        self.dataset_size = dataset_size
        self.training = True

        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.prediction_weight = prediction_weight
        self.cluster_weight = cluster_weight
        self.n_clusters = n_clusters
        print("num cluster", self.n_clusters)
        self.exp_logger = exp_logger
        self.cluster_probs_cache: dict = {}


    def _log(self, msg):
        if self.exp_logger is not None:
            self.exp_logger.log_message(msg)
        else:
            print(msg)

    def _recon_loss(self, recon, x):
        return F.l1_loss(recon, x, reduction='mean')

    def _standard_kl_loss(self, mu, logvar):
        # Positive KL(q(z|x) || N(0, I)), averaged per subject
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl.mean()

    def _prediction_loss(self, feature_pred, feature_target):
        idx = getattr(self, "active_feature_idx", None)
        if idx is not None:
            feature_pred = feature_pred[:, idx]
            feature_target = feature_target[:, idx]
        return F.mse_loss(feature_pred, feature_target, reduction='mean')


    def _cluster_loss(self, cluster_pairwise, image_ids):
        """
        Look up per-sample cluster probs from the cache built at the end of
        the previous epoch, then compute a pairwise BCE loss.
        """
        if not self.cluster_probs_cache:
            return torch.tensor(0.0, device=cluster_pairwise.device)

        probs_list = []
        for iid in image_ids:
            key = iid if isinstance(iid, str) else int(iid)
            if key not in self.cluster_probs_cache:
                return torch.tensor(0.0, device=cluster_pairwise.device)
            probs_list.append(self.cluster_probs_cache[key])

        probs = torch.stack(probs_list).to(cluster_pairwise.device)
        target = torch.matmul(probs, probs.t())
        return F.binary_cross_entropy_with_logits(cluster_pairwise, target)

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
        print("shape 0 iof z np", z_np.shape[0])
        print("shape 1 iof z np", z_np.shape[1])
        if N < max(self.n_clusters + 2, 4):
            print(f"[ClusterCache] skipped — only {N} samples")
            return

        try:
            t0 = time.time()
            k = min(10, N - 2)
            print("num cluster", self.n_clusters)
            if estimate_c:
                candidates = np.array([2, 3, 4])
                K1, K2 = Estimate_Number_of_Clusters_CIMLR([z_np], candidates)
                best_c = int(candidates[np.argmin(K1)])
                if best_c != self.n_clusters:
                    self._log(f"[ClusterCount] c: {self.n_clusters} → {best_c}")
                    self.n_clusters = best_c

            print(f"[ClusterCache] running CIMLR (N={N}, c={self.n_clusters}, k={k})...")
            S, LF, _, _ = CIMLR([z_np], self.n_clusters, k=k)
            LF = np.real(LF)
            self.last_S = np.real(S)

            gmm = GaussianMixture(
                n_components=self.n_clusters,
                covariance_type='diag',
                reg_covar=1e-4,
                random_state=42,
            )
            gmm.fit(LF)
            probs = gmm.predict_proba(LF)  # (N, n_clusters)

            self.cluster_probs_cache = {}
            for i, iid in enumerate(image_ids):
                key = iid if isinstance(iid, str) else int(iid)
                self.cluster_probs_cache[key] = torch.from_numpy(
                    probs[i].astype(np.float32)
                )
            print(f"[ClusterCache] done — total {time.time()-t0:.1f}s, {N} samples, "
                  f"cluster sizes: {np.bincount(probs.argmax(1)).tolist()}")
            self._log(f"Updated {N} samples → "
                    f"cluster sizes: {np.bincount(probs.argmax(1)).tolist()}")

        except Exception as e:
            self._log(f"[ClusterCache] CIMLR/GMM failed, keeping old cache: {e}")


    def __call__(self, model_output, x, feature_target, compute_cluster=True):
        recon = model_output["recon"]
        mu = model_output["mu"]
        logvar = model_output["logvar"]
        feature_pred = model_output["feature_pred"]
        cluster_pairwise = model_output["cluster_pairwise"]
        image_ids = model_output.get("image_ids", [])

        recon_loss = self._recon_loss(recon, x)

        dim_kl = self._standard_kl_loss(mu, logvar)
        kl_loss = self.kl_weight * dim_kl

        pred_loss = self._prediction_loss(feature_pred, feature_target)
        if compute_cluster:
            cluster_loss = self._cluster_loss(cluster_pairwise, image_ids)
        else:
            cluster_loss = torch.tensor(0.0, device=recon.device)

        total_loss = (self.recon_weight * recon_loss
                      + kl_loss
                      + self.prediction_weight * pred_loss
                      + self.cluster_weight * cluster_loss)

        loss_dict = {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "dim_kl": dim_kl,
            "kl_loss": kl_loss,
            "pred_loss": pred_loss,
            "cluster_loss": cluster_loss,
        }

        return total_loss, loss_dict