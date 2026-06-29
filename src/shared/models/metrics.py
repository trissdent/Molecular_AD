import numpy as np
import torch
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from torchmetrics.functional import accuracy, precision, recall, f1_score, jaccard_index


class MetricHandler:

    def __init__(self, task="multiclass", num_classes=2, threshold=0.5, smooth=1e-5, dci_alpha=0.1):
        self.task = task
        self.num_classes = num_classes
        self.threshold = threshold
        self.smooth = smooth
        self.dci_alpha = dci_alpha

    # Classification
    def get_accuracy(self, pred, target):
        return accuracy(pred, target, task=self.task, num_classes=self.num_classes)

    def get_precision(self, pred, target):
        return precision(pred, target, task=self.task, num_classes=self.num_classes, average="macro")

    def get_recall(self, pred, target):
        return recall(pred, target, task=self.task, num_classes=self.num_classes, average="macro")

    def get_f1(self, pred, target):
        return f1_score(pred, target, task=self.task, num_classes=self.num_classes, average="macro")

    # Segmentation
    def get_iou(self, pred, target):
        return jaccard_index(pred, target, task="binary", threshold=self.threshold)

    def get_dice(self, pred, target):
        intersection = (pred * target).sum()
        return (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)


    # recon beta tc vae
    def compute_dci(self, z, features):
        if isinstance(z, torch.Tensor):
            z = z.detach().cpu().numpy()
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()

        z_scaler = StandardScaler()
        f_scaler = StandardScaler()
        z = z_scaler.fit_transform(z)
        features = f_scaler.fit_transform(features)

        z_dim = z.shape[1]
        num_features = features.shape[1]

        importance = np.zeros((z_dim, num_features))
        r2_scores = np.zeros(num_features)

        for j in range(num_features):
            lasso = Lasso(alpha=self.dci_alpha, max_iter=10000)
            lasso.fit(z, features[:, j])
            importance[:, j] = np.abs(lasso.coef_)
            r2_scores[j] = lasso.score(z, features[:, j])

        informativeness = np.mean(r2_scores)
        disentanglement = self._entropy_score_rows(importance)
        completeness = self._entropy_score_cols(importance)

        return {
            "disentanglement": disentanglement,
            "completeness": completeness,
            "informativeness": informativeness,
            "importance_matrix": importance,
            "r2_scores": r2_scores,
        }

    def get_stable_dims(self, train_importance, val_importance,
                        threshold=0.1, top_n=3, min_overlap=1):
        stable_dims = []
        for i in range(train_importance.shape[0]):
            train_row = train_importance[i]
            val_row = val_importance[i]

            if train_row.sum() < 1e-6 or val_row.sum() < 1e-6:
                continue

            train_top = np.argsort(train_row)[-top_n:]
            val_top = np.argsort(val_row)[-top_n:]

            shared = [j for j in set(train_top) & set(val_top)
                      if train_row[j] > threshold and val_row[j] > threshold]

            if len(shared) >= min_overlap:
                stable_dims.append(i)

        return stable_dims

    def _entropy_score_rows(self, matrix):
        scores = []
        for i in range(matrix.shape[0]):
            row = matrix[i]
            if row.sum() < 1e-6:
                continue
            p = row / row.sum()
            p = p[p > 0]
            entropy = -np.sum(p * np.log(p))
            max_entropy = np.log(matrix.shape[1])
            score = 1.0 - entropy / max_entropy if max_entropy > 0 else 0.0
            scores.append(score)

        if not scores:
            return 0.0

        row_sums = np.array([matrix[i].sum() for i in range(matrix.shape[0]) if matrix[i].sum() >= 1e-6])
        weights = row_sums / row_sums.sum()
        return float(np.sum(np.array(scores) * weights))

    def _entropy_score_cols(self, matrix):
        scores = []
        for j in range(matrix.shape[1]):
            col = matrix[:, j]
            if col.sum() < 1e-6:
                continue
            p = col / col.sum()
            p = p[p > 0]
            entropy = -np.sum(p * np.log(p))
            max_entropy = np.log(matrix.shape[0])
            score = 1.0 - entropy / max_entropy if max_entropy > 0 else 0.0
            scores.append(score)

        if not scores:
            return 0.0

        col_sums = np.array([matrix[:, j].sum() for j in range(matrix.shape[1]) if matrix[:, j].sum() >= 1e-6])
        weights = col_sums / col_sums.sum()
        return float(np.sum(np.array(scores) * weights))