import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import ModelManager
from .. import utils


class Encoder3D(nn.Module):

    def __init__(self, z_dim=256, in_channels=1, input_size=96):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(256)
        self.conv5 = nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm3d(512)

        self.feat_dim = self._infer_flatten(in_channels, input_size)
        self.fc_mu = nn.Linear(self.feat_dim, z_dim)
        self.fc_logvar = nn.Linear(self.feat_dim, z_dim)

    def _conv_forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.2)
        return x

    def _infer_flatten(self, in_channels, input_size):
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_size, input_size, input_size)
            out = self._conv_forward(dummy)
            self.spatial = out.shape[2:]            # e.g. (3, 3, 3)
            return int(out.numel())

    def forward(self, x):
        x = self._conv_forward(x)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)


class Decoder3D(nn.Module):

    def __init__(self, z_dim=256, out_channels=1, spatial=(3, 3, 3)):
        super().__init__()
        self.spatial = spatial
        self.fc = nn.Linear(z_dim, 512 * spatial[0] * spatial[1] * spatial[2])

        self.deconv1 = nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(256)
        self.deconv2 = nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.deconv3 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.deconv4 = nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(32)
        self.deconv5 = nn.ConvTranspose3d(32, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 512, *self.spatial)
        x = F.leaky_relu(self.bn1(self.deconv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.deconv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.deconv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.deconv4(x)), 0.2)
        x = torch.sigmoid(self.deconv5(x))
        return x


class PredictionHead(nn.Module):

    def __init__(self, z_dim=256, num_features=1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_features),
        )

    def forward(self, z):
        return self.fc(z)


class ClusterProjectionHead(nn.Module):

    def __init__(self, z_dim=256, projection_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, z):
        logits = self.fc(z)
        pairwise = torch.matmul(logits, logits.t())
        return pairwise


class BetaTCVAE(nn.Module, ModelManager):

    def __init__(self, z_dim=256, in_channels=1, num_features=1,
                 cluster_projection_dim=128, input_size=96):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = Encoder3D(z_dim=z_dim, in_channels=in_channels, input_size=input_size)
        self.decoder = Decoder3D(z_dim=z_dim, out_channels=in_channels,
                                 spatial=self.encoder.spatial)
        self.prediction_head = PredictionHead(z_dim=z_dim, num_features=num_features)
        self.cluster_projection_head = ClusterProjectionHead(z_dim=z_dim, projection_dim=cluster_projection_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        feature_pred = self.prediction_head(z)
        cluster_pairwise = self.cluster_projection_head(z)

        return {
            "recon": recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "feature_pred": feature_pred,
            "cluster_pairwise": cluster_pairwise,
        }