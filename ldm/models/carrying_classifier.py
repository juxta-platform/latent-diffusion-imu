"""MLP classifier on frozen VAE latent features for carrying-type prediction."""

import torch
import torch.nn as nn


class LatentMLPClassifier(nn.Module):
    """Simple MLP that classifies flattened VAE encoder outputs."""

    def __init__(self, in_dim=800, hidden=(256, 128), n_classes=4, dropout=0.2):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CarryingTypeClassifier(nn.Module):
    """Frozen VAE encoder + MLP head for carrying-type classification.

    Usage:
        clf = CarryingTypeClassifier(vae_model, mlp)
        logits = clf(imu_batch)          # imu_batch: [B, 6, 2000] standardized
    """

    def __init__(self, vae, mlp):
        super().__init__()
        self.vae = vae
        self.mlp = mlp
        # Freeze VAE
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, x):
        """Encode IMU and return flattened posterior mean."""
        posterior = self.vae.encode(x)
        z = posterior.mode()  # [B, 8, 100]
        return z.flatten(1)   # [B, 800]

    def forward(self, x):
        z_flat = self.encode(x)
        return self.mlp(z_flat)
