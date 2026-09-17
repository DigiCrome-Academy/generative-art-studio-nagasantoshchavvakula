"""Variational Autoencoder (VAE) with the reparameterization trick.

Learning objectives (docs/PROJECT_BRIEF.md, Phase 1):
"Develop Variational Autoencoder (VAE) with reparameterization trick" and
"Generate new samples by sampling from learned latent distribution."

The VAE loss (reconstruction + KL divergence) lives in
`src/generative_art_studio/training/losses.py::vae_loss` — implement that
alongside `reparameterize` below.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .vanilla_ae import Decoder


class VAEEncoder(nn.Module):
    """Encodes an image into the parameters (mu, logvar) of a diagonal Gaussian."""

    def __init__(self, in_channels: int = 3, latent_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class VAE(nn.Module):
    """Variational Autoencoder: encode -> reparameterize -> decode."""

    def __init__(self, in_channels: int = 3, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = VAEEncoder(in_channels, latent_dim)
        self.decoder = Decoder(in_channels, latent_dim)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z ~ N(mu, sigma^2) in a differentiable way.

        TODO(Phase 1 - VAE reparameterization trick): implement
            std = exp(0.5 * logvar)
            eps ~ N(0, I)               (use torch.randn_like(std))
            z = mu + eps * std
        This is *the* trick that lets gradients flow through a stochastic
        sampling step: instead of sampling z directly from N(mu, sigma^2),
        we sample eps from a fixed N(0, I) and compute z as a deterministic,
        differentiable function of (mu, logvar, eps).
        """
        # raise NotImplementedError(
        #     "TODO: implement the reparameterization trick — "
        #     "std = exp(0.5 * logvar); eps = torch.randn_like(std); return mu + eps * std."
        # )
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device | str = "cpu") -> torch.Tensor:
        """Generate new images by sampling z ~ N(0, I) and decoding it."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)
