"""Denoising Autoencoder: trained to reconstruct a clean image from a
noise-corrupted input.

Learning objective (docs/PROJECT_BRIEF.md, Phase 1):
"Implement denoising autoencoder for image restoration."

The network reuses `VanillaAutoencoder`'s encoder/decoder — the only new
piece you need to write is the noise-injection function below, which is
applied to the input *before* it reaches the encoder during training.
"""
from __future__ import annotations

import torch

from .vanilla_ae import VanillaAutoencoder


def add_gaussian_noise(x: torch.Tensor, noise_factor: float = 0.3) -> torch.Tensor:
    """Corrupt a batch of images with additive Gaussian noise, then clamp.

    TODO(Phase 1 - Denoising AE): implement additive Gaussian noise
    corruption:
        noisy = x + noise_factor * N(0, 1)
    where `N(0, 1)` has the same shape as `x` (use `torch.randn_like(x)`).
    Clamp the result back to the valid image range [-1, 1] (images in this
    project are normalized with Normalize(0.5, 0.5)) before returning.
    """
    # raise NotImplementedError(
    #     "TODO: implement add_gaussian_noise — add noise_factor * torch.randn_like(x) "
    #     "to x, then torch.clamp(..., -1.0, 1.0)."
    # )
    noisy = x + noise_factor * torch.randn_like(x)
    return torch.clamp(noisy, -1.0, 1.0)

class DenoisingAutoencoder(VanillaAutoencoder):
    """Same architecture as VanillaAutoencoder; trained on noisy inputs.

    During training you should call `add_gaussian_noise` on the input batch
    yourself (see `src/generative_art_studio/training`) and compute the
    reconstruction loss against the *original clean* image, not the noisy
    one — that's what teaches the network to denoise.
    """

    def __init__(self, in_channels: int = 3, latent_dim: int = 128, noise_factor: float = 0.3):
        super().__init__(in_channels=in_channels, latent_dim=latent_dim)
        self.noise_factor = noise_factor

    def forward(self, x: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        if add_noise:
            x = add_gaussian_noise(x, self.noise_factor)
        return super().forward(x)
