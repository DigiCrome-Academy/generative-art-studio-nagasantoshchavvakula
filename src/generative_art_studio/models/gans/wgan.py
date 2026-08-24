"""Wasserstein GAN (WGAN / WGAN-GP) critic.

Learning objective (docs/PROJECT_BRIEF.md, Phase 2):
"Implement Wasserstein GAN for training stability."

A WGAN "critic" has the *same architecture* as a DCGAN discriminator minus
the final Sigmoid (it outputs an unbounded real-valued score, not a
probability). The Wasserstein loss and gradient-penalty term live in
`src/generative_art_studio/training/losses.py`.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class WGANCritic(nn.Module):
    """Mirrors DCGANDiscriminator's architecture but has no final Sigmoid.

    TODO(Phase 2 - WGAN): build `self.net` with the same 5-block
    conv stack as `DCGANDiscriminator` in `dcgan.py`, but:
      * end with `Conv2d(feat*8, 1, 4, 1, 0)` and **no Sigmoid** — the
        critic's output is an unbounded score, not a probability.
      * if you plan to use gradient penalty (WGAN-GP) rather than weight
        clipping, use `nn.InstanceNorm2d` instead of `nn.BatchNorm2d`
        (BatchNorm couples samples within a batch, which conflicts with
        the per-sample gradient penalty).
    """

    def __init__(self, img_channels: int = 3, feature_maps: int = 64, use_instance_norm: bool = True):
        super().__init__()
        self.feature_maps = feature_maps
        # self.net: nn.Sequential | None = None  # TODO: build per the docstring above
        # raise NotImplementedError(
        #     "TODO: build self.net as described in the class docstring, "
        #     "then remove this raise."
        # )
        norm = nn.InstanceNorm2d if use_instance_norm else nn.BatchNorm2d

        self.net = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(img_channels, feature_maps, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 16x16
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1),
            norm(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 8x8
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1),
            norm(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 -> 4x4
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1),
            norm(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 4x4 -> 1x1
            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        out = self.net(img)
        return out.view(-1, 1)


def clip_weights(critic: nn.Module, clip_value: float = 0.01) -> None:
    """Weight clipping for the original (non-gradient-penalty) WGAN.

    Fully implemented — call after each critic optimizer step when *not*
    using gradient penalty:
        `clip_weights(critic, config.WGAN_CLIP_VALUE)`
    """
    for p in critic.parameters():
        p.data.clamp_(-clip_value, clip_value)
