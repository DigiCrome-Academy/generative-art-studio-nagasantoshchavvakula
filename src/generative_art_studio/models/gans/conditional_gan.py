"""Conditional GAN (cGAN): condition generation on a class label.

Learning objective (docs/PROJECT_BRIEF.md, Phase 2):
"Develop Conditional GAN for controlled generation."

The conditioning trick: embed the integer label into a dense vector, then
concatenate it with the noise vector (Generator) or with the flattened
image (Discriminator) before feeding the rest of the network.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConditionalGenerator(nn.Module):
    """(latent noise, class label) -> generated image."""

    def __init__(self, latent_dim: int = 100, num_classes: int = 2, img_channels: int = 3, img_size: int = 64):
        super().__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        out_dim = img_channels * img_size * img_size
        self.net = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, out_dim),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Generate images conditioned on `labels`.

        TODO(Phase 2 - Conditional GAN): embed `labels` with
        `self.label_embedding`, concatenate the result with `z` along the
        feature dimension (`torch.cat([z, label_emb], dim=1)`), pass the
        combined vector through `self.net`, and reshape the flat output to
        (B, img_channels, img_size, img_size) — see `VanillaGenerator.forward`
        in `vanilla_gan.py` for the reshape pattern.
        """
        # raise NotImplementedError(
        #     "TODO: embed labels, concatenate with z, run through self.net, "
        #     "reshape to an image tensor."
        # )
        label_emb = self.label_embedding(labels)
        combined = torch.cat([z, label_emb], dim=1)
        out = self.net(combined)
        
        return out.view(
            -1, self.img_channels, self.img_size, self.img_size
        )

class ConditionalDiscriminator(nn.Module):
    """(image, class label) -> real/fake probability, conditioned on the label."""

    def __init__(self, num_classes: int = 2, img_channels: int = 3, img_size: int = 64):
        super().__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        in_dim = img_channels * img_size * img_size + num_classes
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Classify `img` as real/fake conditioned on `labels`.

        TODO(Phase 2 - Conditional GAN): flatten `img` to (B, -1), embed
        `labels` with `self.label_embedding`, concatenate
        `[flat_img, label_emb]` along dim=1, and pass through `self.net`.
        """
        # raise NotImplementedError(
        #     "TODO: flatten img, embed labels, concatenate, run through self.net."
        # )
        flat_img = img.flatten(1)
        label_emb = self.label_embedding(labels)
        combined = torch.cat([flat_img, label_emb], dim=1)
        
        return self.net(combined)
