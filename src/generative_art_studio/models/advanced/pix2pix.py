"""Pix2Pix: paired image-to-image translation with a U-Net generator and a
PatchGAN discriminator (Isola et al. 2017, https://arxiv.org/abs/1611.07004).

Learning objective (docs/PROJECT_BRIEF.md, Phase 3):
"Implement Pix2Pix for image-to-image translation."

The defining trick of the U-Net generator is the **skip connection**: each
encoder (downsampling) feature map is concatenated onto the matching
decoder (upsampling) feature map, so fine spatial detail from the input
isn't lost through the bottleneck. That's the TODO below.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class UNetDown(nn.Module):
    """One encoder block: Conv2d(stride 2) -> [InstanceNorm] -> LeakyReLU."""

    def __init__(self, in_channels: int, out_channels: int, normalize: bool = True):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=not normalize)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetUp(nn.Module):
    """One decoder block: ConvTranspose2d(stride 2) -> InstanceNorm -> ReLU [-> Dropout]."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip_input: torch.Tensor) -> torch.Tensor:
        """Upsample `x`, then concatenate the encoder's `skip_input` onto it.

        TODO(Phase 3 - Pix2Pix U-Net skip connections): run `x` through
        `self.block`, then concatenate the result with `skip_input` along
        the channel dimension: `torch.cat([upsampled, skip_input], dim=1)`.
        This is what makes it a *U-Net* rather than a plain encoder-decoder.
        """
        # raise NotImplementedError(
        #     "TODO: upsample x with self.block, then torch.cat([x, skip_input], dim=1)."
        # )
        unsampled = self.block(x)
        return torch.cat([unsampled, skip_input], dim=1)

class UNetGenerator(nn.Module):
    """64x64 U-Net generator: 4 down blocks, bottleneck, 4 up blocks with skips."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3, features: int = 64):
        super().__init__()
        self.down1 = UNetDown(in_channels, features, normalize=False)      # 64 -> 32
        self.down2 = UNetDown(features, features * 2)                     # 32 -> 16
        self.down3 = UNetDown(features * 2, features * 4)                 # 16 -> 8
        self.down4 = UNetDown(features * 4, features * 8)                 # 8  -> 4

        self.up1 = UNetUp(features * 8, features * 4, dropout=0.5)        # 4  -> 8   (+ skip down3)
        self.up2 = UNetUp(features * 8, features * 2)                     # 8  -> 16  (+ skip down2)
        self.up3 = UNetUp(features * 4, features)                         # 16 -> 32  (+ skip down1)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, 4, stride=2, padding=1),  # 32 -> 64
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        u1 = self.up1(d4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        return self.final(u3)


class PatchGANDiscriminator(nn.Module):
    """Classifies overlapping NxN patches of an image as real/fake, rather
    than the whole image at once — this is what gives Pix2Pix/CycleGAN
    sharp high-frequency detail. Fully implemented.

    For Pix2Pix (paired, conditional) pass `in_channels = img_channels * 2`
    and feed `torch.cat([input_img, target_or_generated_img], dim=1)`.
    For CycleGAN (unpaired, unconditional) pass `in_channels = img_channels`.
    """

    def __init__(self, in_channels: int = 6):
        super().__init__()

        def block(in_c: int, out_c: int, normalize: bool = True) -> list[nn.Module]:
            layers: list[nn.Module] = [nn.Conv2d(in_c, out_c, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 512, 4, padding=1, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1),
        )

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor | None = None) -> torch.Tensor:
        x = torch.cat([img_a, img_b], dim=1) if img_b is not None else img_a
        return self.model(x)
