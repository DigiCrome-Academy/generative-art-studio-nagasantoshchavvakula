"""CycleGAN: unpaired image-to-image translation via cycle consistency
(Zhu et al. 2017, https://arxiv.org/abs/1703.10593).

Learning objective (docs/PROJECT_BRIEF.md, Phase 3):
"Build CycleGAN for unpaired style transfer."

CycleGAN reuses `PatchGANDiscriminator` from `pix2pix.py` (unconditional
mode: pass a single image, no `img_b`). What's new here is a ResNet-style
generator built from residual blocks, and (in
`src/generative_art_studio/training/losses.py`) the cycle-consistency and
identity losses that let training work *without paired data*.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """A residual block: output = input + ConvBlock(input).

    TODO(Phase 3 - CycleGAN residual blocks): implement the skip
    connection in `forward`: run `x` through `self.block` and add the
    *original* `x` back to the result (`return x + self.block(x)`). This is
    what lets CycleGAN generators be deep enough to translate style while
    preserving the input's overall structure.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # raise NotImplementedError(
        #     "TODO: implement the residual skip connection — return x + self.block(x)."
        # )
        return x + self.block(x)

class CycleGANGenerator(nn.Module):
    """Encoder (downsample) -> N residual blocks -> Decoder (upsample).

    Fully implemented aside from `ResidualBlock.forward` above — once you
    implement the skip connection, this generator works end to end. A
    complete CycleGAN model = two of these (G: A->B, F: B->A) plus two
    `PatchGANDiscriminator`s (D_A, D_B).
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3, features: int = 64, num_residual_blocks: int = 6):
        super().__init__()

        model: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, features, 7),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        c = features
        for _ in range(2):
            model += [
                nn.Conv2d(c, c * 2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(c * 2),
                nn.ReLU(inplace=True),
            ]
            c *= 2

        # Residual blocks (operate at the downsampled resolution)
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(c)]

        # Upsampling
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(c, c // 2, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(c // 2),
                nn.ReLU(inplace=True),
            ]
            c //= 2

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(features, out_channels, 7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
