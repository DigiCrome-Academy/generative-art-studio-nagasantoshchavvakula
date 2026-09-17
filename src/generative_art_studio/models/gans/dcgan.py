"""Deep Convolutional GAN (DCGAN) with the architectural best practices from
Radford et al. 2015 (https://arxiv.org/abs/1511.06434).

Learning objective (docs/PROJECT_BRIEF.md, Phase 2):
"Build Deep Convolutional GAN (DCGAN) with best practices."

Best practices you are implementing here:
  * Generator: strided ConvTranspose2d (no pooling), BatchNorm on every
    layer except the output, ReLU activations, Tanh output.
  * Discriminator: strided Conv2d (no pooling), BatchNorm on every layer
    except the input, LeakyReLU(0.2) activations, Sigmoid output.
  * Weight initialization: N(0, 0.02) for conv/batchnorm weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DCGANGenerator(nn.Module):
    """Latent vector (B, latent_dim) -> (B, img_channels, 64, 64) image.

    TODO(Phase 2 - DCGAN): implement `self.net` as a stack of 5
    ConvTranspose2d blocks that upsample a (latent_dim, 1, 1) tensor to a
    (img_channels, 64, 64) image, following DCGAN best practices:

        ConvTranspose2d(latent_dim,      feat*8, 4, 1, 0) -> BN -> ReLU   # 1x1   -> 4x4
        ConvTranspose2d(feat*8,          feat*4, 4, 2, 1) -> BN -> ReLU   # 4x4   -> 8x8
        ConvTranspose2d(feat*4,          feat*2, 4, 2, 1) -> BN -> ReLU   # 8x8   -> 16x16
        ConvTranspose2d(feat*2,          feat,   4, 2, 1) -> BN -> ReLU   # 16x16 -> 32x32
        ConvTranspose2d(feat,            img_channels, 4, 2, 1) -> Tanh   # 32x32 -> 64x64

    where `feat = self.feature_maps`. Build this with `nn.Sequential` and
    assign it to `self.net`; `forward` below already reshapes the input.
    """

    def __init__(self, latent_dim: int = 100, img_channels: int = 3, feature_maps: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.feature_maps = feature_maps
        # self.net: nn.Sequential | None = None  # TODO: build per the docstring above
        # raise NotImplementedError(
        #     "TODO: build self.net as described in the class docstring, "
        #     "then remove this raise."
        # )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(feature_maps, img_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.view(z.size(0), self.latent_dim, 1, 1)
        return self.net(z)

class DCGANDiscriminator(nn.Module):
    """(B, img_channels, 64, 64) image -> (B, 1) real/fake probability.

    TODO(Phase 2 - DCGAN): implement `self.net` as the mirror image of the
    generator: 5 strided Conv2d blocks downsampling 64x64 -> 1x1, using
    LeakyReLU(0.2) and BatchNorm (skip BatchNorm on the first layer),
    ending in a Sigmoid:

        Conv2d(img_channels, feat,   4, 2, 1) -> LeakyReLU(0.2)          # 64x64 -> 32x32
        Conv2d(feat,         feat*2, 4, 2, 1) -> BN -> LeakyReLU(0.2)    # 32x32 -> 16x16
        Conv2d(feat*2,       feat*4, 4, 2, 1) -> BN -> LeakyReLU(0.2)    # 16x16 -> 8x8
        Conv2d(feat*4,       feat*8, 4, 2, 1) -> BN -> LeakyReLU(0.2)    # 8x8   -> 4x4
        Conv2d(feat*8,       1,      4, 1, 0) -> Sigmoid                 # 4x4   -> 1x1
    """

    def __init__(self, img_channels: int = 3, feature_maps: int = 64):
        super().__init__()
        self.feature_maps = feature_maps
        # self.net: nn.Sequential | None = None  # TODO: build per the docstring above
        # raise NotImplementedError(
        #     "TODO: build self.net as described in the class docstring, "
        #     "then remove this raise."
        # )
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, feature_maps, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        out = self.net(img)
        return out.view(-1, 1)

def weights_init_dcgan(module: nn.Module) -> None:
    """DCGAN paper weight init: N(0, 0.02) for Conv/ConvTranspose/BatchNorm.

    Fully implemented — call `model.apply(weights_init_dcgan)` right after
    constructing a DCGANGenerator/DCGANDiscriminator, as recommended by the
    DCGAN paper for training stability.
    """
    classname = module.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)
