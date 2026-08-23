"""Loss functions for every model in this project.

This is the single most important file for the rubric's "Evaluation &
Analysis" and architecture-quality criteria — implementing these correctly
*is* what it means to understand VAEs vs. GANs vs. WGANs vs. Pix2Pix vs.
CycleGAN. Each TODO links back to a specific learning objective in
docs/PROJECT_BRIEF.md.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

bce_loss = nn.BCELoss()
l1_loss = nn.L1Loss()
mse_loss = nn.MSELoss()


# ---------------------------------------------------------------------------
# Phase 1 — VAE loss (reconstruction + KL divergence)
# ---------------------------------------------------------------------------
def vae_loss(
    recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, kl_weight: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """VAE loss = reconstruction loss + kl_weight * KL(q(z|x) || N(0, I)).

    TODO(Phase 1 - VAE training): implement
        recon_loss = F.mse_loss(recon_x, x, reduction="sum") / x.size(0)
        kl_loss    = -0.5 * sum(1 + logvar - mu^2 - exp(logvar)) / x.size(0)
        total      = recon_loss + kl_weight * kl_loss

    `kl_weight` lets you experiment with beta-VAE style disentanglement
    (kl_weight > 1) — see docs/PROJECT_BRIEF.md's "latent space
    manipulation" objective.

    Returns:
        (total_loss, recon_loss, kl_loss) — all three are returned so you
        can log them separately during training.
    """
    # raise NotImplementedError(
    #     "TODO: implement vae_loss — MSE reconstruction term + closed-form "
    #     "Gaussian KL divergence term, combined as recon_loss + kl_weight * kl_loss."
    # )
    recon_loss = F.mse_loss(recon_x, x, reduction="sum") / x.size(0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    total = recon_loss + kl_weight * kl_loss
    
    return total, recon_loss, kl_loss

# ---------------------------------------------------------------------------
# Phase 2 — Vanilla / DCGAN / Conditional GAN adversarial loss (BCE-based)
# ---------------------------------------------------------------------------
def discriminator_loss(real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
    """Standard (non-saturating) discriminator loss for a sigmoid-output D.

    TODO(Phase 2 - GAN fundamentals): the discriminator wants to output 1
    for real images and 0 for fake images. Implement:
        real_loss = BCE(real_pred, ones_like(real_pred))
        fake_loss = BCE(fake_pred, zeros_like(fake_pred))
        return (real_loss + fake_loss) / 2
    (`bce_loss` is provided at module level.)
    """
    # raise NotImplementedError(
    #     "TODO: implement discriminator_loss using bce_loss against ones (real) "
    #     "and zeros (fake), averaged."
    # )
    real_loss = bce_loss(real_pred, torch.ones_like(real_pred))
    fake_loss = bce_loss(fake_pred, torch.zeros_like(fake_pred))
    return(real_loss + fake_loss)/2


def generator_loss(fake_pred: torch.Tensor) -> torch.Tensor:
    """Standard (non-saturating) generator loss for a sigmoid-output D.

    TODO(Phase 2 - GAN fundamentals): the generator wants the discriminator
    to output 1 (real) for its fakes — i.e. it wants to *fool* D. Implement
    the non-saturating trick: `BCE(fake_pred, ones_like(fake_pred))`
    (rather than minimizing `log(1 - D(G(z)))`, which saturates early in
    training — see docs/PROJECT_BRIEF.md's "training instability" topic).
    """
    raise NotImplementedError(
        "TODO: implement generator_loss using bce_loss(fake_pred, ones_like(fake_pred))."
    )


# ---------------------------------------------------------------------------
# Phase 2 — WGAN / WGAN-GP loss (Wasserstein distance + gradient penalty)
# ---------------------------------------------------------------------------
def wgan_critic_loss(real_score: torch.Tensor, fake_score: torch.Tensor) -> torch.Tensor:
    """Wasserstein critic loss: maximize (real - fake), i.e. minimize (fake - real).

    TODO(Phase 2 - WGAN training stability): implement
        return fake_score.mean() - real_score.mean()
    Unlike the BCE-based discriminator, the critic has no Sigmoid — its
    output is an unbounded score, and this loss approximates the (negative)
    Earth-Mover / Wasserstein distance between real and fake distributions.
    """
    raise NotImplementedError(
        "TODO: implement wgan_critic_loss — return fake_score.mean() - real_score.mean()."
    )


def wgan_generator_loss(fake_score: torch.Tensor) -> torch.Tensor:
    """Wasserstein generator loss: maximize fake_score, i.e. minimize -fake_score.

    TODO(Phase 2 - WGAN training stability): implement
        return -fake_score.mean()
    """
    raise NotImplementedError("TODO: implement wgan_generator_loss — return -fake_score.mean().")


def gradient_penalty(
    critic: Callable[[torch.Tensor], torch.Tensor],
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """WGAN-GP's gradient penalty: encourages ||grad_x_hat D(x_hat)||_2 ≈ 1.

    TODO(Phase 2 - WGAN-GP): implement
        1. Sample eps ~ U(0, 1), one scalar per sample in the batch, and
           reshape it to broadcast against `real`'s shape
           (e.g. `eps = torch.rand(real.size(0), 1, 1, 1, device=device)`).
        2. Build the interpolated input:
           `interpolated = (eps * real + (1 - eps) * fake).requires_grad_(True)`
        3. Run the critic on it: `scores = critic(interpolated)`.
        4. Compute d(scores)/d(interpolated) with `torch.autograd.grad`,
           using `grad_outputs=torch.ones_like(scores)`,
           `create_graph=True`, `retain_graph=True`.
        5. Flatten the gradient to (B, -1), take its L2 norm per sample,
           and return `((grad_norm - 1) ** 2).mean()`.
    """
    raise NotImplementedError(
        "TODO: implement the WGAN-GP gradient penalty — see the docstring's 5-step recipe."
    )


# ---------------------------------------------------------------------------
# Phase 3 — Pix2Pix loss (adversarial + L1 reconstruction)
# ---------------------------------------------------------------------------
def pix2pix_generator_loss(
    disc_fake_pred: torch.Tensor, fake_img: torch.Tensor, target_img: torch.Tensor, lambda_l1: float = 100.0
) -> torch.Tensor:
    """Pix2Pix generator loss = adversarial loss + lambda_l1 * L1(fake, target).

    TODO(Phase 3 - Pix2Pix / conditional generation): implement
        adv_loss = BCE-with-logits or MSE against ones_like(disc_fake_pred)
                   (PatchGANDiscriminator has no Sigmoid, so use
                   `F.binary_cross_entropy_with_logits`)
        l1 = l1_loss(fake_img, target_img)
        return adv_loss + lambda_l1 * l1
    The large `lambda_l1` weight is what makes Pix2Pix outputs stay close
    to the target structure rather than just "looking real."
    """
    raise NotImplementedError(
        "TODO: implement pix2pix_generator_loss — BCE-with-logits adversarial term "
        "plus lambda_l1 * L1 reconstruction term."
    )


def patchgan_discriminator_loss(disc_real_pred: torch.Tensor, disc_fake_pred: torch.Tensor) -> torch.Tensor:
    """PatchGAN discriminator loss (used by both Pix2Pix and CycleGAN).

    Fully implemented — PatchGANDiscriminator has no output Sigmoid, so we
    use the logits variant of BCE for numerical stability.
    """
    real_loss = F.binary_cross_entropy_with_logits(disc_real_pred, torch.ones_like(disc_real_pred))
    fake_loss = F.binary_cross_entropy_with_logits(disc_fake_pred, torch.zeros_like(disc_fake_pred))
    return (real_loss + fake_loss) / 2


# ---------------------------------------------------------------------------
# Phase 3 — CycleGAN loss (adversarial + cycle-consistency + identity)
# ---------------------------------------------------------------------------
def cycle_consistency_loss(real: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """||F(G(real)) - real||_1 — the loss that lets CycleGAN train on unpaired data.

    TODO(Phase 3 - CycleGAN unpaired style transfer): implement
        return l1_loss(reconstructed, real)
    where `reconstructed = F(G(real))` (translate to the other domain and
    back). This is what constrains the generators without needing paired
    (input, target) examples like Pix2Pix does.
    """
    raise NotImplementedError("TODO: implement cycle_consistency_loss — return l1_loss(reconstructed, real).")


def identity_loss(real: torch.Tensor, same_domain_output: torch.Tensor) -> torch.Tensor:
    """||G(real_B) - real_B||_1 — encourages color/tint preservation.

    Fully implemented. `same_domain_output` = running a generator on an
    image *already* from its target domain (e.g. G_A2B(real_B)); ideally
    it should act as identity.
    """
    return l1_loss(same_domain_output, real)
