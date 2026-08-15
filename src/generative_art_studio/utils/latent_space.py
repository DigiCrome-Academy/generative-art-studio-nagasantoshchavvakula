"""Latent-space sampling and interpolation utilities.

`sample_latent` and `slerp` are fully implemented. `interpolate_latent`
has a TODO — this is Phase 1's "latent space interpolation and
visualization" learning objective (docs/PROJECT_BRIEF.md, Phase 1).
"""
from __future__ import annotations

import torch


def sample_latent(batch_size: int, latent_dim: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Sample `batch_size` latent vectors from a standard normal N(0, I)."""
    return torch.randn(batch_size, latent_dim, device=device)


def slerp(z1: torch.Tensor, z2: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical linear interpolation between two latent vectors z1, z2 at t in [0, 1].

    Fully implemented reference utility — feel free to call this from
    `interpolate_latent` or directly from your notebooks.
    """
    z1n = z1 / z1.norm()
    z2n = z2 / z2.norm()
    dot = torch.clamp((z1n * z2n).sum(), -1.0, 1.0)
    omega = torch.acos(dot)
    if torch.abs(omega) < 1e-6:
        return (1.0 - t) * z1 + t * z2
    sin_omega = torch.sin(omega)
    return (torch.sin((1.0 - t) * omega) / sin_omega) * z1 + (torch.sin(t * omega) / sin_omega) * z2


def interpolate_latent(z1: torch.Tensor, z2: torch.Tensor, steps: int = 10) -> torch.Tensor:
    """Return a (steps, latent_dim) tensor linearly interpolating from z1 to z2.

    TODO(Phase 1 - VAE latent space): implement linear interpolation between
    two 1-D latent vectors `z1` and `z2` using `steps` evenly spaced points
    from t=0 (== z1) to t=1 (== z2) inclusive.

    Hint: build a tensor of `t` values with `torch.linspace(0, 1, steps)`
    and combine it with z1/z2 via broadcasting: `(1 - t) * z1 + t * z2`.
    """
    if z1.shape != z2.shape or z1.dim() != 1:
        raise ValueError("z1 and z2 must be 1-D tensors of the same shape")
    # raise NotImplementedError(
    #     "TODO: implement interpolate_latent — see docstring for the linear "
    #     "interpolation formula and torch.linspace hint."
    # )
    t = torch.linspace(0, 1, steps, device=z1.device, dtype=z1.dtype)
    return (1 - t.unsqueeze(1)) * z1 + t.unsqueeze(1) * z2
