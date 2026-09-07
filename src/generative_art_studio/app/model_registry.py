"""Platform logic for the Generative Art Studio app, kept separate from the
Streamlit UI so it can be unit-tested without spinning up a server.

Learning objective (docs/PROJECT_BRIEF.md, Phase 4):
"Create unified interface for all generative models", "Implement real-time
generation with user controls", "Add style mixing and interpolation
features", "Implement export functionality for high-resolution outputs."
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from PIL import Image

from ..config import GAN_LATENT_DIM, IMAGE_CHANNELS, VAE_LATENT_DIM
from ..models.autoencoders.vae import VAE
from ..models.gans.vanilla_gan import VanillaGenerator
from ..utils.viz import denormalize


@dataclass
class ModelEntry:
    """One entry in the platform's model registry."""

    name: str
    description: str
    build: Callable[[], nn.Module]
    latent_dim: int


def _build_vae() -> nn.Module:
    return VAE(in_channels=IMAGE_CHANNELS, latent_dim=VAE_LATENT_DIM)


def _build_vanilla_gan() -> nn.Module:
    return VanillaGenerator(latent_dim=GAN_LATENT_DIM, img_channels=IMAGE_CHANNELS)


# Add an entry here for every generator you want selectable in the platform
# UI (DCGAN, Conditional GAN, Pix2Pix, CycleGAN, ...) once you've
# implemented it. VAE and vanilla GAN are wired up already as examples.
MODEL_REGISTRY: dict[str, ModelEntry] = {
    "vae": ModelEntry("Variational Autoencoder", "Sample from a learned latent Gaussian.", _build_vae, VAE_LATENT_DIM),
    "vanilla_gan": ModelEntry("Vanilla GAN", "Fully-connected Generator/Discriminator pair.", _build_vanilla_gan, GAN_LATENT_DIM),
}


def list_available_models() -> list[str]:
    """Return the keys of every model registered in MODEL_REGISTRY."""
    return list(MODEL_REGISTRY.keys())


def load_model(model_key: str, checkpoint_path: str | Path | None = None, device: str = "cpu") -> nn.Module:
    """Build a model by registry key and optionally load trained weights.

    Fully implemented. If `checkpoint_path` is None or doesn't exist, the
    model is returned with its (untrained) random initialization — handy
    for demoing the platform's UI before you've finished training anything.
    """
    if model_key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{model_key}'. Available: {list_available_models()}")
    model = MODEL_REGISTRY[model_key].build().to(device)
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    model.eval()
    return model


def generate_samples(model_key: str, model: nn.Module, num_samples: int, seed: int | None = None, device: str = "cpu") -> torch.Tensor:
    """Generate a batch of images from `model` for the platform's "generate" button.

    TODO(Phase 4 - real-time generation with user controls): given a
    model looked up by `model_key`, produce a `(num_samples, C, H, W)`
    batch of generated images:
      * if `model_key == "vae"`, call `model.sample(num_samples, device=device)`
        (see `VAE.sample` in `models/autoencoders/vae.py`).
      * otherwise (any GAN generator), sample a latent batch with
        `sample_latent(num_samples, MODEL_REGISTRY[model_key].latent_dim, device)`
        (from `utils.latent_space`) and call `model(z)`.
    Seed the RNG first if `seed is not None` (`torch.manual_seed(seed)`) so
    the UI's "regenerate with this seed" control is reproducible.
    """
    if seed is not None:
        torch.manual_seed(seed)
    if model_key == "vae":
        return model.sample(num_samples, device=device)
    
    from ..utils.latent_space import sample_latent
    z = sample_latent(num_samples, MODEL_REGISTRY[model_key].latent_dim, device)
    return model(z)


def export_image(image: torch.Tensor, path: str | Path, scale_factor: int = 4) -> Path:
    """Upscale a single generated image tensor and save it as a high-res PNG.

    TODO(Phase 4 - export functionality for high-resolution outputs):
      1. Denormalize `image` from [-1, 1] to [0, 1] with `denormalize`
         (imported above) — add a batch dim first if needed
         (`image.unsqueeze(0)`), then squeeze it back off.
      2. Upsample it by `scale_factor` using
         `torch.nn.functional.interpolate(..., scale_factor=scale_factor, mode="bicubic", align_corners=False)`.
      3. Convert to a `PIL.Image` (permute to HWC, multiply by 255, cast to
         uint8 numpy array, `Image.fromarray(...)`) and save it to `path`
         (create parent directories with `Path(path).parent.mkdir(parents=True, exist_ok=True)`).
      4. Return the `Path` you saved to.
    """
    # raise NotImplementedError(
    #     "TODO: implement export_image — denormalize, upsample by scale_factor, "
    #     "convert to PIL, and save a high-resolution PNG. See the docstring."
    # )
    image = image.unsqueeze(0)
    image = denormalize(image).squeeze(0)
    
    image = torch.nn.functional.interpolate(
        image.unsqueeze(0),
        scale_factor=scale_factor,
        mode="bicubic",
        align_corners=False
    ).squeeze(0)
    
    image = image.clamp(0, 1)
    
    array = (
        image.permute(1, 2, 0).mul(255).byte().cpu().numpy()
    )
    
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(output_path)
    return output_path

class Gallery:
    """In-memory gallery of generated artwork for the platform's gallery tab.

    Fully implemented — the Streamlit app stores one of these in
    `st.session_state` so it survives reruns within a browser session.
    """

    def __init__(self):
        self._items: list[dict] = []

    def add(self, image: torch.Tensor, model_key: str, seed: int | None = None) -> None:
        self._items.append({"image": image.detach().cpu(), "model_key": model_key, "seed": seed})

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)
