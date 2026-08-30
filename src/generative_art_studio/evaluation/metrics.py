"""Generative-model evaluation metrics: Fréchet Inception Distance (FID)
and Inception Score (IS).

Learning objective (docs/PROJECT_BRIEF.md, Phase 3 + rubric "Evaluation &
Analysis", 15%): "Evaluate generated images using FID and Inception
Score."

`compute_fid` and `compute_inception_score` implement the *math* of each
metric given feature vectors / class-probability vectors you already have
— this is what the test suite checks, using small synthetic arrays, so you
don't need internet access or a GPU to validate your implementation.

`get_inception_feature_extractor` is a fully-implemented convenience
wrapper around `torchvision`'s pretrained Inception v3 that produces those
feature vectors / probabilities from real images. It downloads pretrained
weights on first use, so it's exercised in your notebooks (with a real
dataset and internet access), not in CI.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg


def compute_fid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    """Fréchet Inception Distance between two sets of feature vectors.

    `real_features` / `fake_features` are (N, D) arrays of Inception-pool
    activations (or, in tests, any (N, D) feature arrays — the formula
    doesn't care where the features came from).

    TODO(Evaluation & Analysis - FID): implement the closed-form Fréchet
    distance between two multivariate Gaussians fit to the two feature
    sets:
        mu1, sigma1 = real_features.mean(axis=0), cov(real_features)
        mu2, sigma2 = fake_features.mean(axis=0), cov(fake_features)
        fid = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2 * sqrt(sigma1 @ sigma2))

    Use `np.cov(features, rowvar=False)` for the covariance and
    `scipy.linalg.sqrtm` for the matrix square root (it can return a
    complex-valued array due to numerical error — take `.real` before
    using it). Lower FID = more similar distributions = better generator.
    """
    # raise NotImplementedError(
    #     "TODO: implement compute_fid — see the docstring for the Fréchet distance formula "
    #     "and the np.cov / scipy.linalg.sqrtm hints."
    # )
    mu1 = real_features.mean(axis=0)
    sigma1 = np.cov(real_features, rowvar=False)
    
    mu2 = fake_features.mean(axis=0)
    sigma2 = np.cov(fake_features, rowvar=False)
    
    mean_diff = mu1 - mu2
    cov_mean = linalg.sqrtm(sigma1 @ sigma2)
    
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    
    fid = mean_diff @ mean_diff + np.trace(
        sigma1 + sigma2 - 2 * cov_mean
    )
    return float(fid)


def compute_inception_score(preds: np.ndarray, splits: int = 10) -> tuple[float, float]:
    """Inception Score from an (N, num_classes) array of softmax class
    probabilities produced by an Inception classifier on generated images.

    TODO(Evaluation & Analysis - Inception Score): implement the classic IS
    formula (Salimans et al. 2016). For each of `splits` equal chunks of
    `preds`:
        1. p(y|x)  = the chunk itself (already softmax probabilities)
        2. p(y)    = p(y|x).mean(axis=0)  — the marginal over that chunk
        3. KL(x)   = sum_y p(y|x) * (log(p(y|x)) - log(p(y)))   for each row x
        4. score   = exp(mean_x KL(x))
    Then return `(mean(scores across splits), std(scores across splits))`.
    Add a small epsilon (e.g. 1e-16) inside the logs to avoid log(0).
    """
    # raise NotImplementedError(
    #     "TODO: implement compute_inception_score — see the docstring for the "
    #     "per-split KL-divergence formula."
    # )
    eps = 1e-16
    scores = []

    for split in np.array_split(preds, splits):
        p_y = split.mean(axis=0)

        kl = split * (
            np.log(split + eps) - np.log(p_y + eps)
        )

        kl = np.sum(kl, axis=1)

        score = np.exp(np.mean(kl))
        scores.append(score)

    return float(np.mean(scores)), float(np.std(scores))

class _InceptionFeatureExtractor(nn.Module):
    """Wraps torchvision's Inception v3 to expose both pool features (for FID)
    and softmax class probabilities (for Inception Score)."""

    def __init__(self):
        super().__init__()
        from torchvision.models import Inception_V3_Weights, inception_v3

        self.inception = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
        self.inception.fc = nn.Identity()  # expose the 2048-d pooled feature
        self.inception.eval()
        # Keep a second head with the real classifier for Inception Score.
        self._classifier_fc = inception_v3(weights=Inception_V3_Weights.DEFAULT).fc
        self._classifier_fc.eval()

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """images: (B, 3, H, W) in [-1, 1]. Returns (pool_features, class_probs)."""
        images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
        images = (images + 1) / 2  # [-1, 1] -> [0, 1], what torchvision's weights expect
        features = self.inception(images)
        logits = self._classifier_fc(features)
        probs = F.softmax(logits, dim=1)
        return features, probs


def get_inception_feature_extractor() -> _InceptionFeatureExtractor:
    """Fully implemented. Downloads pretrained Inception v3 weights on first
    call (requires internet) — use this in your notebooks to get real
    `real_features`/`fake_features`/`preds` arrays to feed into
    `compute_fid` / `compute_inception_score`. Not exercised by the
    offline test suite.
    """
    return _InceptionFeatureExtractor()
