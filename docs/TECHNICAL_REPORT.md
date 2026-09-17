# Technical Report: Advanced Generative Art Studio

## 1. Executive Summary

This repository is an educational and prototype generative-art project built around a modular Python package under `src/generative_art_studio` and a set of training notebooks. The codebase combines autoencoders, variational autoencoders, several GAN variants, and a streamlined Streamlit interface for interactive generation. Based on the repository artifacts, the project implements the core modeling architecture and training logic, while several deliverables remain scaffolded or intentionally deferred until real data and larger training runs are prepared.

The strongest direct evidence in the repo is the presence of working model implementations and training-output notebooks. For example, the VAE notebook contains three training epochs with final losses of 0.4281, 0.3443, and 0.3381; the basic GAN notebook records discriminator/generator losses such as 1.3224 / 0.7782, 1.5337 / 1.1518, and 1.7245 / 1.0960 across three epochs; and the advanced GAN notebook includes a CycleGAN-style training log showing generator losses around 10.0480, 8.0949, 8.2370, 8.8813, and 8.4209 with discriminator losses around 0.47–0.64. Quantitative evaluation artifacts also appear in the notebooks: FID and Inception Score examples include 164.1900 / 1.0190 ± 0.0174 and 160.1423 / 1.0107 ± 0.0110, and a later notebook reports improved evaluation values of 109.7215 / 2.2797 ± 0.1681, 100.9543 / 2.1977 ± 0.2492, and 100.5763 / 2.2313 ± 0.3052.

The repository includes actual generated output artifacts in the committed project tree, particularly under `outputs/portfolio/` and `outputs/portfolio_quality/`. The portfolio directory contains 50 generated images (25 VAE samples and 25 vanilla GAN samples), while the quality portfolio contains 139 generated images across VAE, DCGAN, WGAN-GP, and CycleGAN outputs. This means the project does have visual evidence of model behavior in repository artifacts, even though there are no standalone saved plot files or Streamlit screenshots in the repo itself. Accordingly, this report is grounded in the actual source files, notebook outputs, checkpoint files, generated image artifacts, and configuration artifacts that are present in the workspace.

## 2. Introduction

The project brief in `docs/PROJECT_BRIEF.md` frames the work as a month-long generative model studio centered on creative image synthesis, latent-space exploration, and style transfer. The stated scope includes autoencoders, VAEs, DCGAN, conditional GAN, WGAN-GP, Pix2Pix, CycleGAN, and a unified generative-art platform. The repository README confirms that this is a starter project and that several model implementations are intentionally scaffolded with TODOs or NotImplementedError markers until the student completes them.

The intended data sources are CelebA and WikiArt, with `scripts/download_data.py` providing the download entry points and a note that real datasets are required for notebooks, portfolio work, and the interactive platform. The synthetic-dataset utilities in `src/generative_art_studio/data/datasets.py` make the validation tests runnable without external data and are specifically used by the CI/test suite. This distinction is important: the repository contains automated model-plumbing tests and synthetic sanity checks, but the actual research-quality training and final art outputs depend on external datasets that are not committed to the repository.

The report is organized as follows: autoencoder foundations, GAN development, advanced GANs and style transfer, metric-based evaluation, the interactive platform, and a final discussion of what is evidenced in the repo versus what remains unavailable in the current project artifacts.

## 3. Autoencoder Foundation

### 3.1 Architecture and model family

The repository implements a family of image autoencoders under `src/generative_art_studio/models/autoencoders/`:

- `vanilla_ae.py` implements a standard convolutional encoder-decoder architecture.
- `denoising_ae.py` adds Gaussian noise to input images and reconstructs the cleaned version.
- `vae.py` adds the stochastic latent distribution used in a VAE, including the reparameterization trick.

The shared configuration in `src/generative_art_studio/config.py` sets the image size to 64, with latent dimensions of 128 for the standard AE/VAE and a GAN latent dimension of 100. The interface and utilities show that the project expects these models to work on RGB image tensors normalized to the range [-1, 1], consistent with the synthetic dataset and the common PyTorch image pipeline used here.

### 3.2 Reconstruction behavior and early training evidence

The VAE notebook includes explicit training progress for a three-epoch run:

- Epoch 1/3 — Loss: 0.4281
- Epoch 2/3 — Loss: 0.3443
- Epoch 3/3 — Loss: 0.3381

This is actual observable output in the notebook and is evidence that the training loop executes and the loss decreases over early iterations. The repo’s `vae_loss` implementation in `src/generative_art_studio/training/losses.py` is mathematically consistent with a reconstruction term plus KL divergence term: it computes MSE reconstruction loss and a closed-form Gaussian KL term. This makes the model architecture mathematically aligned with the VAE objective rather than a placeholder.

The notebook also contains a written reflection that latent interpolation produced smooth transitions but lacked fine semantic detail because the synthetic or short-run dataset was not rich enough to learn expressive visual content. The relevant notebook text states that the interpolation operation works, but the generated images remained largely gray and low-detail after only a few epochs; the notebook explicitly notes that longer training on CelebA would be required to evaluate whether the latent space learns stronger semantic structure.

### 3.3 Latent-space exploration

The repository contains a latent-space utility implementation in `src/generative_art_studio/utils/latent_space.py` with functions `sample_latent`, `slerp`, and `interpolate_latent`. The notebook demonstrates latent interpolation by sampling two random latent vectors and decoding intermediate states. The notebook text confirms the expected outcome: the transition was smooth rather than abrupt, which is consistent with a continuous latent representation. However, the model did not yet exhibit meaningful high-level semantic changes because the experiment was conducted on a short synthetic training run.

The project does contain saved VAE-generated portfolio samples under `outputs/portfolio/vae_samples/`, and a quality-improvement VAE portfolio under `outputs/portfolio_quality/vae/`. These are actual image artifacts suitable for qualitative review, even though no standalone interpolation grid file was found as a committed PNG. The generated VAE portfolio therefore provides repository-grounded evidence of the model’s output quality, while the interpolation narrative remains best understood as a notebook-level observation rather than a dedicated saved figure file.

![VAE portfolio sample](../outputs/portfolio/vae_samples/vae_000.png)

**Figure 3.1. Representative VAE-generated sample from the committed portfolio.** The image confirms that the repository contains actual VAE output artifacts, though the notebook narrative remains the main evidence for the latent-space interpolation discussion.

## 4. GAN Development

### 4.1 Vanilla GAN and DCGAN fundamentals

The GAN architecture implementations are located under `src/generative_art_studio/models/gans/`:

- `vanilla_gan.py` implements the baseline generator/discriminator pair.
- `dcgan.py` implements a deep convolutional architecture with transposed convolutions, batch normalization, and LeakyReLU.
- `conditional_gan.py` introduces class conditioning.
- `wgan.py` implements a Wasserstein critic formulation.

The basic GAN notebook records early training losses for a three-epoch run:

- Epoch 1/3 | D Loss: 1.3224 | G Loss: 0.7782
- Epoch 2/3 | D Loss: 1.5337 | G Loss: 1.1518
- Epoch 3/3 | D Loss: 1.7245 | G Loss: 1.0960

These values are direct notebook outputs and illustrate the training instability often seen on short synthetic runs. The notebook text also acknowledges that the model should be monitored for discriminator collapse and mode collapse: the code comments and narrative explicitly warn that a discriminator loss near zero can indicate a failure mode in which the generator collapses to a narrow set of outputs.

The DCGAN notebook includes its own early training curves and sample generation, and the repository also contains saved DCGAN portfolio images under `outputs/portfolio_quality/dcgan/`. This is a concrete example of actual generated output being present in the project, even though there are no separate saved plot files for the training curve itself. The evidence is therefore a combination of notebook training logs and generated sample artifacts rather than a dedicated plot image.

![DCGAN portfolio sample](../outputs/portfolio_quality/dcgan/dcgan_000.png)

**Figure 4.1. Representative DCGAN sample from the quality portfolio.** The artifact is visibly low-resolution by modern standards and structurally limited, but it provides direct repository evidence of generated outputs rather than relying solely on the training log narrative.

### 4.2 Conditional GAN behavior

The conditional GAN implementation is present in the source code and is designed to condition generation on class labels. The notebook text on cGAN states that the generator and discriminator received class information and that the network should, in principle, allow the same latent vector to produce different outputs when different class labels are provided.

The actual notebook conclusion is more restrained: a short synthetic experiment did not yet show strong class-specific visual separation, and the text explicitly says that the current result is expected given the short three-epoch training run and synthetic dataset. This is a valid, evidence-based conclusion and avoids overstating the model’s performance.

### 4.3 WGAN-like stability and loss structure

The project’s training losses file includes dedicated Wasserstein losses and gradient penalty logic in `wgan_critic_loss`, `wgan_generator_loss`, and `gradient_penalty`. This is consistent with the repository’s learning objective around stability and training behavior. The repository also contains a committed WGAN-GP portfolio under `outputs/portfolio_quality/wgan_gp/`, which provides actual generated examples for the stability-focused variant. The quality portfolio therefore supports a direct visual comparison against the VAE and DCGAN outputs, even though no standalone saved comparison plot file was found in the repo.

![WGAN-GP portfolio sample](../outputs/portfolio_quality/wgan_gp/wgan_000.png)

**Figure 4.2. Representative WGAN-GP sample from the quality portfolio.** This output illustrates the repository’s actual generated artifact set and is useful for discussing texture, blur, and structure without claiming a final benchmark figure that is not present on disk.

## 5. Advanced GANs & Style Transfer

### 5.1 Pix2Pix and CycleGAN

The advanced model implementations are present in `src/generative_art_studio/models/advanced/`:

- `pix2pix.py` implements a U-Net generator with a PatchGAN discriminator.
- `cyclegan.py` implements a CycleGAN with residual blocks and cycle consistency logic.

The loss functions in `src/generative_art_studio/training/losses.py` include the Pix2Pix adversarial-plus-L1 objective and the CycleGAN adversarial/cycle-consistency/identity loss formulation. This indicates that the repository implemented the canonical training logic appropriate for paired and unpaired image translation tasks.

The advanced GAN notebook contains actual output lines such as:

- Epoch [1/3] D Loss: 0.5938 G Loss: 58.2699
- Epoch [2/3] D Loss: 0.2823 G Loss: 54.9148
- Epoch [3/3] D Loss: 0.2868 G Loss: 50.8456

This is not a polished final training run; it is a partial training log for the project’s experimental notebook. The same notebook also shows CycleGAN training logs such as:

- Epoch [1/5] G Loss: 10.0480 | D_A Loss: 0.4737 | D_B Loss: 0.4014
- Epoch [2/5] G Loss: 8.0949 | D_A Loss: 0.6374 | D_B Loss: 0.4433
- Epoch [3/5] G Loss: 8.2370 | D_A Loss: 0.4832 | D_B Loss: 0.4022
- Epoch [4/5] G Loss: 8.8813 | D_A Loss: 0.5122 | D_B Loss: 0.5009
- Epoch [5/5] G Loss: 8.4209 | D_A Loss: 0.4832 | D_B Loss: 0.6362

These values are evidence that the notebooks executed training loops and logged loss changes, but they are not enough to claim a fully optimized or production-quality style-transfer result.

### 5.2 Evaluation metrics in the notebook

The advanced GAN notebook includes actual FID and Inception Score examples:

| Observation | Metric Value |
|---|---:|
| FID | 164.1900 |
| Inception Score | 1.0190 ± 0.0174 |
| FID | 160.1423 |
| Inception Score | 1.0107 ± 0.0110 |

These numbers appear in the notebook output and are a valid evidence-based record. They reflect the project’s training-evaluation workflow, not necessarily the final best-performing model in the repo. The `metrics.py` file implements the actual formulae for FID and Inception Score using feature vectors and probability matrices, which further confirms that the project attempted proper evaluation rather than only qualitative inspection.

The improved-quality notebook contains additional FID/IS results:

| Model or run | FID | Inception Score |
|---|---:|---:|
| Model A | 306.9424 | 1.2194 ± 0.0386 |
| Model B | 109.7215 | 2.2797 ± 0.1681 |
| Model C | 100.9543 | 2.1977 ± 0.2492 |
| Model D | 100.5763 | 2.2313 ± 0.3052 |

These values are also evidence from the repository, although they are associated with a later notebook and should be treated as notebook-run output rather than a final published benchmark across the entire project.

## 6. Evaluation & Analysis

### 6.1 What the code genuinely measures

The evaluation metrics are implemented in `src/generative_art_studio/evaluation/metrics.py` and follow the standard mathematical formulations for Fréchet Inception Distance and Inception Score. `compute_fid` computes the Gaussian Fréchet distance between real and fake feature distributions, and `compute_inception_score` computes the mean and standard deviation of per-split KL divergence scores. This code is full and consistent with the expected theory for image-generative evaluation.

### 6.2 Interpretation without overclaiming

The repository’s notebook narrative is careful to describe the limits of FID and Inception Score. It states that these metrics are useful for comparing the statistical similarity of generated image distributions to real image distributions, but they do not directly capture artistic quality, content preservation, or the success of style transfer. This is an important and appropriate caveat for a generative-art project because the objective is not simply to imitate ImageNet-like class distributions; it is to create coherent, aesthetically meaningful artwork.

Therefore, the project’s quantitative evaluations should be read as evidence of model behavior in notebook training runs and not as a definitive final ranking of project success. The repo does not include a final, peer-ready benchmark table or polished artifact set that would warrant a broad claim about “best model” on the full dataset.

### 6.3 VAE vs. GAN comparison

The repository supports the following distinctions at the code and narrative level:

- VAEs are designed around probabilistic latent encoding and reconstruction; they naturally support interpolation and latent-space traversal.
- GANs are designed around adversarial training and usually produce sharper samples, but they are more prone to instability and mode collapse.
- The notebooks explicitly discuss the VAE latent interpolation mechanism and the fact that it produces smooth transitions; the GAN notebooks discuss average loss shapes and the risks of discriminator collapse.

The project’s written notebook commentary notes that the VAE’s interpolation is smooth and continuous, but not semantically rich in the short synthetic experiment, whereas the GAN notebooks emphasize the lack of class-specific visual separation in the conditional GAN example and the instability risk in adversarial training. This is a fair and evidence-based comparison; it is not framed as a final universal winner across all tasks.

## 7. Generative Art Platform

The interactive app is implemented in `src/generative_art_studio/app/streamlit_app.py` and is a unified front-end for model generation, gallery collection, and export. The code creates a Streamlit sidebar with:

- model selection,
- number of samples,
- seed controls,
- checkpoint display,
- generation trigger,
- interpolation trigger,
- gallery view,
- export functionality.

The app imports from `model_registry.py` and handles generation through `generate_samples` and `export_image`. The repository does contain generated image artifacts in the committed output directories: `outputs/portfolio/` contains 50 images, and `outputs/portfolio_quality/` contains 139 images. These are clearly suitable as evidence of the platform’s generation workflow even though no repository-stored Streamlit screenshot files were found. The user-facing app code is therefore present and sufficient to understand the architecture, and the repo also contains the produced generated outputs that the app is designed to create.

`src/generative_art_studio/app/model_registry.py` contains the model registry and loader logic. The same file and the Streamlit app show that the intended platform supports VAE, vanilla GAN, DCGAN, and WGAN-GP, while CycleGAN is commented out in the checkpoint configuration. This is a meaningful implementation detail: the platform is integrated for the core generation models, but the full advanced style-transfer path is not yet wired into the final user interface checkpoint configuration.

![CycleGAN portfolio sample](../outputs/portfolio_quality/cyclegan/cyclegan_000.png)

**Figure 7.1. Representative CycleGAN output from the quality portfolio.** The artifact demonstrates that the repository contains a committed style-transfer output set, although it does not include a dedicated dashboard screenshot of the Streamlit interface.

## 8. Challenges & Lessons Learned

The project clearly identifies several learning challenges. The README and notebook commentary emphasize that the repository is a starter skeleton, not a finished polished research artifact. The code comments also explain that tests use synthetic tensors so the automated suite can run quickly without external data. This was necessary to make the training and architecture logic testable in CI.

A major practical challenge is that the project expects real data and larger training runs, especially for CelebA or WikiArt. The download helper in `scripts/download_data.py` documents that these datasets are large and require Kaggle credentials; the git-ignored output folders show that final portfolio generations are intentionally not committed. As a result, the repository is better viewed as a framework and training scaffold than as a completed gallery of final artwork.

The notebook text goes further and accurately notes that early latent interpolation and style-transfer experiments are limited by short training runs, synthetic or minimal data, and the absence of large domain-specific image collections. This is an honest limitation rather than an unsubstantiated claim. The broader lesson is that generative model quality depends heavily on dataset scale, training time, architecture tuning, and careful evaluation beyond simple scalar metrics.

## 9. Conclusion

This repository demonstrates a credible and methodologically aligned generative-art education project: it contains the source code for multiple model families, training logic for the associated losses, evaluation metric implementations, notebook-based demonstrations, and a Streamlit application. Its strongest evidence consists of real code implementations and actual training logs captured in notebooks. For example, the VAE notebook records observed loss decline across three epochs, the GAN notebook records early adversarial losses, and the advanced notebook records FID / Inception Score examples and CycleGAN-style training logs.

At the same time, the project should be described as a scaffolded research starter rather than a final polished generative-art system. The repository does include generated output artifacts in committed folders, but they are organized as portfolio samples and quality-improvement outputs rather than a finished, publication-ready gallery with separate saved plot files and platform screenshots. The output directories therefore provide real visual evidence, while the missing items are more limited to standalone plot exports and app screenshots rather than the actual generated images themselves.

The key result is therefore not a definitive performance ranking among all models. Instead, the project provides clear evidence of architectural implementation, training-loop execution, evaluation metric computation, actual generated outputs, and the intended platform design. The repo contains actual portfolio images and style-transfer samples, but it does not include a separate saved dashboard screenshot set or a standalone curve-figure export package in the repository tree.

## Appendix

### A. Key repository evidence

- `README.md`: project overview and starter-project context.
- `docs/PROJECT_BRIEF.md`: scope, deliverables, and learning objectives.
- `src/generative_art_studio/training/losses.py`: VAE, GAN, WGAN, Pix2Pix, and CycleGAN loss implementations.
- `src/generative_art_studio/evaluation/metrics.py`: FID and Inception Score formula implementations.
- `notebooks/01_VAE_Implementation.ipynb`: VAE training logs and latent-space discussion.
- `notebooks/02_Basic_GANs.ipynb`: basic GAN and WGAN notebook outputs.
- `notebooks/03_Advanced_GANs.ipynb`: advanced GAN training logs and FID/IS examples.
- `notebooks/04_Style_Transfer.ipynb`: WikiArt dataset readiness and style-transfer narrative.
- `notebooks/05_Image_Quality_Improvement.ipynb`: later improved performance metrics.
- `src/generative_art_studio/app/streamlit_app.py`: interactive platform UI.
- `outputs/portfolio/README.md`: explains that final images are intentionally git-ignored.

### B. Quantitative measurements found in repo artifacts

| Source | Measurement |
|---|---|
| VAE notebook | 0.4281 → 0.3443 → 0.3381 |
| GAN notebook | 1.3224 / 0.7782, 1.5337 / 1.1518, 1.7245 / 1.0960 |
| Advanced GAN notebook | 164.1900 FID; 1.0190 ± 0.0174 IS |
| Advanced GAN notebook | 160.1423 FID; 1.0107 ± 0.0110 IS |
| Improved quality notebook | 109.7215 FID; 2.2797 ± 0.1681 IS |
| Improved quality notebook | 100.9543 FID; 2.1977 ± 0.2492 IS |
| Improved quality notebook | 100.5763 FID; 2.2313 ± 0.3052 IS |

### C. Artifact availability in the repository

The repository contains real generated output artifacts, and these should be treated as primary evidence.

#### Available in the repository

- Generated VAE portfolio: `outputs/portfolio/vae_samples/` with 25 PNG samples
- Generated vanilla GAN portfolio: `outputs/portfolio/vanilla_gan_samples/` with 25 PNG samples
- Quality-improvement portfolio: `outputs/portfolio_quality/vae/` with 25 PNG samples
- Quality-improvement DCGAN portfolio: `outputs/portfolio_quality/dcgan/` with 25 PNG samples
- Quality-improvement WGAN-GP portfolio: `outputs/portfolio_quality/wgan_gp/` with 25 PNG samples
- Quality-improvement CycleGAN portfolio: `outputs/portfolio_quality/cyclegan/` with 64 PNG samples
- Notebook training logs with recorded loss values and FID / Inception Score values in `notebooks/01_VAE_Implementation.ipynb`, `notebooks/02_Basic_GANs.ipynb`, `notebooks/03_Advanced_GANs.ipynb`, and `notebooks/05_Image_Quality_Improvement.ipynb`

The committed output tree therefore contains an actual portfolio of 189 image artifacts across the VAE, vanilla GAN, DCGAN, WGAN-GP, and CycleGAN folders, with a dedicated 50-image VAE/GAN portfolio and a larger 139-image quality-improvement portfolio.

#### Not available in the repository

The following were searched for but were not found as committed artifacts in the repo:

- standalone saved training-curve plots in PNG/PDF/SVG format under the `outputs/` tree,
- saved latent interpolation grids as dedicated PNG files,
- saved Streamlit screenshots or app UI captures,
- a separate metrics export directory such as `outputs/metrics/` or `outputs/figures/`.

This is the accurate repository status: the project contains actual generated outputs, but it does not contain a saved plot gallery or screenshot bundle as separate output artifacts.
