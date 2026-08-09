[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/2QZzXgq0)
# Advanced Generative Art Studio 🎨🤖

A hands-on, auto-graded starter project for **Month 5: Advanced Generative Models**.
You will build a creative AI toolkit that combines **Autoencoders**, **Variational
Autoencoders (VAEs)**, and a family of **GANs** (DCGAN, Conditional GAN, WGAN,
Pix2Pix, CycleGAN) into a single interactive **Generative Art Platform**.

This repository is a **skeleton**, not a finished project. Code you need to write is
marked with `# TODO` and raises `NotImplementedError` until you implement it. A test
suite (`tests/`) checks your work automatically, and a GitHub Actions workflow runs
that suite **every time you push**, posting a weighted score to the Actions run summary.

---

## 1. What you're building

| Phase | Week | Topic | Where |
|---|---|---|---|
| 1 | 1 | Vanilla AE, Denoising AE, VAE + latent space | [`src/generative_art_studio/models/autoencoders/`](src/generative_art_studio/models/autoencoders/), [`notebooks/01_VAE_Implementation.ipynb`](notebooks/01_VAE_Implementation.ipynb) |
| 2 | 2 | Vanilla GAN, DCGAN, Conditional GAN, WGAN | [`src/generative_art_studio/models/gans/`](src/generative_art_studio/models/gans/), [`notebooks/02_Basic_GANs.ipynb`](notebooks/02_Basic_GANs.ipynb) |
| 3 | 3 | Pix2Pix, CycleGAN, FID / Inception Score | [`src/generative_art_studio/models/advanced/`](src/generative_art_studio/models/advanced/), [`notebooks/03_Advanced_GANs.ipynb`](notebooks/03_Advanced_GANs.ipynb), [`notebooks/04_Style_Transfer.ipynb`](notebooks/04_Style_Transfer.ipynb) |
| 4 | 4 | Unified Streamlit art platform | [`src/generative_art_studio/app/streamlit_app.py`](src/generative_art_studio/app/streamlit_app.py) |

Full brief: [docs/PROJECT_BRIEF.md](docs/PROJECT_BRIEF.md). Grading weights: [docs/RUBRIC.md](docs/RUBRIC.md).

## 2. Getting started

```bash
# 1. Clone YOUR fork/copy of this repo
git clone <your-repo-url>.git
cd advanced-generative-art-studio

# 2. Create an environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies (CPU-only PyTorch by default; see docs/SETUP.md for GPU/Colab)
pip install -r requirements.txt

# 4. Run the test suite locally — this is exactly what CI runs on push
pytest -v

# 5. See your weighted grade breakdown locally
python scripts/grade.py
```

The tests use tiny synthetic tensors/images (see `tests/conftest.py`), so the full
suite runs on a laptop CPU in well under a minute — you do **not** need the real
CelebA dataset to pass tests or get CI credit. You *do* need real data for the
notebooks, portfolio, and platform deliverables — see [docs/SETUP.md](docs/SETUP.md)
and `scripts/download_data.py`.

## 3. How auto-grading works

1. Every `git push` (or PR) triggers [`.github/workflows/grading.yml`](.github/workflows/grading.yml).
2. It installs dependencies and runs `pytest` grouped by **rubric category markers**
   (`vae`, `gan`, `advanced`, `evaluation`, `platform`).
3. `scripts/grade.py` turns the pytest JSON report into a weighted score matching
   [docs/RUBRIC.md](docs/RUBRIC.md) and writes it to the run's **Job Summary** tab
   (Actions → your run → Summary) plus a downloadable `grade_report.json` artifact.
4. The workflow never fails your push outright (partial credit is expected mid-course) —
   check the Summary tab after every push to see your live score.
5. **Portfolio & Presentation (5%)** and parts of manual review are not auto-gradable
   and are scored by your instructor from your submitted deliverables.

## 4. Repository layout

```
├── src/generative_art_studio/   # the installable package you implement
│   ├── data/                    # dataset loading (real + synthetic-for-CI)
│   ├── models/
│   │   ├── autoencoders/        # vanilla AE, denoising AE, VAE
│   │   ├── gans/                # vanilla GAN, DCGAN, cGAN, WGAN(-GP)
│   │   └── advanced/            # Pix2Pix, CycleGAN
│   ├── training/                # loss functions + training loops
│   ├── evaluation/               # FID / Inception Score
│   ├── utils/                   # seeding, visualization, latent-space tools
│   └── app/                     # Phase 4 Streamlit generative-art platform
├── notebooks/                   # the 4 required deliverable notebooks (starter cells)
├── tests/                       # the auto-grading test suite (do not edit)
├── scripts/                     # dataset download + local grading helper
├── docs/                        # brief, rubric, setup guide, report template
├── data/                        # put downloaded datasets here (git-ignored)
└── outputs/ , checkpoints/      # generated art & model checkpoints (git-ignored)
```

## 5. Deliverables checklist

- [ ] 4 Jupyter notebooks completed (`notebooks/`)
- [ ] Generative Art Platform running (`streamlit run src/generative_art_studio/app/streamlit_app.py`)
- [ ] Technical report, 15–18 pages ([template](docs/TECHNICAL_REPORT_TEMPLATE.md))
- [ ] Generated art portfolio, 50+ images (`outputs/portfolio/`)
- [ ] 10–12 minute video presentation
- [ ] All CI tests passing / green Actions run on your final push

## 6. Getting help

- Read the docstrings and `# TODO` comments in each module — they explain exactly
  what's expected and link back to the relevant learning objective.
- Run a single phase's tests while you work, e.g. `pytest -m vae -v`.
- See [docs/SETUP.md](docs/SETUP.md) for GPU/Colab setup and dataset download steps.

Good luck, and have fun making art with math. 🖼️
