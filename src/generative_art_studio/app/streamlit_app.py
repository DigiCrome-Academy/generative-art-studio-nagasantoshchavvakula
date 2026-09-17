"""Advanced Generative Art Studio — Phase 4 unified platform.

Run with:
    streamlit run src/generative_art_studio/app/streamlit_app.py

This UI is fully wired up against `model_registry.py`. It will work end to
end once you've implemented `generate_samples` / `export_image` there (and
the underlying model TODOs from Phases 1-3) — until then, clicking
"Generate" will show the NotImplementedError as an on-page error, which is
expected mid-course.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import torch

# Allow running via `streamlit run .../streamlit_app.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from generative_art_studio.app.model_registry import (  # noqa: E402
    Gallery,
    export_image,
    generate_samples,
    list_available_models,
    load_model,
)
from generative_art_studio.utils.viz import make_image_grid  # noqa: E402

st.set_page_config(page_title="Generative Art Studio", page_icon="🎨", layout="wide")

if "gallery" not in st.session_state:
    st.session_state.gallery = Gallery()

st.title("🎨 Advanced Generative Art Studio")
st.caption("A unified playground for your VAE, DCGAN, Conditional GAN, and style-transfer models.")

with st.sidebar:
    st.header("Controls")
    model_key = st.selectbox("Model", options=list_available_models())
    num_samples = st.slider("Number of samples", min_value=1, max_value=16, value=8)
    seed = st.number_input("Seed (for reproducible generation)", min_value=0, value=42, step=1)
    # checkpoint_path = st.text_input("Checkpoint path (optional)", value="")
    checkpoint_options = {
    "vae": "checkpoints/vae_quality.pt",
    "vanilla_gan": "checkpoints/vanilla_gan_quality.pt",
    "dcgan": "checkpoints/dcgan_quality.pt",
    "wgan_gp": "checkpoints/wgan_gp_quality.pt",
    "cyclegan": "checkpoints/cyclegan_a2b_quality.pt",
    }

    checkpoint_path = checkpoint_options[model_key]

    st.text_input(
        "Checkpoint",
        value=Path(checkpoint_path).name,
        key=f"checkpoint_display_{model_key}",
    )
        
    generate_clicked = st.button("✨ Generate", use_container_width=True)
    st.divider()
    st.subheader("Interpolation")
    interpolate_clicked = st.button("🔀 Interpolate between two seeds", use_container_width=True)
    seed_a = st.number_input("Seed A", min_value=0, value=1, step=1)
    seed_b = st.number_input("Seed B", min_value=0, value=2, step=1)
    interp_steps = st.slider("Interpolation steps", min_value=3, max_value=20, value=8)

tab_generate, tab_gallery, tab_export = st.tabs(["Generate", "Gallery", "Export"])

with tab_generate:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_key, checkpoint_path or None, device=device)

    if generate_clicked:
        try:
            images = generate_samples(model_key, model, num_samples, seed=int(seed), device=device)
            grid = make_image_grid(images, nrow=min(num_samples, 4))
            st.image(grid.permute(1, 2, 0).cpu().numpy(), caption=f"{model_key} — seed {seed}", use_container_width=True)
            for img in images:
                st.session_state.gallery.add(img, model_key, seed=int(seed))
        except NotImplementedError as exc:
            st.error(f"Not implemented yet: {exc}")

    if interpolate_clicked:
        st.info(
            "Wire this button up to `utils.latent_space.interpolate_latent` once you've "
            "implemented it, decoding each interpolated latent vector with your model's "
            "decoder/generator to show a smooth transition grid — see Phase 1's latent "
            "space objective in docs/PROJECT_BRIEF.md."
        )

with tab_gallery:
    st.subheader(f"Gallery ({len(st.session_state.gallery)} pieces)")
    if len(st.session_state.gallery) == 0:
        st.write("Nothing generated yet — head to the **Generate** tab.")
    else:
        cols = st.columns(4)
        for i, item in enumerate(st.session_state.gallery):
            with cols[i % 4]:
                img = (item["image"] * 0.5 + 0.5).clamp(0, 1)
                st.image(img.permute(1, 2, 0).numpy(), caption=f"{item['model_key']} / seed {item['seed']}")
        if st.button("Clear gallery"):
            st.session_state.gallery.clear()
            st.rerun()

with tab_export:
    st.subheader("Export a high-resolution PNG")
    if len(st.session_state.gallery) == 0:
        st.write("Generate something first.")
    else:
        idx = st.number_input("Gallery item index", min_value=0, max_value=len(st.session_state.gallery) - 1, value=0)
        scale = st.select_slider("Upscale factor", options=[1, 2, 4, 8], value=4)
        out_path = st.text_input("Save to", value="outputs/portfolio/artwork_001.png")
        if st.button("💾 Export"):
            try:
                item = list(st.session_state.gallery)[idx]
                saved_path = export_image(item["image"], out_path, scale_factor=scale)
                st.success(f"Saved to {saved_path}")
            except NotImplementedError as exc:
                st.error(f"Not implemented yet: {exc}")
