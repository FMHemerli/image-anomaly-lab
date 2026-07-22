"""Interactive defect inspector -- upload a part, see where the model looks.

Run with:
    streamlit run app/streamlit_app.py

It loads a memory bank already fitted by scripts/fit_memory_bank.py (so no training
happens in the app), runs an uploaded image through the same frozen backbone, and
shows the anomaly heatmap, the image-level score, and a pass/fail verdict against
the saved threshold. This is the portfolio-facing face of the whole pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from image_anomaly_lab import BackboneConfig, DataConfig, resolve_torch_device
from image_anomaly_lab.backbones import PatchFeatureExtractor
from image_anomaly_lab.data import build_transform
from image_anomaly_lab.detectors.memory_bank import (
    MemoryBank,
    image_score_from_map,
    patch_scores_to_map,
)

ROOT = Path(__file__).resolve().parents[1]
BANKS = ROOT / "memory_bank"
OUTPUTS = ROOT / "outputs"

st.set_page_config(page_title="Defect inspector", page_icon=":material/search_insights:", layout="wide")


# --- Cached resources: load the heavy pieces once, reuse across reruns ----------


@st.cache_resource(show_spinner="Loading backbone...")
def load_extractor(backbone_name: str, layers: tuple[str, ...]):
    """The frozen CNN is expensive to build -- cache it as a shared resource."""
    device = resolve_torch_device()
    return PatchFeatureExtractor(BackboneConfig(name=backbone_name, layers=layers), device)


@st.cache_resource(show_spinner="Loading memory bank...")
def load_bank(category: str) -> MemoryBank:
    return MemoryBank.load(BANKS / f"{category}.npz")


@st.cache_data
def saved_threshold(category: str) -> float | None:
    """Read the threshold chosen at fit time, if the run JSON is present."""
    path = OUTPUTS / f"{category}_patchcore.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["metrics"].get("threshold")


def available_categories() -> list[str]:
    return sorted(p.stem for p in BANKS.glob("*.npz")) if BANKS.exists() else []


def overlay_heatmap(image: Image.Image, anomaly_map: np.ndarray, vmax: float) -> Image.Image:
    """Blend a jet-coloured, normalised heatmap over the original part image."""
    from matplotlib import cm

    norm = np.clip(anomaly_map / vmax, 0.0, 1.0) if vmax > 0 else anomaly_map * 0.0
    heat = (cm.jet(norm)[:, :, :3] * 255).astype(np.uint8)
    heat_img = Image.fromarray(heat).resize(image.size)
    return Image.blend(image.convert("RGB"), heat_img, alpha=0.5)


# --- Fast UI first (title, controls); slow inference happens only on upload ------

st.title(":material/search_insights: Visual defect inspector")
st.caption("Trained only on good parts. It flags -- and localises -- whatever departs from normal.")

categories = available_categories()
if not categories:
    st.warning(
        "No fitted memory bank found in `memory_bank/`. Fit one first:\n\n"
        "```\npython scripts/download_data.py --category metal_nut\n"
        "python scripts/fit_memory_bank.py --category metal_nut\n```",
        icon=":material/info:",
    )
    st.stop()

with st.sidebar:
    st.header("Settings")
    category = st.selectbox("Part category", categories)
    default_thr = saved_threshold(category)
    st.caption(
        f"Saved threshold: {default_thr:.4g}" if default_thr is not None
        else "No saved threshold -- set one below."
    )
    threshold = st.number_input(
        "Decision threshold",
        value=float(default_thr) if default_thr is not None else 1.0,
        help="Image score above this is called a defect. Comes from Youden's J at fit time.",
    )

bank = load_bank(category)
extractor = load_extractor("wide_resnet50_2", ("layer2", "layer3"))
transform = build_transform(DataConfig(category=category))

uploaded = st.file_uploader(
    "Upload a part image", type=["png", "jpg", "jpeg"], help="A photo of the same part category."
)

if uploaded is None:
    st.info("Upload an image to inspect.", icon=":material/upload:")
    st.stop()

# --- Slow work last: run the upload through the pipeline ------------------------

image = Image.open(uploaded).convert("RGB")
image_size = 256

with st.spinner("Scoring..."):
    import torch

    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        feats = extractor(tensor)
        vectors, grid_hw = extractor.to_patch_vectors(feats)
    scores = bank.patch_scores(vectors.cpu().numpy())
    anomaly_map = patch_scores_to_map(scores, grid_hw, image_size, bank.cfg.blur_sigma)
    score = image_score_from_map(anomaly_map)

is_defect = score > threshold
display_image = image.resize((image_size, image_size))

with st.container(horizontal=True):
    st.metric("Anomaly score", f"{score:.3f}", border=True)
    st.metric(
        "Verdict",
        "defect" if is_defect else "pass",
        delta="fail" if is_defect else "ok",
        delta_color="inverse",
        border=True,
    )
    st.metric("Threshold", f"{threshold:.3g}", border=True)

col1, col2 = st.columns(2)
with col1:
    with st.container(border=True):
        st.markdown("**Uploaded part**")
        st.image(display_image, width="stretch")
with col2:
    with st.container(border=True):
        st.markdown("**Anomaly heatmap**")
        st.image(overlay_heatmap(display_image, anomaly_map, vmax=max(score, 1e-6)), width="stretch")

st.caption(
    "Red marks where local appearance is furthest from anything seen in good parts. "
    "The score is the map's peak; the verdict compares it to the threshold."
)
