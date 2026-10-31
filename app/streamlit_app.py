"""Phase 8: local Streamlit GUI for the dual-head model (CPU).

  streamlit run app/streamlit_app.py
Model: data_local/main/best.pt (or env BRATS_DUAL_CKPT); bundled cases: data_local/cases/*.npz
(or env BRATS_DUAL_CASES). Uploads must already be BraTS-like 2 mm arrays from
src/preprocess_one.py (skull-stripped, SRI24 grid, cropped, 2x pooled, shared SynthStrip boundary).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import infer  # noqa: E402

CKPT = Path(os.environ.get("BRATS_DUAL_CKPT", ROOT / "data_local/main/best.pt"))
CASES = Path(os.environ.get("BRATS_DUAL_CASES", ROOT / "data_local/cases"))
COLOURS = {"WT": (1.0, 0.85, 0.0), "TC": (1.0, 0.3, 0.0), "ET": (0.9, 0.0, 0.2)}  # drawn WT, TC, ET
CH = {"T1": 0, "T1ce": 1, "T2": 2, "FLAIR": 3}

st.set_page_config(page_title="Brain tumour detection and segmentation", layout="wide")
st.error("Research prototype, not for clinical use. Results are not validated for diagnosis.")
st.title("Brain tumour detection and segmentation (3D dual-head U-Net)")


@st.cache_resource
def get_model(path):
    return infer.load(path)


if not CKPT.exists():
    st.warning(f"No model checkpoint at {CKPT}. Download `main/best.pt` into data_local/main/ first.")
    st.stop()
model, state, thr = get_model(str(CKPT))
st.caption(f"Model: {state['config'].get('name')} (epoch {state['epoch']}), detection threshold {thr:.3f} "
           f"(frozen on validation), shared boundary border {state['config'].get('border')} mm.")

with st.sidebar:
    st.header("Case")
    bundled = sorted(CASES.glob("*.npz")) if CASES.exists() else []
    source = st.radio("Source", ["Bundled case", "Upload preprocessed array"], index=0 if bundled else 1)
    case_path = None
    if source == "Bundled case":
        if bundled:
            case_path = st.selectbox("Bundled case", bundled, format_func=lambda p: p.stem)
        else:
            st.info("No bundled cases in data_local/cases.")
    else:
        st.caption("Upload a .npz from src/preprocess_one.py or a (4, 96, 112, 80) .npy in order "
                   "[T1, T1ce, T2, FLAIR]. Raw scans must be converted first.")
        up = st.file_uploader("Preprocessed case", type=["npz", "npy"])
        if up is not None:
            tmp = Path(tempfile.gettempdir()) / f"brats_dual_upload{Path(up.name).suffix}"
            tmp.write_bytes(up.getvalue())
            case_path = tmp
    st.header("Display")
    cam_mode = st.selectbox("Grad-CAM++ overlay", ["off", "detection", "segmentation"])
    view = st.selectbox("View", ["axial", "coronal", "sagittal"])
    show_gt = st.checkbox("Show ground-truth contour (if available)", value=False)

if case_path is None:
    st.stop()
try:
    x, y, meta = infer.load_case(case_path)
    res = infer.analyse(model, x.astype(np.float32), thr, None if cam_mode == "off" else cam_mode)
except AssertionError as e:
    st.error(f"Input rejected: {e}")
    st.stop()
if meta and not meta.get("shared_boundary", True):
    st.warning("This case was not passed through the shared SynthStrip boundary; results may not be comparable.")

c1, c2, c3 = st.columns(3)
c1.metric("Tumour probability (T1 + T2)", f"{res['prob']:.3f}")
c2.metric("Decision", "Tumour" if res["tumour"] else "No tumour")
c3.metric("Inference time (CPU)", f"{res['ms']:.0f} ms")
if res["can_segment"]:
    v = res["volumes_ml"]
    st.write("Segmented volumes: " + ", ".join(f"**{r}** {v[r]:.1f} mL" for r in ("WT", "TC", "ET")))
else:
    st.info("T1ce and/or FLAIR missing: detection only, no segmentation.")
if cam_mode == "segmentation" and not res["can_segment"]:
    st.info("Segmentation Grad-CAM++ needs all four modalities.")

axis = {"axial": 3, "coronal": 2, "sagittal": 1}[view]
n = x.shape[axis]
default = n // 2
if res["seg"] is not None and res["seg"][2].any():
    default = int(np.round(np.nonzero(res["seg"][2])[axis - 1].mean()))
k = st.slider("Slice", 0, n - 1, default)
present = [m for m in CH if x[CH[m]].any()]
mod = st.radio("Background modality", present, horizontal=True)


def cut(a):
    return np.take(a, k, axis=axis - 1).T


import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
for ax in axes:
    ax.imshow(cut(x[CH[mod]].astype(np.float32)), origin="lower", cmap="gray", vmin=-2.5, vmax=3)
    ax.axis("off")
axes[0].set_title(f"{mod}, {view} slice {k}")
if res["seg"] is not None:
    rgba = np.zeros(cut(res["seg"][0]).shape + (4,))
    for r in ("WT", "TC", "ET"):
        m = cut(res["seg"][("ET", "TC", "WT").index(r)])
        rgba[m] = (*COLOURS[r], 0.45)
    axes[1].imshow(rgba, origin="lower")
    axes[1].set_title("Segmentation: WT yellow, TC orange, ET red")
if res["cam"] is not None:
    axes[1].imshow(np.ma.masked_less(cut(res["cam"]), 0.05), origin="lower", cmap="jet", alpha=0.4, vmin=0, vmax=1)
    axes[1].set_title(f"Grad-CAM++ ({cam_mode})")
if show_gt and y.any():
    for ax in axes:
        ax.contour(cut((y & 4) > 0).astype(float), levels=[0.5], colors="c", linewidths=0.8)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.caption("Voxels are 2 mm (8 mm³). Inputs must already be BraTS-like (skull-stripped, SRI24-registered, "
           "cropped and 2x pooled); use src/preprocess_one.py for raw BraTS-format cases.")
