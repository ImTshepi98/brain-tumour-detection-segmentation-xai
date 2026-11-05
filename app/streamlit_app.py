"""Local Streamlit GUI for brain tumour detection, segmentation and explanation (CPU).

  streamlit run app/streamlit_app.py
Models: Hemisphere model data_local/p2_main/best.pt (or env BRATS_DUAL_P2_CKPT) with its results_val.json for
the frozen thresholds; Whole-scan model data_local/main/best.pt (or env BRATS_DUAL_CKPT). Example cases:
data_local/cases/*.npz (or env BRATS_DUAL_CASES). Uploads must already be BraTS-like 2 mm arrays from
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
import p2_infer  # noqa: E402

sys.path.insert(0, str(ROOT / "app"))
import hemisphere_view  # noqa: E402
import ui  # noqa: E402

CKPT = Path(os.environ.get("BRATS_DUAL_CKPT", ROOT / "data_local/main/best.pt"))
HEMI_CKPT = Path(os.environ.get("BRATS_DUAL_P2_CKPT", ROOT / "data_local/p2_main/best.pt"))
CASES = Path(os.environ.get("BRATS_DUAL_CASES", ROOT / "data_local/cases"))
MODELS = {
    "Hemisphere model": "Recommended. Analyses each hemisphere; validated against dataset shortcuts.",
    "Whole-scan model": "Original model. Accurate segmentation; detection partly relies on the dataset.",
}
OVERLAYS = ["Segmentation", "Detection heat map", "Segmentation heat map"]

ui.setup()


@st.cache_resource(show_spinner="Loading the whole-scan model...")
def get_model(path):
    return infer.load(path)


@st.cache_resource(show_spinner="Loading the hemisphere model...")
def get_hemi_model(path):
    return p2_infer.load(path)


def case_label(p):
    kind = "healthy volunteer" if p.stem.upper().startswith("IXI") else "tumour case"
    return f"{p.stem}  ·  {kind}"


with st.sidebar:
    ui.sidebar_brand()
    ui.step(1, "Model")
    which = st.radio("Model", list(MODELS), captions=list(MODELS.values()), key="model",
                     label_visibility="collapsed")

    ui.step(2, "Scan")
    bundled = sorted(CASES.glob("*.npz")) if CASES.exists() else []
    source = st.radio("Source", ["Example case", "Upload"], index=0 if bundled else 1, horizontal=True,
                      key="source", label_visibility="collapsed")
    case_path = None
    if source == "Example case":
        if bundled:
            case_path = st.selectbox("Example case", bundled, format_func=case_label, key="case")
        else:
            st.info("No example cases found in data_local/cases.")
    else:
        up = st.file_uploader("Preprocessed scan (.npz or .npy)", type=["npz", "npy"],
                              help="A .npz from src/preprocess_one.py, or a (4, 96, 112, 80) .npy with the "
                                   "sequences in the order T1, T1ce, T2, FLAIR. Raw scans must be converted first.")
        if up is not None:
            tmp = Path(tempfile.gettempdir()) / f"brats_dual_upload{Path(up.name).suffix}"
            tmp.write_bytes(up.getvalue())
            case_path = tmp

    ui.step(3, "Display")
    view = st.segmented_control("Orientation", list(ui.AXIS), default="Axial", key="view") or "Axial"
    show_gt = st.toggle("Show expert outline", key="show_gt",
                        help="Draws the reference whole-tumour outline when the case includes one.")
    ui.sidebar_footer()

ui.header()

hemi = which == "Hemisphere model"
ck = HEMI_CKPT if hemi else CKPT
if not ck.exists():
    st.markdown(f'<div class="empty">No model file found at <code>{ck}</code>.<br>Copy the trained checkpoint '
                'there and reload the page.</div>', unsafe_allow_html=True)
    st.stop()
if case_path is None:
    st.markdown('<div class="empty">Select an example case or upload a preprocessed scan in the sidebar to '
                'begin.</div>', unsafe_allow_html=True)
    st.stop()
try:
    x, y, meta = infer.load_case(case_path)
except AssertionError as e:
    st.error(f"This file could not be read as a preprocessed scan: {e}")
    st.stop()
if meta and not meta.get("shared_boundary", True):
    st.warning("This scan was not passed through the shared skull-strip boundary, so results may not be "
               "comparable with the evaluation.", icon=":material/warning:")

if hemi:
    model, state, thr = get_hemi_model(str(ck))
    hemisphere_view.render(model, state, thr, x, y, (str(case_path), Path(case_path).stat().st_mtime), view,
                           show_gt)
    st.stop()

# ------------------------------------------------------------------ Whole-scan model
model, state, thr = get_model(str(ck))
overlay = st.session_state.get("ws_overlay", OVERLAYS[0])
cam_mode = {"Detection heat map": "detection", "Segmentation heat map": "segmentation"}.get(overlay)
try:
    with st.spinner("Analysing the scan..."):
        res = infer.analyse(model, x.astype(np.float32), thr, cam_mode)
except AssertionError as e:
    st.error(f"This input could not be analysed: {e}")
    st.stop()

c = st.columns([1, 1, 1.15], gap="medium")
with c[0]:
    ui.card("Finding", ui.badge(res["tumour"]) + ui.prob_bar(res["prob"], "Tumour probability (T1 + T2)") +
            ui.kv([("Decision threshold", f"{thr:.3f}"), ("Analysis time", f"{res['ms'] / 1000:.1f} s on CPU")]))
with c[1]:
    if res["can_segment"]:
        v = res["volumes_ml"]
        body = '<p class="main">' + f'{v["WT"]:.1f} mL</p><p class="muted">whole tumour</p>' + \
            ui.kv([(ui.REGION_NAME[r], f"{v[r]:.1f} mL") for r in ("TC", "ET")])
    else:
        body = '<p class="muted">T1ce and/or FLAIR are missing, so this model can detect but not segment.</p>'
    ui.card("Segmented volumes", body)
with c[2]:
    ui.card("Interpret with care",
            '<span class="badge info">Known limitation</span><p class="muted" style="margin-top:10px">This model '
            "detected every test scan correctly, but controls showed that its detection decision relies partly "
            "on recognising the source dataset rather than the tumour. Its segmentation is not affected. For "
            "detection, prefer the Hemisphere model.</p>")
st.write("")

tab_view, tab_info = st.tabs([":material/visibility: Scan viewer", ":material/info: Model details"])
with tab_view:
    st.radio("Overlay", OVERLAYS, horizontal=True, key="ws_overlay")
    if cam_mode == "segmentation" and not res["can_segment"]:
        st.info("The segmentation heat map needs all four sequences.", icon=":material/info:")
    ref = res["seg"][2] if res["seg"] is not None else None
    k, mod = ui.slice_controls(x, view, ref, "ws")
    gt = (y & 4) > 0 if show_gt and y is not None and y.any() else None
    seg = ui.seg_dict(res["seg"]) if res["seg"] is not None and cam_mode is None else None
    ui.figure(x, view, k, mod, seg=seg, heat=res["cam"], gt=gt, right_title=overlay)
    items = ui.seg_legend() if seg else [("box", "linear-gradient(90deg,#30123b,#28bceb,#a4fc3c,#fb8022,#7a0403)",
                                          "Low to high Grad-CAM++ attention")] if res["cam"] is not None else []
    ui.legend(items + ([("ring", "", "Expert outline")] if gt is not None else []))
with tab_info:
    c1, c2 = st.columns([1.6, 1], gap="large")
    with c1:
        st.markdown("**How it works**")
        st.write("A single 3D network with a shared encoder produces two outputs. The detection output receives T1 "
                 "and T2 only and gives a tumour probability for the whole scan. The segmentation output receives "
                 "all four sequences and delineates the enhancing tumour, tumour core and whole tumour. Grad-CAM++ "
                 "heat maps can be shown for either output.")
        st.markdown("**What the evaluation found**")
        st.write("The model segmented tumours accurately and separated tumour from healthy scans perfectly on the "
                 "test set. Counterfactual controls, however, showed that replacing the tumour with healthy tissue "
                 "rarely changed the detection decision, because the two source datasets can be told apart from "
                 "tumour-free tissue alone. The Hemisphere model was developed to address this.")
    with c2:
        ui.card("Configuration", ui.kv([
            ("Checkpoint epoch", str(state["epoch"])),
            ("Decision threshold", f"{thr:.3f}"),
            ("Input", "4 x 96 x 112 x 80 (2 mm)"),
            ("Voxel size", "2 mm (8 mm³)"),
            ("Analysis time", f"{res['ms'] / 1000:.1f} s on CPU")]))
