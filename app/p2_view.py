"""PLAN2 view for the Streamlit GUI: per-hemisphere detection (head and segmentation-derived), stitched
segmentation, and explanations of one hemisphere's detection (Grad-CAM++, occlusion, region SHAP bars)."""
from __future__ import annotations

import numpy as np
import streamlit as st

import p2_infer

COLOURS = {"WT": (1.0, 0.85, 0.0), "TC": (1.0, 0.3, 0.0), "ET": (0.9, 0.0, 0.2)}
CH = {"T1": 0, "T1ce": 1, "T2": 2, "FLAIR": 3}


def render(model, state, thr, x, y, case_key, view, show_gt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st.caption(f"Model: PLAN2 `{state['config'].get('name')}` (epoch {state['epoch']}). The scan is split at the "
               f"midline into two hemispheres; each is analysed on its own. Thresholds frozen on validation: "
               f"head probability {thr[0]:.2f}, segmentation-derived tumour volume {thr[1]:.2f} mL. Validation "
               "verdict: segmentation-derived detector shortcut-controlled; detection head passes every "
               "behavioural check but its heat maps fail the localisation check (S7).")
    with st.sidebar:
        st.header("Explanation (PLAN2)")
        side = st.radio("Hemisphere to explain", ["L", "R"], format_func=lambda s: p2_infer.SIDE_NAME[s])
        method = st.selectbox("Heat map", ["off", "Grad-CAM++ (fast)", "occlusion preview (16 mm stride)",
                                           "occlusion (8 mm stride, as evaluated)"])
        do_shap = st.checkbox("Region SHAP bar chart (slow on CPU)", value=False)
        perms = st.slider("SHAP permutations (evaluation used about 8)", 1, 8, 2, disabled=not do_shap)

    try:
        res = analyse(model, x, thr, case_key)
    except AssertionError as e:
        st.error(f"Input rejected: {e}")
        st.stop()

    c = st.columns(4)
    c[0].metric("Decision (head, any hemisphere)", "Tumour" if res["tumour_head"] else "No tumour")
    c[1].metric("Decision (segmentation-derived)", "Tumour" if res["tumour_seg"] else "No tumour")
    c[2].metric("Inference time (CPU, both hemispheres)", f"{res['ms']:.0f} ms")
    c[3].metric("Midline index (axis 0)", f"{p2_infer.MIDLINE}")
    rows = []
    for s, r in res["sides"].items():
        rows.append({"hemisphere": p2_infer.SIDE_NAME[s], "tumour probability (T1+T2)": round(r["prob"], 3),
                     "head decision": "tumour" if r["head_tumour"] else "no tumour",
                     "pass-B WT volume (mL)": round(r["wt_ml_b"], 1),
                     "segmentation-derived decision": "tumour" if r["seg_tumour"] else "no tumour",
                     **({f"{k} (mL, 4-seq)": round(v, 1) for k, v in r["volumes_ml"].items()}
                        if "volumes_ml" in r else {})})
    st.table(rows)
    if not res["can_segment"]:
        st.info("T1ce and/or FLAIR missing: ET/TC/WT segmentation (pass A) is off; the WT outline shown is pass B "
                "(T1 + T2).")

    hm = None
    h = res["sides"][side]["x"]
    if method.startswith("Grad"):
        hm = cached("cam", case_key, side, lambda: p2_infer.gradcam(model, h))
    elif method.startswith("occlusion"):
        stride = 8 if "8 mm" in method else 16
        with st.spinner(f"Occlusion, {stride} mm stride ({'~10-15 min' if stride == 8 else '~1-2 min'} on CPU)"):
            hm = cached(f"occ{stride}", case_key, side, lambda: p2_infer.occlusion(model, h, stride=stride // 2))
    if hm is not None:
        hm = p2_infer.to_scan(hm, side, x.shape[1])

    axis = {"axial": 3, "coronal": 2, "sagittal": 1}[view]
    n = x.shape[axis]
    ref = res["seg"][2] if res["seg"] is not None else res["wt_b"]
    default = int(np.round(np.nonzero(ref)[axis - 1].mean())) if ref.any() else n // 2
    k = st.slider("Slice", 0, n - 1, default, key="p2_slice")
    present = [m for m in CH if x[CH[m]].any()]
    mod = st.radio("Background modality", present, horizontal=True, key="p2_mod")

    def cut(a):
        return np.take(a, k, axis=axis - 1).T

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    for ax in axes:
        ax.imshow(cut(x[CH[mod]].astype(np.float32)), origin="lower", cmap="gray", vmin=-2.5, vmax=3)
        ax.axis("off")
    axes[0].set_title(f"{mod}, {view} slice {k}")
    if view != "sagittal":
        for ax in axes:
            ax.axvline(p2_infer.MIDLINE - 0.5, color="w", lw=0.6, ls="--")
    if res["seg"] is not None:
        rgba = np.zeros(cut(res["seg"][0]).shape + (4,))
        for r in ("WT", "TC", "ET"):
            rgba[cut(res["seg"][("ET", "TC", "WT").index(r)])] = (*COLOURS[r], 0.45)
        axes[1].imshow(rgba, origin="lower")
        title = "Segmentation: WT yellow, TC orange, ET red"
    else:
        rgba = np.zeros(cut(res["wt_b"]).shape + (4,))
        rgba[cut(res["wt_b"])] = (*COLOURS["WT"], 0.45)
        axes[1].imshow(rgba, origin="lower")
        title = "Pass-B WT (T1 + T2), yellow"
    if hm is not None:
        axes[1].imshow(np.ma.masked_less(cut(hm), 0.05), origin="lower", cmap="jet", alpha=0.4, vmin=0, vmax=1)
        title = f"{method.split(' (')[0]}, {side} hemisphere"
    axes[1].set_title(title)
    if show_gt and y is not None and y.any():
        for ax in axes:
            ax.contour(cut((y & 4) > 0).astype(float), levels=[0.5], colors="c", linewidths=0.8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Dashed white line: the midline cut (axial/coronal views). Heat maps show the selected hemisphere "
               "only. Validation found that Grad-CAM++ hardly changes when the detection head is randomised "
               "(sanity check), so occlusion and SHAP are the more trustworthy explanations here.")

    if do_shap:
        with st.spinner(f"Region SHAP, {perms} permutation(s): about {4 * perms} min on CPU"):
            out = cached(f"shap{perms}", case_key, side, lambda: p2_infer.shap_regions(
                model, h, res["sides"][side]["wt_b"], perms=perms))
        shap_rows = out[0]
        top = shap_rows[:12]
        fig, ax = plt.subplots(figsize=(8, 0.35 * len(top) + 1))
        names = [t[0] for t in top][::-1]
        vals = [t[1] for t in top][::-1]
        ax.barh(names, vals, color=["#c0392b" if v > 0 else "#2471a3" for v in vals])
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel("Shapley value on the detection logit (red raises, blue lowers the tumour score)")
        ax.set_title(f"Region SHAP, {p2_infer.SIDE_NAME[side]} hemisphere: 12 largest regions")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        tum = sum(t[1] for t in shap_rows if t[0].startswith("predicted tumour"))
        out_b = sum(t[1] for t in shap_rows if t[0] == "outside brain")
        rest = sum(t[1] for t in shap_rows) - tum - out_b
        st.write(f"Totals: predicted tumour **{tum:+.2f}**, rest of brain **{rest:+.2f}**, outside brain "
                 f"**{out_b:+.2f}** logits. Absent regions are replaced by the blurred image (sigma 8 mm).")


def analyse(model, x, thr, case_key):
    key = ("res", case_key)
    if st.session_state.get("p2_key") != key:
        st.session_state["p2_key"] = key
        st.session_state["p2_cache"] = {"res": p2_infer.analyse(model, x.astype(np.float32), thr)}
    return st.session_state["p2_cache"]["res"]


def cached(name, case_key, side, fn):
    c = st.session_state["p2_cache"]
    if (name, side) not in c:
        c[(name, side)] = fn()
    return c[(name, side)]
