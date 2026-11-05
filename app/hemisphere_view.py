"""Hemisphere model view: per-hemisphere decisions, stitched segmentation, explanations of one hemisphere's
classifier decision (Grad-CAM++, occlusion, region SHAP) and model details."""
from __future__ import annotations

import numpy as np
import streamlit as st

import p2_infer
import ui

SIDE = {"L": "Left hemisphere", "R": "Right hemisphere"}
METHODS = {"Grad-CAM++": ("cam", None, "Instant. Gradient-based; failed the randomisation sanity check."),
           "Occlusion (preview)": ("occ16", 8, "About 1-2 min on CPU. Coarser 16 mm steps."),
           "Occlusion (full)": ("occ8", 4, "About 10-15 min on CPU. 8 mm steps, as evaluated.")}


def render(model, state, thr, x, y, case_key, view, show_gt):
    try:
        res = analyse(model, x, thr, case_key)
    except AssertionError as e:
        st.error(f"This input could not be analysed: {e}")
        st.stop()
    gt = (y & 4) > 0 if show_gt and y is not None and y.any() else None
    _summary(res, thr)

    tab_view, tab_xai, tab_info = st.tabs([":material/visibility: Scan viewer", ":material/insights: Explanation",
                                           ":material/info: Model details"])
    with tab_view:
        _viewer(res, x, view, gt)
    with tab_xai:
        _explanation(model, res, x, view, case_key)
    with tab_info:
        _details(state, thr, res)


def _summary(res, thr):
    agree = res["tumour_head"] == res["tumour_seg"]
    c = st.columns([1.1, 1, 1], gap="medium")
    with c[0]:
        ui.card("Overall finding",
                ui.badge(res["tumour_seg"]) +
                f'<p class="muted" style="margin:10px 0 6px">From the segmentation-based decision; the classifier '
                f'{"agrees" if agree else "disagrees"}.</p>' +
                ui.kv([("Classifier decision", "Tumour" if res["tumour_head"] else "No tumour"),
                       ("Analysis time", f"{res['ms'] / 1000:.1f} s on CPU")]))
    for col, s in zip(c[1:], ("L", "R")):
        r = res["sides"][s]
        rows = [("Tumour volume (T1/T2 outline)", f"{r['wt_ml_b']:.1f} mL"),
                ("Classifier decision", "Tumour" if r["head_tumour"] else "No tumour")]
        if "volumes_ml" in r:
            rows += [(ui.REGION_NAME[k], f"{r['volumes_ml'][k]:.1f} mL") for k in ("WT", "TC", "ET")]
        with col:
            ui.card(SIDE[s], ui.badge(r["seg_tumour"]) + ui.prob_bar(r["prob"]) + ui.kv(rows))
    st.write("")


def _viewer(res, x, view, gt):
    if not res["can_segment"]:
        st.info("T1ce and/or FLAIR are missing, so the tumour sub-regions cannot be segmented. The overlay shows "
                "the tumour outline predicted from T1 and T2.", icon=":material/info:")
    ref = res["seg"][2] if res["seg"] is not None else res["wt_b"]
    k, mod = ui.slice_controls(x, view, ref, "hemi")
    seg = ui.seg_dict(res["seg"]) if res["seg"] is not None else {"WT": res["wt_b"]}
    ui.figure(x, view, k, mod, seg=seg, gt=gt, midline=p2_infer.MIDLINE, right_title="Segmentation")
    items = ui.seg_legend() if res["seg"] is not None else [("box", "#FFD900", "Tumour outline (T1/T2)")]
    items += [("line", "", "Midline")] + ([("ring", "", "Expert outline")] if gt is not None else [])
    ui.legend(items)


def _explanation(model, res, x, view, case_key):
    left, right = st.columns([1, 2.2], gap="large")
    with left:
        dflt = max(("L", "R"), key=lambda s: res["sides"][s]["prob"])
        side = st.segmented_control("Hemisphere", ["L", "R"], format_func=SIDE.get, default=dflt,
                                    key="xai_side") or dflt
        method = st.radio("Method", list(METHODS), captions=[m[2] for m in METHODS.values()], key="xai_method")
        name, stride, _ = METHODS[method]
        h = res["sides"][side]["x"]
        hm = _get(name, side)
        if hm is None and name == "cam":
            hm = _put(name, side, p2_infer.gradcam(model, h))
        elif hm is None:
            if st.button(f"Compute {method.lower()}", type="primary", icon=":material/play_arrow:",
                         use_container_width=True):
                with st.spinner(f"Computing {method.lower()}..."):
                    hm = _put(name, side, p2_infer.occlusion(model, h, stride=stride))
        st.caption("Occlusion and SHAP passed the randomisation sanity check; Grad-CAM++ did not, so treat its map "
                   "as indicative only. No method was faithful by the deletion test.")
    with right:
        if hm is None:
            st.markdown('<div class="empty">Press <b>Compute</b> to generate this explanation.</div>',
                        unsafe_allow_html=True)
        else:
            ref = res["seg"][2] if res["seg"] is not None else res["wt_b"]
            k, mod = ui.slice_controls(x, view, ref, "xai")
            ui.figure(x, view, k, mod, heat=p2_infer.to_scan(hm, side, x.shape[1]), midline=p2_infer.MIDLINE,
                      right_title=f"{method}  ·  {SIDE[side].lower()}")
            ui.legend([("box", "linear-gradient(90deg,#30123b,#28bceb,#a4fc3c,#fb8022,#7a0403)",
                        "Low to high influence on the classifier"), ("line", "", "Midline")])

    st.divider()
    st.markdown("**Region contributions (SHAP)**")
    st.caption("Shares the classifier's score among regions: each predicted tumour component, 16 mm blocks of the "
               "remaining brain, and the area outside the brain. Absent regions are replaced by a blurred image.")
    c1, c2 = st.columns([2, 1])
    perms = c1.slider("Permutations (more is slower and more stable; evaluation used about 8)", 1, 8, 2,
                      key="shap_perms")
    key = f"shap{perms}"
    out = _get(key, side)
    if out is None and c2.button(f"Compute SHAP (about {4 * perms} min)", icon=":material/play_arrow:",
                                 use_container_width=True):
        with st.spinner("Computing region SHAP..."):
            out = _put(key, side, p2_infer.shap_regions(model, h, res["sides"][side]["wt_b"], perms=perms))
    if out is not None:
        _shap_chart(out[0], side)


def _shap_chart(rows, side):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = rows[:12][::-1]
    fig, ax = plt.subplots(figsize=(9, 0.34 * len(top) + 1.1))
    ax.barh([t[0] for t in top], [t[1] for t in top], color=["#DC2626" if t[1] > 0 else "#2563EB" for t in top])
    ax.axvline(0, color="#334155", lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("Contribution to the classifier score (red raises, blue lowers)")
    ax.set_title(f"{SIDE[side]}: 12 largest regions", loc="left", fontsize=11)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    tum = sum(t[1] for t in rows if t[0].startswith("predicted tumour"))
    out_b = sum(t[1] for t in rows if t[0] == "outside brain")
    rest = sum(t[1] for t in rows) - tum - out_b
    c = st.columns(3)
    c[0].metric("Predicted tumour", f"{tum:+.2f}")
    c[1].metric("Rest of brain", f"{rest:+.2f}")
    c[2].metric("Outside brain", f"{out_b:+.2f}")


def _details(state, thr, res):
    c1, c2 = st.columns([1.6, 1], gap="large")
    with c1:
        st.markdown("**How it works**")
        st.write("The scan is split at the brain midline and each hemisphere is analysed separately by the same "
                 "3D network. Two decisions are reported for each hemisphere. The **segmentation-based decision** "
                 f"calls a hemisphere tumour-positive when the tumour outline predicted from T1 and T2 exceeds "
                 f"{thr[1]:.2f} mL. The **classifier decision** uses a dedicated detection output with a "
                 f"probability threshold of {thr[0]:.2f}. When all four sequences are present, the enhancing "
                 "tumour, tumour core and whole tumour are also segmented.")
        st.markdown("**How it was validated**")
        st.write("The model was trained so that tumour and tumour-free samples come from both source datasets, and "
                 "it was tested against ten checks for dataset shortcuts that were fixed before training. The "
                 "segmentation-based decision passed every required check. The classifier's decisions passed every "
                 "behavioural check, but its heat maps did not concentrate on the tumour; the overall finding "
                 "therefore uses the segmentation-based decision.")
    with c2:
        ui.card("Configuration", ui.kv([
            ("Checkpoint epoch", str(state["epoch"])),
            ("Segmentation threshold", f"{thr[1]:.2f} mL"),
            ("Classifier threshold", f"{thr[0]:.2f}"),
            ("Midline position", f"voxel {p2_infer.MIDLINE}"),
            ("Input", "4 x 96 x 112 x 80 (2 mm)"),
            ("Analysis time", f"{res['ms'] / 1000:.1f} s on CPU")]))


def analyse(model, x, thr, case_key):
    key = ("res", case_key)
    if st.session_state.get("hemi_key") != key:
        st.session_state["hemi_key"] = key
        st.session_state["hemi_cache"] = {"res": p2_infer.analyse(model, x.astype(np.float32), thr)}
    return st.session_state["hemi_cache"]["res"]


def _get(name, side):
    return st.session_state["hemi_cache"].get((name, side))


def _put(name, side, value):
    st.session_state["hemi_cache"][(name, side)] = value
    return value
