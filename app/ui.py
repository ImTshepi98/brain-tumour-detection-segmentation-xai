"""Shared look and feel for the Streamlit GUI: styling, header, sidebar step labels, result cards and the
slice figure. Visible wording avoids internal project names; models are called the Hemisphere model and the
Whole-scan model."""
from __future__ import annotations

import html

import numpy as np
import streamlit as st

NAVY = "#1E3A8A"
COLOURS = {"WT": (1.0, 0.85, 0.0), "TC": (1.0, 0.45, 0.0), "ET": (0.86, 0.08, 0.24)}
REGION_NAME = {"WT": "Whole tumour", "TC": "Tumour core", "ET": "Enhancing tumour"}
CH = {"T1": 0, "T1ce": 1, "T2": 2, "FLAIR": 3}
AXIS = {"Axial": 3, "Coronal": 2, "Sagittal": 1}

_CSS = """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem;}
[data-testid="stSidebar"] {background: #F6F8FC; border-right: 1px solid #E2E8F0;}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {padding-top: 0.6rem;}
.app-header {display: flex; align-items: center; gap: 14px; padding-bottom: 14px;
             border-bottom: 1px solid #E2E8F0; margin-bottom: 14px;}
.app-logo {width: 44px; height: 44px; border-radius: 11px; flex: none;
           background: linear-gradient(135deg, #1E3A8A, #3B82F6); display: flex; align-items: center;
           justify-content: center; color: #fff; font-weight: 700; font-size: 17px; letter-spacing: .02em;}
.app-title {font-size: 1.5rem; font-weight: 700; color: #0F172A; line-height: 1.2; margin: 0;}
.app-sub {color: #64748B; font-size: .92rem; margin: 2px 0 0;}
.notice {background: #FFF7ED; border: 1px solid #FED7AA; color: #9A3412; border-radius: 8px;
         padding: 8px 12px; font-size: .84rem; margin-bottom: 18px;}
.side-brand {font-weight: 700; color: #0F172A; font-size: 1.02rem; margin: 0 0 2px;}
.side-sub {color: #64748B; font-size: .8rem; margin-bottom: 6px;}
.step {display: flex; align-items: center; gap: 8px; font-size: .74rem; font-weight: 700;
       letter-spacing: .08em; text-transform: uppercase; color: #334155; margin: 20px 0 6px;}
.step b {display: inline-flex; width: 20px; height: 20px; border-radius: 50%; background: #1E3A8A; color: #fff;
         align-items: center; justify-content: center; font-size: .7rem; letter-spacing: 0;}
.side-foot {color: #94A3B8; font-size: .75rem; margin-top: 28px; border-top: 1px solid #E2E8F0; padding-top: 10px;}
.card {background: #fff; border: 1px solid #E2E8F0; border-radius: 12px; padding: 16px 18px; height: 100%;
       box-shadow: 0 1px 2px rgba(15, 23, 42, .05);}
.card .label {margin: 0 0 8px; font-size: .74rem; color: #64748B; font-weight: 700; text-transform: uppercase;
              letter-spacing: .06em;}
.card .main {font-size: 1.25rem; font-weight: 700; color: #0F172A; margin: 6px 0 2px;}
.badge {display: inline-flex; align-items: center; gap: 6px; padding: 4px 11px; border-radius: 999px;
        font-weight: 600; font-size: .86rem;}
.badge::before {content: ""; width: 7px; height: 7px; border-radius: 50%; background: currentColor;}
.badge.pos {background: #FEE2E2; color: #B91C1C;}
.badge.neg {background: #DCFCE7; color: #15803D;}
.badge.info {background: #E0E7FF; color: #3730A3;}
.bar {height: 6px; background: #E2E8F0; border-radius: 3px; margin: 10px 0 2px; overflow: hidden;}
.bar > span {display: block; height: 100%; background: #1E3A8A; border-radius: 3px;}
.kv {display: flex; justify-content: space-between; font-size: .87rem; color: #334155; padding: 5px 0;
     border-top: 1px solid #F1F5F9;}
.kv span:last-child {font-weight: 600; color: #0F172A; font-variant-numeric: tabular-nums;}
.muted {color: #64748B; font-size: .82rem;}
.legend {display: flex; flex-wrap: wrap; gap: 6px 16px; margin: 6px 0 4px;}
.legend span {display: inline-flex; align-items: center; gap: 6px; font-size: .84rem; color: #334155;}
.legend i {width: 12px; height: 12px; border-radius: 3px; display: inline-block;}
.legend i.line {height: 0; border-top: 2px dashed #94A3B8; border-radius: 0; width: 16px;}
.legend i.ring {background: transparent; border: 2px solid #22D3EE;}
.empty {border: 1px dashed #CBD5E1; border-radius: 12px; padding: 40px; text-align: center; color: #64748B;}
</style>
"""


def setup():
    st.set_page_config(page_title="Brain Tumour Analysis", page_icon=":material/neurology:", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(_CSS, unsafe_allow_html=True)


def header():
    st.markdown(
        '<div class="app-header"><div class="app-logo">BT</div><div>'
        '<p class="app-title">Brain Tumour Analysis</p>'
        '<p class="app-sub">3D detection, segmentation and explanation of brain MRI</p></div></div>'
        '<div class="notice"><b>Research prototype.</b> Not for clinical use; results are not validated for '
        'diagnosis.</div>', unsafe_allow_html=True)


def sidebar_brand():
    st.markdown('<p class="side-brand">Analysis settings</p>'
                '<p class="side-sub">Choose a model and a scan, then explore the results.</p>',
                unsafe_allow_html=True)


def step(n, label):
    st.markdown(f'<div class="step"><b>{n}</b>{html.escape(label)}</div>', unsafe_allow_html=True)


def sidebar_footer():
    st.markdown('<div class="side-foot">Research prototype. Inputs must be preprocessed 2 mm arrays in the '
                'BraTS space.</div>', unsafe_allow_html=True)


def badge(positive, pos="Tumour detected", neg="No tumour detected"):
    return f'<span class="badge {"pos" if positive else "neg"}">{pos if positive else neg}</span>'


def card(label, body):
    st.markdown(f'<div class="card"><p class="label">{html.escape(label)}</p>{body}</div>', unsafe_allow_html=True)


def kv(rows):
    return "".join(f'<div class="kv"><span>{html.escape(k)}</span><span>{html.escape(v)}</span></div>'
                   for k, v in rows)


def prob_bar(p, label="Classifier probability"):
    return (f'<div class="muted" style="display:flex;justify-content:space-between;margin-top:10px">'
            f'<span>{label}</span><span>{p:.3f}</span></div>'
            f'<div class="bar"><span style="width:{max(1.5, 100 * p):.1f}%"></span></div>')


def legend(items):
    """items: list of (kind, colour, text); kind 'box', 'line' or 'ring'."""
    out = []
    for kind, colour, text in items:
        style = f' style="background:{colour}"' if kind == "box" else ""
        cls = {"box": "", "line": ' class="line"', "ring": ' class="ring"'}[kind]
        out.append(f"<span><i{cls}{style}></i>{html.escape(text)}</span>")
    st.markdown(f'<div class="legend">{"".join(out)}</div>', unsafe_allow_html=True)


def seg_legend():
    return [("box", _hex(COLOURS[r]), REGION_NAME[r] + f" ({r})") for r in ("WT", "TC", "ET")]


def _hex(c):
    return "#%02x%02x%02x" % tuple(int(255 * v) for v in c)


def default_slice(ref, view, n):
    axis = AXIS[view]
    return int(np.round(np.nonzero(ref)[axis - 1].mean())) if ref is not None and ref.any() else n // 2


def slice_controls(x, view, ref, key):
    """Slice slider and background sequence, side by side above the image."""
    axis = AXIS[view]
    n = x.shape[axis]
    present = [m for m in CH if x[CH[m]].any()]
    c1, c2 = st.columns([3, 2])
    k = c1.slider("Slice", 0, n - 1, default_slice(ref, view, n), key=f"{key}_slice_{view}")
    mod = c2.segmented_control("Background sequence", present, default=present[0], key=f"{key}_mod") or present[0]
    return k, mod


def figure(x, view, k, mod, seg=None, heat=None, gt=None, midline=None, right_title="Overlay"):
    """Two dark panels: the scan, and the scan with segmentation and/or heat map."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axis = AXIS[view]
    cut = lambda a: np.take(a, k, axis=axis - 1).T  # noqa: E731
    brain = cut((x != 0).any(0))
    img = np.ma.masked_where(~brain, cut(x[CH[mod]].astype(np.float32)))  # outside the brain: panel colour
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor="#0B1220")
    for ax, title in zip(axes, (f"{mod}  ·  {view.lower()} slice {k}", right_title)):
        ax.set_facecolor("#0B1220")
        ax.imshow(img, origin="lower", cmap="gray", vmin=-2.5, vmax=3, interpolation="nearest")
        ax.set_title(title, color="#E2E8F0", fontsize=11, pad=8)
        ax.axis("off")
        if midline is not None and view != "Sagittal":
            ax.axvline(midline - 0.5, color="#94A3B8", lw=0.8, ls="--")
    if seg:  # {region: mask}, drawn largest first
        rgba = np.zeros(cut(next(iter(seg.values()))).shape + (4,))
        for r, m in seg.items():
            rgba[cut(m)] = (*COLOURS[r], 0.5)
        axes[1].imshow(rgba, origin="lower", interpolation="nearest")
    if heat is not None:
        axes[1].imshow(np.ma.masked_less(cut(heat), 0.05), origin="lower", cmap="turbo", alpha=0.45, vmin=0,
                       vmax=1, interpolation="bilinear")
    if gt is not None and gt.any():
        for ax in axes:
            ax.contour(cut(gt).astype(float), levels=[0.5], colors="#22D3EE", linewidths=1.0)
    fig.tight_layout(pad=0.6)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def seg_dict(seg):
    """(ET, TC, WT) boolean stack -> {'WT', 'TC', 'ET'} in drawing order (largest first)."""
    return {"WT": seg[2], "TC": seg[1], "ET": seg[0]}
