"""PLAN2 section 5: Grad-CAM++, occlusion, region SHAP and LIME for the detection logit, compared side by side.

Methods (settings fixed in PLAN2 5):
  gradcam    Grad-CAM++ on the detection logit at the stage one above the bottleneck (enc[-2]);
             gradcam_old = PLAN's layer (enc[-1]) for reference
  occlusion  16 mm cube, stride 8 mm, filled with the Gaussian-blurred input (sigma 8 mm)
  shap       permutation Shapley values over regions: each predicted pass-B tumour component, 16 mm brain
             cubes (no SRI24 atlas labels are available in this project, the plan's fallback) minus the
             tumour, and outside-brain; 2000 model evaluations; absent region = blurred baseline. The same
             evaluations give Shapley values for the pass-B WT volume.
  lime       ~100 3D SLIC supervoxels on T1/T2 inside the brain, 1000 samples, same baseline, weighted ridge
             on the logit; 3 seeds, top-5 stability (Jaccard)
Comparison per case: deletion and insertion AUC of the detection probability (most relevant first, 11 steps,
against a random order), tumour ratio and pointing game (positives), region-level Spearman agreement,
seconds per case. Sanity (Adebayo et al. 2018): cascading randomisation (stage 1: detection head; stage 2:
whole network) on a subset; Spearman of each method's map with the trained model's map.
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shortcut_checks as sc  # noqa: E402

CUBE = 8
SHAP_EVALS = 2000
LIME_SAMPLES, LIME_SEGMENTS, LIME_SEEDS = 1000, 100, (0, 1, 2)
STEPS = np.linspace(0, 1, 11)


def baseline(x):
    return np.stack([ndi.gaussian_filter(c, sc.BLUR_SIGMA) for c in np.asarray(x, np.float32)])


@torch.no_grad()
def _eval_masks(model, x, base, regions, masks, device, bs=32):
    """Logits and pass-B WT volumes (mL) for each binary mask over regions (1 = region present)."""
    xt, bt = torch.from_numpy(x).to(device), torch.from_numpy(base).to(device)
    lab = torch.from_numpy(regions.astype(np.int64)).to(device)
    out_l, out_v = [], []
    for s in range(0, len(masks), bs):
        mk = torch.from_numpy(np.asarray(masks[s:s + bs], np.float32)).to(device)
        keep = mk[:, lab]                                  # (B, D, H, W)
        xb = xt[None] * keep[:, None] + bt[None] * (1 - keep[:, None])
        lg, wt, *_ = model.forward_det(xb)
        out_l.append(lg.cpu().numpy())
        out_v.append(((torch.sigmoid(wt) > 0.5).flatten(1).sum(1).float() * 8 / 1000).cpu().numpy())
    return np.concatenate(out_l), np.concatenate(out_v)


def shap_regions(x, wt_pred):
    """Region label map: 0 = outside brain, 1..k = tumour components, then 16 mm brain cubes."""
    brain = (x != 0).any(0)
    lab = np.zeros(brain.shape, np.int32)
    comp, k = ndi.label(wt_pred & brain)
    lab[comp > 0] = comp[comp > 0]
    nxt = k + 1
    rest = brain & (comp == 0)
    for a in range(0, brain.shape[0], CUBE):
        for b in range(0, brain.shape[1], CUBE):
            for c in range(0, brain.shape[2], CUBE):
                sl = (slice(a, a + CUBE), slice(b, b + CUBE), slice(c, c + CUBE))
                if rest[sl].any():
                    lab[sl][rest[sl]] = nxt
                    nxt += 1
    return lab, k


def shap_values(model, x, regions, base, device, evals=SHAP_EVALS, seed=0, bs=32):
    """Permutation Shapley (antithetic pairs) for logit and volume; returns (phi_logit, phi_vol) per region."""
    n = int(regions.max()) + 1
    rng = np.random.default_rng(seed)
    per = max(1, evals // (n + 1))
    masks, orders = [], []
    for p in range(per):
        order = rng.permutation(n) if p % 2 == 0 else orders[-1][::-1]
        orders.append(order)
        m = np.zeros(n, np.float32)
        masks.append(m.copy())
        for j in order:
            m[j] = 1
            masks.append(m.copy())
    lg, vol = _eval_masks(model, x, base, regions, masks, device, bs=bs)
    phi_l, phi_v = np.zeros(n), np.zeros(n)
    k = 0
    for order in orders:
        seg_l, seg_v = lg[k:k + n + 1], vol[k:k + n + 1]
        phi_l[order] += np.diff(seg_l)
        phi_v[order] += np.diff(seg_v)
        k += n + 1
    return phi_l / len(orders), phi_v / len(orders), len(masks)


def lime_values(model, x, base, device, seed, samples=LIME_SAMPLES, segments=None):
    from sklearn.linear_model import Ridge
    brain = (x != 0).any(0)
    if segments is None:
        from skimage.segmentation import slic
        img = np.moveaxis(x[[0, 2]], 0, -1)
        segments = slic(img, n_segments=LIME_SEGMENTS, compactness=0.1, channel_axis=-1, mask=brain,
                        start_label=1, enforce_connectivity=False)
    n = int(segments.max()) + 1                              # 0 = outside brain (always present)
    rng = np.random.default_rng(seed)
    z = (rng.random((samples, n)) < 0.5).astype(np.float32)
    z[:, 0] = 1
    z[0] = 1
    lg, _ = _eval_masks(model, x, base, segments, z, device)
    dist = 1 - z[:, 1:].mean(1)
    w = np.exp(-(dist ** 2) / 0.25 ** 2)
    coef = Ridge(alpha=1.0).fit(z[:, 1:], lg, sample_weight=w).coef_
    return np.concatenate([[0.0], coef]), segments


def region_map(values, regions):
    return np.asarray(values, np.float32)[regions]


@torch.no_grad()
def curves(model, x, cmap, base, device, seed=0):
    """Deletion (zero) and insertion (from the blurred baseline) probability curves for the map's order and a
    random order; returns AUCs."""
    brain = (x != 0).any(0).ravel()
    idx = np.flatnonzero(brain)
    order = idx[np.argsort(-cmap.ravel()[idx], kind="stable")]
    rnd = np.random.default_rng(seed).permutation(idx)
    xs = []
    for o in (order, rnd):
        for f in STEPS:
            sel = o[:int(round(f * len(o)))]
            d = x.reshape(x.shape[0], -1).copy()
            d[:, sel] = 0
            xs.append(d.reshape(x.shape))
            i = base.reshape(base.shape[0], -1).copy()
            i[:, sel] = x.reshape(x.shape[0], -1)[:, sel]
            xs.append(i.reshape(x.shape))
    p = []
    for s in range(0, len(xs), 22):
        xb = torch.from_numpy(np.stack(xs[s:s + 22])).to(device)
        p.append(torch.sigmoid(model.forward_cls(xb)).cpu().numpy())
    p = np.concatenate(p).reshape(2, len(STEPS), 2)            # (order, step, deletion/insertion)
    auc = lambda v: float(np.trapezoid(v, STEPS)) if hasattr(np, "trapezoid") else float(np.trapz(v, STEPS))
    return dict(deletion_auc=auc(p[0, :, 0]), deletion_auc_random=auc(p[1, :, 0]),
                insertion_auc=auc(p[0, :, 1]), insertion_auc_random=auc(p[1, :, 1]))


def explain_case(model, x, tumour, device):
    """All maps for one hemisphere + per-method metrics. Returns (maps, metrics, regions)."""
    from explain import mass_stats
    x = np.asarray(x, np.float32)
    brain = (x != 0).any(0)
    base = baseline(x)
    xt = torch.from_numpy(x)[None].to(device)
    with torch.no_grad():
        _, wt, *_ = model.forward_det(xt)
    wt_pred = (torch.sigmoid(wt[0]) > 0.5).cpu().numpy()
    maps, secs = {}, {}
    t = time.time()
    maps["gradcam"] = sc.gradcam_det(model, xt, model.enc[-2])[0]
    secs["gradcam"] = time.time() - t
    t = time.time()
    maps["gradcam_old"] = sc.gradcam_det(model, xt, model.enc[-1])[0]
    secs["gradcam_old"] = time.time() - t
    t = time.time()
    maps["occlusion"], _ = sc.occlusion(model, x, device)
    secs["occlusion"] = time.time() - t
    t = time.time()
    regions, n_tum = shap_regions(x, wt_pred)
    phi_l, phi_v, n_eval = shap_values(model, x, regions, base, device)
    maps["shap"] = region_map(np.maximum(phi_l, 0), regions)
    maps["shap_volume"] = region_map(np.maximum(phi_v, 0), regions)
    secs["shap"] = time.time() - t
    t = time.time()
    lime, segs, tops = [], None, []
    for s in LIME_SEEDS:
        coef, segs = lime_values(model, x, base, device, s, segments=segs)
        lime.append(coef)
        tops.append(set(np.argsort(-coef)[:5].tolist()))
    maps["lime"] = region_map(np.maximum(np.mean(lime, 0), 0), segs)
    jac = [len(a & b) / len(a | b) for i, a in enumerate(tops) for b in tops[i + 1:]]
    secs["lime"] = time.time() - t
    met = {}
    for name in ("gradcam", "gradcam_old", "occlusion", "shap", "lime"):
        cm = maps[name]
        r = dict(seconds=secs[name], **curves(model, x, cm, base, device))
        if tumour.any():
            ms = mass_stats(cm, brain, tumour)
            r.update(tumour_ratio=ms.get("tumour_ratio", float("nan")), outside_mass=ms["outside_mass"],
                     pointing=bool(tumour.ravel()[int(np.argmax(cm))]) if cm.max() > 0 else False)
        met[name] = r
    met["lime"]["top5_jaccard"] = float(np.mean(jac))
    met["shap"]["regions"], met["shap"]["tumour_components"], met["shap"]["evals"] = int(regions.max()) + 1, n_tum, n_eval
    reg_vals = {}
    for name in ("gradcam", "occlusion", "shap", "lime"):
        s = ndi.mean(maps[name], labels=regions, index=np.arange(1, regions.max() + 1))
        reg_vals[name] = np.nan_to_num(np.asarray(s, float))
    agree = {}
    names = list(reg_vals)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            agree[f"{a}~{b}"] = float(spearmanr(reg_vals[a], reg_vals[b])[0]) if len(reg_vals[a]) > 2 else float("nan")
    met["agreement"] = agree
    return maps, met, regions


def randomised(model, stage, seed=0):
    """Cascading randomisation: stage 1 = detection head re-initialised; stage 2 = every layer."""
    m = copy.deepcopy(model)
    g = torch.Generator().manual_seed(seed)
    mods = list(m.cls_head.modules()) if stage == 1 else list(m.modules())
    with torch.no_grad():
        for mod in mods:
            for name, p in mod.named_parameters(recurse=False):
                p.copy_(torch.randn(p.shape, generator=g) * (0.05 if "weight" in name else 0.0))
    return m.eval()


def sanity(model, x, maps, device):
    """Spearman (brain voxels) between the trained map and the map of each randomisation stage."""
    x = np.asarray(x, np.float32)
    brain = (x != 0).any(0)
    out = {}
    for stage in (1, 2):
        rm = randomised(model, stage)
        xt = torch.from_numpy(x)[None].to(device)
        rmaps = dict(gradcam=sc.gradcam_det(rm, xt, rm.enc[-2])[0], occlusion=sc.occlusion(rm, x, device)[0])
        base = baseline(x)
        with torch.no_grad():
            _, wt, *_ = rm.forward_det(xt)
        regions, _ = shap_regions(x, (torch.sigmoid(wt[0]) > 0.5).cpu().numpy())
        rmaps["shap"] = region_map(np.maximum(shap_values(rm, x, regions, base, device)[0], 0), regions)
        coef, segs = lime_values(rm, x, base, device, 0)
        rmaps["lime"] = region_map(np.maximum(coef, 0), segs)
        for k, v in rmaps.items():
            a, b = maps[k][brain], v[brain]
            out[f"{k}_stage{stage}"] = float(spearmanr(a, b)[0]) if a.std() > 0 and b.std() > 0 else float("nan")
    return out


def pick_cases(rows, seed=42, plan=(("B+r", 20), ("B+p", 10), ("I+p", 10), ("B-r", 10), ("I-r", 10))):
    rng = np.random.default_rng(seed)
    out = []
    for cell, n in plan:
        idx = [i for i, r in enumerate(rows) if r["cell"] == cell]
        out += sorted(rng.choice(idx, min(n, len(idx)), replace=False).tolist()) if idx else []
    return out


def run(model, ev, rows, device, out_dir, n_sanity=5, plan=None, seed=42):
    """XAI comparison on the picked cases; writes xai_cases.csv and returns a summary dict."""
    import csv
    import json
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    picks = pick_cases(rows, seed, plan) if plan else pick_cases(rows, seed)
    flat, san = [], []
    for n, i in enumerate(picks):
        x = np.asarray(ev.x[i], np.float32)
        tumour = (ev.bits[i] & 4) > 0
        maps, met, _ = explain_case(model, x, tumour, device)
        for name, r in met.items():
            if name == "agreement":
                continue
            flat.append(dict(id=rows[i]["id"], cell=rows[i]["cell"], method=name,
                             **{k: v for k, v in r.items() if not isinstance(v, (dict, list))}))
        flat.append(dict(id=rows[i]["id"], cell=rows[i]["cell"], method="agreement", **met["agreement"]))
        if rows[i]["cell"] == "B+r" and len(san) < n_sanity:
            san.append(dict(id=rows[i]["id"], **sanity(model, x, maps, device)))
        if n < 4:
            np.savez_compressed(out_dir / f"maps_{rows[i]['id'].replace(':', '_')}.npz", x=x[0], tumour=tumour,
                                **{k: v for k, v in maps.items()})
        print(f"xai {n + 1}/{len(picks)} {rows[i]['id']} {rows[i]['cell']}", flush=True)
    keys = sorted({k for r in flat for k in r})
    with open(out_dir / "xai_cases.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(flat)
    summ = {}
    for name in ("gradcam", "gradcam_old", "occlusion", "shap", "lime"):
        rs = [r for r in flat if r["method"] == name]
        pos = [r for r in rs if "tumour_ratio" in r]
        summ[name] = dict(
            n=len(rs), seconds_median=float(np.median([r["seconds"] for r in rs])) if rs else float("nan"),
            **{k: float(np.median([r[k] for r in rs])) for k in ("deletion_auc", "deletion_auc_random",
                                                                   "insertion_auc", "insertion_auc_random")} if rs else {},
            tumour_ratio_median=float(np.nanmedian([r["tumour_ratio"] for r in pos])) if pos else float("nan"),
            outside_mass_median=float(np.nanmedian([r["outside_mass"] for r in pos])) if pos else float("nan"),
            pointing_rate=float(np.mean([r["pointing"] for r in pos])) if pos else float("nan"))
    summ["lime"]["top5_jaccard_median"] = float(np.median([r["top5_jaccard"] for r in flat
                                                           if r["method"] == "lime"])) if flat else float("nan")
    ag = [r for r in flat if r["method"] == "agreement"]
    summ["agreement_median"] = {k: float(np.nanmedian([r[k] for r in ag])) for k in ag[0] if k not in
                                ("id", "cell", "method")} if ag else {}
    summ["sanity"] = san
    summ["sanity_median"] = {k: float(np.nanmedian([s[k] for s in san])) for k in san[0] if k != "id"} if san else {}
    (out_dir / "xai_summary.json").write_text(json.dumps(summ, indent=1))
    return summ
