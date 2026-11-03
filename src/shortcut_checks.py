"""PLAN2 section 4: pre-registered shortcut checks S1-S9 and G1 (tag plan2-prereg), one code path for val and test.

Computation helpers take the model and per-row predictions; `decide` applies the fixed pass rules:
  S1 MUST  direction-free AUROC (max(A, 1-A)) of the score separating datasets within a label:
           B-r vs I-r and B+p vs I+p, both <= 0.70
  S2 MUST  |TNR(B-r) - TNR(I-r)| <= 0.10 and |TPR(B+p) - TPR(I+p)| <= 0.10 at the frozen threshold
  S3 MUST  AUROC(B+r vs B-r) >= 0.80 and AUROC(I+p vs I-r) >= 0.80
  S4 MUST  detected B+r with the tumour replaced by mirrored healthy tissue: >= 70% fall below the threshold
           and median logit change <= -2
  S5 MUST  tumour pasted into real val negatives: >= 60% cross in each dataset, rates differ by <= 0.15
  S6 MUST  sham paste into real val negatives: <= 5% cross in each dataset and median |logit change| < 1
  S7 MUST  on detected B+r: Grad-CAM++ or occlusion with median tumour ratio >= 2 and median outside-brain
           mass <= 0.20, and the same map's top-30% deletion beats random (one-sided paired Wilcoxon p < 0.05)
  S8 SHOULD 1-voxel brain-mask erosion: median |logit change| < 1 and <= 5% crossings in each real cell
  S9 SHOULD report only: logistic-regression probe on pooled detection features, dataset within label
  G1 MUST  median pass-B WT volume <= 1 mL on B-r and on I-r
"Cross" = below the threshold before and at or above it after (S5, S6, S8 crossings counted both ways for S8).
The segmentation-derived detector (pass-B WT volume >= v*) gets S1-S6, S8 and G1 on its volume; the parts
stated in logits (S4 and S6 magnitudes, S8 magnitude) and S7 have no counterpart for a volume and are
reported as n/a (decided before any PLAN2 result, DECISIONS 1 Nov).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hemi  # noqa: E402

S7_DELETE = 0.3
S7_MAX_CASES = 40
OCC_CUBE, OCC_STRIDE, BLUR_SIGMA = 8, 4, 4.0   # 16 mm cube, 8 mm stride, Gaussian sigma 8 mm (2 mm voxels)


def auroc(pos, neg):
    from sklearn.metrics import roc_auc_score
    pos, neg = list(pos), list(neg)
    if not pos or not neg:
        return float("nan")
    return float(roc_auc_score([1] * len(pos) + [0] * len(neg), pos + neg))


def _sc(rows, cell, key):
    return [r[key] for r in rows if r["cell"] == cell]


@torch.no_grad()
def logits_of(model, xs, device, bs=8):
    out, vols = [], []
    for s in range(0, len(xs), bs):
        x = torch.from_numpy(np.stack([np.asarray(a, np.float32) for a in xs[s:s + bs]])).to(device)
        lg, wt, *_ = model.forward_det(x)
        out += lg.cpu().tolist()
        vols += ((torch.sigmoid(wt) > 0.5).flatten(1).sum(1).float() * 8 / 1000).cpu().tolist()
    return out, vols


# ---------------------------------------------------------------- S4: necessity (mirror the tumour away)
def s4_mirror(model, rows, scans, m, width, device, thr_head, thr_seg):
    """Rows of B+r detected by the head or the seg detector. scans: subject -> (x full, bits full)."""
    from evaluate import mirror_tumour
    todo = [r for r in rows if r["cell"] == "B+r" and (r["prob"] >= thr_head or r["wt_ml_b"] >= thr_seg)]
    xs = []
    for r in todo:
        x, bits = scans[r["subject"]]
        xm, _ = mirror_tumour(x, bits)
        xs.append(hemi.renorm(hemi.cut(xm, m, r["id"].split(":")[1], width)))
    lg, vol = logits_of(model, xs, device)
    return [dict(id=r["id"], logit=r["logit"], prob=r["prob"], wt_ml_b=r["wt_ml_b"], logit_after=a,
                 prob_after=float(1 / (1 + np.exp(-a))), wt_ml_b_after=v) for r, a, v in zip(todo, lg, vol)]


# ---------------------------------------------------------------- S8: boundary erosion
def s8_erosion(model, ev, rows, device):
    import p2_metrics as pm
    real = [i for i, r in enumerate(rows) if r["cell"] in ("B+r", "B-r", "I-r")]
    out = []
    for s in range(0, len(real), 8):
        idx = real[s:s + 8]
        x = torch.from_numpy(np.stack([np.asarray(ev.x[i], np.float32) for i in idx])).to(device)
        bits = torch.from_numpy(np.stack([ev.bits[i] for i in idx])).to(device)
        target = torch.stack([(bits >> k) & 1 for k in range(3)], 1).float()
        x = pm.erode_batch(x, target, None, 1, fixed=True)
        with torch.no_grad():
            lg, wt, *_ = model.forward_det(x)
        vol = (torch.sigmoid(wt) > 0.5).flatten(1).sum(1).float() * 8 / 1000
        for j, i in enumerate(idx):
            out.append(dict(id=rows[i]["id"], cell=rows[i]["cell"], logit=rows[i]["logit"], prob=rows[i]["prob"],
                            wt_ml_b=rows[i]["wt_ml_b"], logit_after=float(lg[j]),
                            prob_after=float(torch.sigmoid(lg[j])), wt_ml_b_after=float(vol[j])))
    return out


# ---------------------------------------------------------------- S9: feature probe
@torch.no_grad()
def pooled_features(model, xs, device, bs=8):
    from datasets import detection_input
    out = []
    for s in range(0, len(xs), bs):
        x = torch.from_numpy(np.stack([np.asarray(a, np.float32) for a in xs[s:s + bs]])).to(device)
        out.append(model.encode(detection_input(x))[-1].mean((2, 3, 4)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1))


def s9_probe(model, ev, rows, device, seed=42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    res = {}
    for name, (a, b) in (("negatives B-r vs I-r", ("B-r", "I-r")), ("pasted positives B+p vs I+p", ("B+p", "I+p"))):
        idx = [i for i, r in enumerate(rows) if r["cell"] in (a, b)]
        y = np.array([rows[i]["cell"] == a for i in idx], int)
        if len(idx) < 10 or y.min() == y.max() or min(y.sum(), (1 - y).sum()) < 5:
            res[name] = float("nan")
            continue
        f = pooled_features(model, [ev.x[i] for i in idx], device)
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), f, y, cv=cv,
                              method="predict_proba")[:, 1]
        res[name] = auroc(p[y == 1], p[y == 0])
    return res


# ---------------------------------------------------------------- S7: localisation and faithfulness
def gradcam_det(model, x, layer):
    from gradcam3d import GradCAMpp
    cam = GradCAMpp(layer)
    try:
        with torch.enable_grad():
            logit = model.forward_cls(x)
            maps = cam.compute(logit.sum(), x.shape[2:])
    finally:
        cam.remove()
    return maps.detach().cpu().numpy()


@torch.no_grad()
def occlusion(model, x, device, bs=32, stride=OCC_STRIDE):
    """Occlusion map of one hemisphere (4, D, H, W): mean logit drop over the 16 mm cubes covering each voxel,
    each cube filled with the Gaussian-blurred input (sigma 8 mm); cubes without brain are skipped. ReLU'd.
    stride: the checks use OCC_STRIDE (8 mm); only the GUI preview passes a coarser one."""
    x = np.asarray(x, np.float32)
    base = np.stack([ndi.gaussian_filter(c, BLUR_SIGMA) for c in x])
    brain = (x != 0).any(0)
    shape = x.shape[1:]
    starts = [list(range(0, max(s - OCC_CUBE, 0) + 1, stride)) for s in shape]
    for d, s in enumerate(shape):
        if starts[d][-1] + OCC_CUBE < s:
            starts[d].append(s - OCC_CUBE)
    pos = [(a, b, c) for a in starts[0] for b in starts[1] for c in starts[2]
           if brain[a:a + OCC_CUBE, b:b + OCC_CUBE, c:c + OCC_CUBE].any()]
    xt = torch.from_numpy(x).to(device)
    bt = torch.from_numpy(base).to(device)
    l0 = float(model.forward_cls(xt[None])[0])
    acc = np.zeros(shape, np.float64)
    cnt = np.zeros(shape, np.float64)
    for s in range(0, len(pos), bs):
        chunk = pos[s:s + bs]
        xb = xt[None].repeat(len(chunk), 1, 1, 1, 1)
        for j, (a, b, c) in enumerate(chunk):
            sl = (slice(a, a + OCC_CUBE), slice(b, b + OCC_CUBE), slice(c, c + OCC_CUBE))
            xb[(j, slice(None)) + sl] = bt[(slice(None),) + sl]
        drops = l0 - model.forward_cls(xb).cpu().numpy()
        for (a, b, c), dv in zip(chunk, drops):
            acc[a:a + OCC_CUBE, b:b + OCC_CUBE, c:c + OCC_CUBE] += dv
            cnt[a:a + OCC_CUBE, b:b + OCC_CUBE, c:c + OCC_CUBE] += 1
    return np.maximum(np.where(cnt > 0, acc / np.maximum(cnt, 1), 0), 0).astype(np.float32), len(pos)


@torch.no_grad()
def deletion_drop(model, x, cmap, device, frac=S7_DELETE, seed=42):
    """(prob drop zeroing the top `frac` brain voxels by cmap, prob drop zeroing as many random brain voxels)."""
    x = np.asarray(x, np.float32)
    brain = (x != 0).any(0).ravel()
    idx = np.flatnonzero(brain)
    k = int(round(frac * len(idx)))
    top = idx[np.argsort(-cmap.ravel()[idx], kind="stable")[:k]]
    rnd = np.random.default_rng(seed).choice(idx, k, replace=False)
    xs = []
    for sel in (top, rnd):
        xx = x.reshape(x.shape[0], -1).copy()
        xx[:, sel] = 0
        xs.append(xx.reshape(x.shape))
    xb = torch.from_numpy(np.stack([x] + xs)).to(device)
    p = torch.sigmoid(model.forward_cls(xb)).cpu().numpy()
    return float(p[0] - p[1]), float(p[0] - p[2])


def s7_localisation(model, ev, rows, thr, device, layer, max_cases=S7_MAX_CASES, seed=42, maps_out=None):
    from explain import mass_stats
    det = [i for i, r in enumerate(rows) if r["cell"] == "B+r" and r["prob"] >= thr]
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(det, min(max_cases, len(det)), replace=False).tolist()) if det else []
    out = []
    for i in pick:
        x = np.asarray(ev.x[i], np.float32)
        brain = (x != 0).any(0)
        tumour = (ev.bits[i] & 4) > 0
        xt = torch.from_numpy(x)[None].to(device)
        maps = dict(gradcam=gradcam_det(model, xt, layer)[0])
        maps["occlusion"], n_pos = occlusion(model, x, device)
        r = dict(id=rows[i]["id"], prob=rows[i]["prob"], occlusion_cubes=n_pos)
        for name, cm in maps.items():
            ms = mass_stats(cm, brain, tumour)
            dc, dr = deletion_drop(model, x, cm, device)
            r.update({f"{name}_tumour_ratio": ms.get("tumour_ratio", float("nan")),
                      f"{name}_outside_mass": ms["outside_mass"], f"{name}_shell_ratio": ms["shell_ratio"],
                      f"{name}_drop_map": dc, f"{name}_drop_random": dr})
        if maps_out is not None and len(maps_out) < 6:
            maps_out[rows[i]["id"]] = dict(x=x[0], tumour=tumour, **maps)
        out.append(r)
    return out


# ---------------------------------------------------------------- decision (fixed rules)
def _cross_up(before, after, thr):
    """Share of cases below thr before that are at or above thr after (and the n of cases below before)."""
    pairs = [(b, a) for b, a in zip(before, after) if b < thr]
    return (float(np.mean([a >= thr for _, a in pairs])) if pairs else float("nan")), len(pairs)


def _paired(rows, pasted_cell, host_cell, key):
    host = {r["sample"]: r for r in rows if r["cell"] == host_cell}
    return [(host[r["sample"]][key], r[key]) for r in rows if r["cell"] == pasted_cell and r["sample"] in host]


def decide(rows, thr, s4=None, s7=None, s8=None, s9=None, detector="head"):
    """[dict(id, tier, value, rule, passed)] for one detector. detector 'head': score = logit/prob;
    'seg': score = pass-B WT volume (mL), threshold v*."""
    head = detector == "head"
    sk = "logit" if head else "wt_ml_b"         # ranking score for AUROCs
    pk = "prob" if head else "wt_ml_b"          # thresholded score
    t = thr
    out = []

    def add(cid, tier, value, rule, passed):
        out.append(dict(id=cid, tier=tier, value=value, rule=rule,
                        passed=None if passed is None else bool(passed)))

    a1 = max(*(lambda v: (v, 1 - v))(auroc(_sc(rows, "B-r", sk), _sc(rows, "I-r", sk))))
    a2 = max(*(lambda v: (v, 1 - v))(auroc(_sc(rows, "B+p", sk), _sc(rows, "I+p", sk))))
    add("S1", "MUST", dict(neg_B_vs_I=a1, pastedpos_B_vs_I=a2), "both <= 0.70", a1 <= 0.70 and a2 <= 0.70)

    def rate(cell, pos):
        v = np.asarray(_sc(rows, cell, pk), float)
        return float(np.mean(v >= t) if pos else np.mean(v < t)) if len(v) else float("nan")
    g_tnr, g_tpr = abs(rate("B-r", 0) - rate("I-r", 0)), abs(rate("B+p", 1) - rate("I+p", 1))
    add("S2", "MUST", dict(TNR_Br=rate("B-r", 0), TNR_Ir=rate("I-r", 0), TPR_Bp=rate("B+p", 1),
                           TPR_Ip=rate("I+p", 1), tnr_gap=g_tnr, tpr_gap=g_tpr),
        "both gaps <= 0.10", g_tnr <= 0.10 and g_tpr <= 0.10)
    b3 = auroc(_sc(rows, "B+r", sk), _sc(rows, "B-r", sk))
    i3 = auroc(_sc(rows, "I+p", sk), _sc(rows, "I-r", sk))
    add("S3", "MUST", dict(Bpr_vs_Br=b3, Ip_vs_Ir=i3), "both >= 0.80", b3 >= 0.80 and i3 >= 0.80)

    if s4 is not None:
        det = [r for r in s4 if r[pk] >= t]
        fell = float(np.mean([r[pk + "_after"] < t for r in det])) if det else float("nan")
        if head:
            med = float(np.median([r["logit_after"] - r["logit"] for r in det])) if det else float("nan")
            add("S4", "MUST", dict(n=len(det), share_below_after=fell, median_logit_change=med),
                ">= 70% fall below and median logit change <= -2", det and fell >= 0.70 and med <= -2)
        else:
            med = float(np.median([r["wt_ml_b_after"] - r["wt_ml_b"] for r in det])) if det else float("nan")
            add("S4", "MUST", dict(n=len(det), share_below_after=fell, median_volume_change_ml=med),
                ">= 70% fall below (logit part n/a for a volume)", det and fell >= 0.70)

    c5 = {}
    for d, (pc, hc) in (("brats", ("B+p", "B-r")), ("ixi", ("I+p", "I-r"))):
        pr = _paired(rows, pc, hc, pk)
        c5[d], c5[d + "_n"] = _cross_up([b for b, _ in pr], [a for _, a in pr], t)
    ok5 = c5["brats"] >= 0.60 and c5["ixi"] >= 0.60 and abs(c5["brats"] - c5["ixi"]) <= 0.15
    add("S5", "MUST", c5, ">= 60% cross in each dataset, rates differ by <= 0.15", ok5)

    c6, ok6 = {}, True
    for d, (pc, hc) in (("brats", ("B-s", "B-r")), ("ixi", ("I-s", "I-r"))):
        pr = _paired(rows, pc, hc, pk)
        c6[d], c6[d + "_n"] = _cross_up([b for b, _ in pr], [a for _, a in pr], t)
        ok6 &= c6[d] <= 0.05
        if head:
            lp = _paired(rows, pc, hc, "logit")
            c6[d + "_median_abs_logit_change"] = float(np.median([abs(a - b) for b, a in lp])) if lp else float("nan")
            ok6 &= c6[d + "_median_abs_logit_change"] < 1
    add("S6", "MUST", c6, "<= 5% cross in each dataset" + (" and median |logit change| < 1 in each" if head else
                                                         " (logit part n/a for a volume)"), ok6)

    if head and s7 is not None:
        res, any_ok = {"n": len(s7)}, False
        for name in ("gradcam", "occlusion"):
            tr = float(np.nanmedian([r[f"{name}_tumour_ratio"] for r in s7])) if s7 else float("nan")
            om = float(np.nanmedian([r[f"{name}_outside_mass"] for r in s7])) if s7 else float("nan")
            dm, dr = [r[f"{name}_drop_map"] for r in s7], [r[f"{name}_drop_random"] for r in s7]
            try:
                p = float(wilcoxon(dm, dr, alternative="greater").pvalue) if len(s7) >= 6 else float("nan")
            except ValueError:
                p = float("nan")
            ok = tr >= 2 and om <= 0.20 and p < 0.05
            res[name] = dict(median_tumour_ratio=tr, median_outside_mass=om, deletion_p=p, passed=bool(ok))
            any_ok |= ok
        add("S7", "MUST", res, "one map: median tumour ratio >= 2, outside mass <= 0.20, deletion p < 0.05",
            any_ok)
    elif not head:
        add("S7", "MUST", "n/a", "not applicable to the segmentation-derived detector", None)

    if s8 is not None:
        res8, ok8 = {}, True
        for cell in ("B+r", "B-r", "I-r"):
            rs = [r for r in s8 if r["cell"] == cell]
            if not rs:
                continue
            flips = float(np.mean([(r[pk] >= t) != (r[pk + "_after"] >= t) for r in rs]))
            res8[cell] = dict(n=len(rs), crossings=flips)
            ok8 &= flips <= 0.05
            if head:
                res8[cell]["median_abs_logit_change"] = float(np.median([abs(r["logit_after"] - r["logit"]) for r in rs]))
                ok8 &= res8[cell]["median_abs_logit_change"] < 1
        add("S8", "SHOULD", res8, "median |logit change| < 1 and <= 5% crossings in each real cell"
            if head else "<= 5% crossings in each real cell (logit part n/a)", ok8)
    if s9 is not None:
        add("S9", "SHOULD", s9, "report only", None)

    g = {c: float(np.median(_sc(rows, c, "wt_ml_b"))) if _sc(rows, c, "wt_ml_b") else float("nan")
         for c in ("B-r", "I-r")}
    add("G1", "MUST", g, "median pass-B WT <= 1 mL on B-r and on I-r", all(v <= 1 for v in g.values()))
    return out


def verdict(checks):
    must = [c for c in checks if c["tier"] == "MUST" and c["passed"] is not None]
    failed = [c["id"] for c in must if not c["passed"]]
    return dict(shortcut_controlled=not failed, failed_must=failed)
