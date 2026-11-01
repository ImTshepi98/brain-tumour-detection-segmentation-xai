"""Phase 6: evaluate one trained run on val or (once) on test.

Detection: recall, precision, F1, specificity, AUROC, confusion matrix, bootstrap 95% CIs
(1000 case resamples, seed 42); specificity / FPR on IXI by age band (inside, below, above,
no age) and by site, recall on BraTS. Segmentation (BraTS): per-case DSC and HD95 (MONAI,
percentile 95, spacing 2 mm) for ET/TC/WT with BraTS conventions (both empty: DSC 1, HD95 0;
exactly one empty: DSC 0, HD95 373.13). Inference time per case at batch 1 after warm-up.
Controls (val only, added after Phase 3): detection probability with the BraTS tumour replaced
by mirrored contralateral tissue, and with the brain mask eroded by 1 voxel.

The detection threshold is frozen before test: --threshold-from <results_val.json> takes the
max-F1 threshold chosen on val. Test needs --confirm-test and writes test_lock.json before any
test array is read; a second test evaluation into the same --out is refused.

  python src/evaluate.py --ckpt runs/main/best.pt --split val --out runs/main/eval ... [--controls]
  python src/evaluate.py --ckpt runs/main/best.pt --split test --confirm-test \
      --threshold-from runs/main/eval/results_val.json --out runs/main/eval ...
  python src/evaluate.py --debug --out runs/debug-eval      # synthetic CPU dry run
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datasets as ds  # noqa: E402
import metrics  # noqa: E402
from common import SEED, zscore_nonzero  # noqa: E402
from model import build  # noqa: E402

HD95_MAX = 373.13  # BraTS convention for an empty prediction or an empty ground truth
SPACING = 2.0
LR_AXIS = 0  # left-right axis of the (x, y, z) arrays


def load_checkpoint(path, device):
    st = torch.load(path, map_location=device, weights_only=False)
    model = build(st["config"]).to(device)
    model.load_state_dict(st["model"])
    model.eval()
    return model, st


# ---------------------------------------------------------------- per-case predictions
def hd95(pred, gt):
    """pred, gt bool (3, ...) -> (3,) HD95 in mm with BraTS conventions."""
    from monai.metrics import compute_hausdorff_distance
    out = np.zeros(3)
    for c in range(3):
        p, g = pred[c], gt[c]
        if not p.any() and not g.any():
            out[c] = 0.0
        elif not p.any() or not g.any():
            out[c] = HD95_MAX
        else:
            out[c] = float(compute_hausdorff_distance(p[None, None].float(), g[None, None].float(),
                                                      include_background=True, percentile=95, spacing=SPACING))
    return out


@torch.no_grad()
def predict(model, cohort, device, bs=2, seg=True):
    recs = []
    for start in range(0, len(cohort), bs):
        idx = np.arange(start, min(start + bs, len(cohort)))
        x, target, _, has_seg = cohort.batch(idx, device)
        prob = torch.sigmoid(model.forward_cls(x)).cpu().numpy()
        if seg and has_seg.any():
            pred = metrics.postprocess(model.forward_seg(x[has_seg]))
            gt = target[has_seg] > 0.5
            dice = metrics.dice_per_case(pred, gt)
            hds = [hd95(pred[j].cpu(), gt[j].cpu()) for j in range(len(pred))]
        k = 0
        for j, i in enumerate(idx):
            r = dict(id=cohort.ids[i], source=cohort.source[i], site=cohort.site[i], label=int(cohort.label[i]),
                     prob=float(prob[j]))
            if seg and cohort.has_seg[i]:
                for c, name in enumerate(metrics.REGIONS):
                    r[f"dsc_{name}"] = float(dice[k, c])
                    r[f"hd95_{name}"] = float(hds[k][c])
                    r[f"{name}_present"] = bool((cohort.y[i] >> c & 1).any())
                k += 1
            recs.append(r)
    return recs


# ---------------------------------------------------------------- detection statistics
def choose_threshold(labels, probs):
    """Max F1 over candidate thresholds (midpoints between consecutive distinct probabilities, and 0.5);
    ties -> closest to 0.5. No rounding: main-r5 v1 rounded candidates to 6 decimals, which put the cut
    just below a healthy case's probability and just above a tumour case's (see DECISIONS 2026-11-01)."""
    u = np.unique(np.asarray(probs, float))
    cands = sorted(set(((u[:-1] + u[1:]) / 2).tolist()) | {0.5})
    scored = [(metrics.detection(labels, probs, t)["f1"], -abs(t - 0.5), t) for t in cands]
    return float(max(scored)[2])


def bootstrap(labels, probs, thr, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    labels, probs = np.asarray(labels), np.asarray(probs)
    keys = ("recall", "precision", "f1", "specificity", "auroc")
    draws = {k: [] for k in keys}
    for _ in range(n):
        i = rng.integers(0, len(labels), len(labels))
        m = metrics.detection(labels[i], probs[i], thr)
        for k in keys:
            draws[k].append(m[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in draws.items()}


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [0.0 if k == 0 else c - h, 1.0 if k == n else c + h]  # exact at the ends (float rounding)


def read_ages(path):
    if not path or not Path(path).exists():
        return {}
    with open(path) as f:
        return {r["subject"]: (float(r["age"]) if r["age"] else None) for r in csv.DictReader(f)}


def groups(recs, thr, ages, band=(40, 75)):
    """Specificity / FPR on IXI by age band and site; recall on BraTS (no ages for BraTS)."""
    def band_of(s):
        a = ages.get(s)
        if a is None:
            return "no_age"
        return "in_band" if band[0] <= a <= band[1] else ("below_band" if a < band[0] else "above_band")
    out = {}
    ixi = [r for r in recs if r["label"] == 0]
    sel = {f"ixi_{b}": [r for r in ixi if band_of(r["id"]) == b]
           for b in ("in_band", "below_band", "above_band", "no_age")}
    sel.update({f"ixi_site_{s}": [r for r in ixi if r["site"] == s] for s in sorted({r["site"] for r in ixi})})
    sel["ixi_all"] = ixi
    for name, rs in sel.items():
        tn = sum(r["prob"] < thr for r in rs)
        out[name] = dict(n=len(rs), tn=tn, fp=len(rs) - tn, specificity=tn / len(rs) if rs else float("nan"),
                         fpr=(len(rs) - tn) / len(rs) if rs else float("nan"), specificity_ci=wilson(tn, len(rs)))
    br = [r for r in recs if r["label"] == 1]
    tp = sum(r["prob"] >= thr for r in br)
    out["brats_all"] = dict(n=len(br), tp=tp, fn=len(br) - tp, recall=tp / len(br) if br else float("nan"),
                            recall_ci=wilson(tp, len(br)))
    return out


def seg_summary(recs):
    seg = [r for r in recs if "dsc_WT" in r]
    out = {}
    for name in metrics.REGIONS:
        for m in ("dsc", "hd95"):
            v = np.array([r[f"{m}_{name}"] for r in seg])
            out[f"{m}_{name}"] = dict(mean=float(v.mean()) if len(v) else float("nan"),
                                      sd=float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
                                      median=float(np.median(v)) if len(v) else float("nan"), n=len(v))
    return out


# ---------------------------------------------------------------- controls (val only)
def mirror_tumour(x, bits):
    """Replace WT voxels with the mirror image across the brain's left-right midline. Voxels whose
    mirror is outside the brain or inside WT get each channel's median of tumour-free brain.
    Returns (x_new float32, fraction of WT voxels filled from a valid mirror)."""
    x = np.asarray(x, np.float32)
    brain = (x != 0).any(0)
    wt = (np.asarray(bits) & 4) > 0
    if not wt.any():
        return x.copy(), float("nan")
    lo, hi = [(int(i.min()), int(i.max())) for i in np.nonzero(brain)][LR_AXIS]
    n = x.shape[1 + LR_AXIS]
    src = lo + hi - np.arange(n)
    valid = (src >= 0) & (src < n)
    srcc = np.clip(src, 0, n - 1)
    xm = np.take(x, srcc, axis=1 + LR_AXIS)
    shape = [1, 1, 1]
    shape[LR_AXIS] = n
    ok = wt & np.take(brain & ~wt, srcc, axis=LR_AXIS) & valid.reshape(shape)
    out = x.copy()
    out[:, ok] = xm[:, ok]
    rest = wt & ~ok
    for c in range(x.shape[0]):
        v = x[c][brain & ~wt]
        v = v[v != 0]
        out[c][rest] = float(np.median(v)) if len(v) and x[c][wt].any() else 0.0
    return out, float(ok.sum() / wt.sum())


def erode_brain(x, bits=None, iterations=1):
    """Brain mask eroded by `iterations` voxels (2 mm each; BraTS WT kept), then each channel
    re-z-scored over its non-zero voxels inside (the shared_mask rule)."""
    x = np.asarray(x, np.float32)
    brain = (x != 0).any(0)
    keep = ndi.binary_erosion(brain, iterations=iterations)
    if bits is not None:
        keep |= brain & ((np.asarray(bits) & 4) > 0)
    out = np.zeros_like(x)
    for c in range(x.shape[0]):
        if x[c][keep].any():
            out[c][keep] = zscore_nonzero(x[c][keep])[0]
    return out


@torch.no_grad()
def controls(model, cohort, device, thr):
    def prob(arr):
        return float(torch.sigmoid(model.forward_cls(torch.from_numpy(arr[None]).float().to(device)))[0])
    rows = []
    for i in range(len(cohort)):
        x = cohort.x[i].astype(np.float32)
        base = prob(x)
        r = dict(id=cohort.ids[i], label=int(cohort.label[i]), site=cohort.site[i], prob=base)
        r["prob_eroded"] = prob(erode_brain(x, cohort.y[i] if cohort.has_seg[i] else None))
        if cohort.has_seg[i]:
            xm, frac = mirror_tumour(x, cohort.y[i])
            r["prob_mirrored"], r["mirror_valid_frac"] = prob(xm), frac
        rows.append(r)

    def summ(rs, key):
        d = np.array([r[key] - r["prob"] for r in rs])
        return dict(n=len(rs), mean_change=float(d.mean()) if len(d) else float("nan"),
                    median_change=float(np.median(d)) if len(d) else float("nan"),
                    above_thr_before=int(sum(r["prob"] >= thr for r in rs)),
                    above_thr_after=int(sum(r[key] >= thr for r in rs)))
    br = [r for r in rows if r["label"] == 1]
    ixi = [r for r in rows if r["label"] == 0]
    return rows, dict(tumour_mirrored_brats=summ(br, "prob_mirrored"), eroded_ixi=summ(ixi, "prob_eroded"),
                      eroded_brats=summ(br, "prob_eroded"),
                      mirror_valid_frac_median=float(np.nanmedian([r["mirror_valid_frac"] for r in br]))
                      if br else float("nan"))


@torch.no_grad()
def inference_time(model, cohort, device, n=10, warmup=3):
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
    xs = [torch.from_numpy(cohort.x[i][None].astype(np.float32)).to(device) for i in range(min(n + warmup, len(cohort)))]
    times = []
    for k, x in enumerate(xs):
        sync()
        t = time.perf_counter()
        model.forward_cls(x)
        model.forward_seg(x)
        sync()
        if k >= warmup:
            times.append((time.perf_counter() - t) * 1000)
    return dict(ms_per_case_mean=float(np.mean(times)) if times else float("nan"),
                ms_per_case_sd=float(np.std(times)) if times else float("nan"), n=len(times), device=str(device),
                hardware=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")


def figures(recs, res, out, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    out.mkdir(parents=True, exist_ok=True)
    y, p = [r["label"] for r in recs], [r["prob"] for r in recs]
    if len(set(y)) == 2:
        fpr, tpr, _ = roc_curve(y, p)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax.set(xlabel="false positive rate", ylabel="true positive rate",
               title=f"ROC ({tag}), AUROC {res['detection']['auroc']:.3f}")
        fig.tight_layout()
        fig.savefig(out / f"roc_{tag}.png", dpi=110)
        plt.close(fig)
    d = res["detection"]
    fig, ax = plt.subplots(figsize=(3.4, 3))
    cm = np.array([[d["tn"], d["fp"]], [d["fn"], d["tp"]]])
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["healthy", "tumour"], yticklabels=["healthy", "tumour"],
           xlabel="predicted", ylabel="true", title=f"threshold {d['threshold']:.3f}")
    fig.tight_layout()
    fig.savefig(out / f"confusion_{tag}.png", dpi=110)
    plt.close(fig)
    seg = [r for r in recs if "dsc_WT" in r]
    if seg:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.boxplot([[r[f"dsc_{n}"] for r in seg] for n in metrics.REGIONS])
        ax.set_xticks([1, 2, 3], list(metrics.REGIONS))
        ax.set(ylabel="DSC", ylim=(0, 1), title=f"Segmentation ({tag}, n={len(seg)})")
        fig.tight_layout()
        fig.savefig(out / f"dsc_{tag}.png", dpi=110)
        plt.close(fig)
    g = {k: v for k, v in res["groups"].items() if k.startswith("ixi_") and v["n"]}
    fig, ax = plt.subplots(figsize=(6, 3))
    names = list(g)
    spec = [g[k]["specificity"] for k in names]
    err = np.array([[s - g[k]["specificity_ci"][0], g[k]["specificity_ci"][1] - s] for k, s in zip(names, spec)]).T
    err = np.clip(err, 0, None)  # Wilson bounds can sit a hair inside an observed 1.0 (float rounding)
    ax.bar(range(len(names)), spec, yerr=err, capsize=3)
    ax.set_xticks(range(len(names)), [f"{k[4:]}\n(n={g[k]['n']})" for k in names], fontsize=7)
    ax.set(ylabel="specificity", ylim=(0, 1.05), title=f"IXI specificity by age band and site ({tag})")
    fig.tight_layout()
    fig.savefig(out / f"specificity_groups_{tag}.png", dpi=110)
    plt.close(fig)


def evaluate(model, cohort, device, thr, ages, band=(40, 75), seg=True, bs=2):
    recs = predict(model, cohort, device, bs, seg)
    y, p = [r["label"] for r in recs], [r["prob"] for r in recs]
    det = metrics.detection(y, p, thr)
    return recs, dict(n=len(recs), detection=det, detection_ci=bootstrap(y, p, thr), groups=groups(recs, thr, ages, band),
                      segmentation=seg_summary(recs) if seg else {})


def _debug_setup(out):
    """Train a tiny model on synthetic data and write fake ages, so the whole script can be dry-run."""
    import train
    run = out / "train"
    if not (run / "best.pt").exists():
        train.main(["--debug", "--out", str(run), "--set", "model.channels=[4,8,16,32,64]", "optim.epochs=1"])
    rows = ds.read_splits(run / "debug_data" / "splits.csv")
    ages = out / "ages.csv"
    with open(ages, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "age"])
        for k, r in enumerate(r for r in rows if r["source"] == "ixi"):
            w.writerow([r["subject"], (25, 50, 80, "")[k % 4]])
    d = run / "debug_data"
    return dict(ckpt=run / "best.pt", brats_dir=d / "brats2mm", ixi_dirs=[d / "ixi2mm"],
                brats_sdt_dirs=[d / "sdt_brats"], splits=d / "splits.csv", ages=ages)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--confirm-test", action="store_true")
    ap.add_argument("--threshold", type=float, default=None, help="fixed threshold (val only unless frozen)")
    ap.add_argument("--threshold-from", help="results_val.json whose chosen threshold is used (required for test)")
    ap.add_argument("--controls", action="store_true", help="tumour-mirroring and mask-erosion controls (val only)")
    ap.add_argument("--brats-dir")
    ap.add_argument("--ixi-dirs", nargs="*", default=[])
    ap.add_argument("--brats-sdt-dirs", nargs="*", default=[])
    ap.add_argument("--splits")
    ap.add_argument("--ages", default="reports/age_ixi.csv")
    ap.add_argument("--band", nargs=2, type=float, default=[40, 75])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-count-check", action="store_true", help="skip the real split counts (kernel dry runs)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.debug:
        d = _debug_setup(out)
        args.ckpt, args.brats_dir, args.splits, args.ages = d["ckpt"], d["brats_dir"], d["splits"], d["ages"]
        args.ixi_dirs, args.brats_sdt_dirs = d["ixi_dirs"], d["brats_sdt_dirs"]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, st = load_checkpoint(args.ckpt, device)
    cfg = st["config"]
    seg = cfg["loss"]["w_seg"] > 0
    tag = args.tag or args.split

    if args.split == "test":
        assert args.confirm_test, "test is locked: pass --confirm-test (read once per model)"
        assert args.threshold_from, "test needs --threshold-from <results_val.json> (threshold frozen on val)"
        assert not args.controls, "controls run on val only"
        lock = out / "test_lock.json"
        assert not lock.exists(), f"test already evaluated for this run ({lock}); refusing a second read"
        thr = json.loads(Path(args.threshold_from).read_text())["threshold"]
        lock.write_text(json.dumps(dict(ckpt=str(args.ckpt), protocol_hash=st["protocol_hash"], epoch=st["epoch"],
                                        threshold=thr, time=time.strftime("%Y-%m-%d %H:%M:%S")), indent=2))
    rows = ds.read_splits(args.splits)
    ds.check_splits(rows, None if args.debug or args.no_count_check else ds.EXPECTED)
    cohort = ds.Cohort.load(rows, (args.split,), args.brats_dir, args.ixi_dirs, args.brats_sdt_dirs,
                            border=cfg["border"], allow_test=args.split == "test")
    ages = read_ages(args.ages)

    if args.split == "val":
        recs = predict(model, cohort, device, seg=seg)
        y, p = [r["label"] for r in recs], [r["prob"] for r in recs]
        if args.threshold is not None:
            thr, how = args.threshold, "fixed"
        elif args.threshold_from:
            thr, how = json.loads(Path(args.threshold_from).read_text())["threshold"], "frozen"
        else:
            thr, how = choose_threshold(y, p), "max_f1_on_val"
        res = dict(n=len(recs), detection=metrics.detection(y, p, thr), detection_ci=bootstrap(y, p, thr),
                   groups=groups(recs, thr, ages, tuple(args.band)), segmentation=seg_summary(recs) if seg else {},
                   detection_at_0_5=metrics.detection(y, p, 0.5))
    else:
        recs, res = evaluate(model, cohort, device, thr, ages, tuple(args.band), seg)
        how = "frozen_from_val"
    res.update(split=args.split, threshold=thr, threshold_rule=how, ckpt=str(args.ckpt), epoch=st["epoch"],
               protocol_hash=st["protocol_hash"], run=cfg.get("name"), border=cfg["border"],
               inference=inference_time(model, cohort, device))
    if args.controls:
        assert args.split == "val"
        crow, res["controls"] = controls(model, cohort, device, thr)
        with open(out / f"controls_{tag}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in crow for k in r}))
            w.writeheader()
            w.writerows(crow)
    with open(out / f"cases_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in recs for k in r}))
        w.writeheader()
        w.writerows(recs)
    (out / f"results_{tag}.json").write_text(json.dumps(res, indent=2))
    figures(recs, res, out / "figures", tag)
    print(json.dumps({k: res[k] for k in ("split", "n", "threshold", "detection")}, indent=1), flush=True)
    return res


if __name__ == "__main__":
    main()
