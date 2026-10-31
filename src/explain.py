"""Phase 7: Grad-CAM++ heatmaps, overlays and quantitative checks for one trained run.

Per case (detection map on T1/T2; segmentation map for WT on 4 channels, BraTS only):
  (1) share of heatmap mass inside the ground-truth tumour vs the tumour's share of brain volume;
  (2) shortcut check: share of mass in a 3-voxel shell at the brain boundary vs the shell's share
      of brain volume (both classes), and the share of mass outside the brain;
  (3) deletion test (detection): zero the top 10/20/30% brain voxels by CAM, report the drop in
      tumour probability, next to deleting the same number of random brain voxels.
Overlays (axial/coronal/sagittal) for up to 6 selected cases: best/median/worst WT DSC BraTS,
one healthy case inside and one outside the age band, false positives/negatives first if any.
Test: needs --confirm-test and the run's results_test.json (evaluation done first, nothing tuned).

  python src/explain.py --ckpt runs/main/best.pt --split val --cases-csv runs/main/eval/cases_val.csv ...
  python src/explain.py --debug --out runs/debug-explain
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datasets as ds  # noqa: E402
from common import SEED  # noqa: E402
from evaluate import load_checkpoint, read_ages  # noqa: E402
from gradcam3d import detection_cam, segmentation_cam  # noqa: E402

SHELL = 3
DELETE = (0.1, 0.2, 0.3)


def mass_stats(cam, brain, tumour=None):
    total = float(cam.sum())
    shell = brain & ~ndi.binary_erosion(brain, iterations=SHELL)
    nb = max(int(brain.sum()), 1)
    out = dict(shell_mass=float(cam[shell].sum()) / total if total else float("nan"),
               shell_vol=float(shell.sum()) / nb,
               outside_mass=float(cam[~brain].sum()) / total if total else float("nan"))
    out["shell_ratio"] = out["shell_mass"] / out["shell_vol"] if out["shell_vol"] else float("nan")
    if tumour is not None and tumour.any():
        out.update(tumour_mass=float(cam[tumour].sum()) / total if total else float("nan"),
                   tumour_vol=float(tumour.sum()) / nb)
        out["tumour_ratio"] = out["tumour_mass"] / out["tumour_vol"]
    return out


@torch.no_grad()
def deletion(model, x, cam, brain, device, seed=SEED):
    """Probability drop when the top-k% CAM brain voxels (or as many random brain voxels) are zeroed."""
    def prob(a):
        return float(torch.sigmoid(model.forward_cls(torch.from_numpy(a[None]).float().to(device)))[0])
    rng = np.random.default_rng(seed)
    p0 = prob(x)
    idx = np.flatnonzero(brain.ravel())
    order = idx[np.argsort(-cam.ravel()[idx], kind="stable")]
    out = {}
    for f in DELETE:
        k = int(round(f * len(idx)))
        for name, sel in (("cam", order[:k]), ("random", rng.choice(idx, k, replace=False))):
            xx = x.reshape(x.shape[0], -1).copy()
            xx[:, sel] = 0
            out[f"drop_{name}_{int(f * 100)}"] = p0 - prob(xx.reshape(x.shape))
    return p0, out


def select_cases(cases, ages, band=(40, 75), thr=0.5, n=6):
    """Deterministic pick from a cases_<split>.csv: FP/FN first, then best/median/worst WT DSC BraTS,
    one healthy inside and one outside the age band."""
    br = sorted([r for r in cases if int(r["label"]) == 1 and r.get("dsc_WT") not in (None, "")],
                key=lambda r: float(r["dsc_WT"]))
    hl = sorted([r for r in cases if int(r["label"]) == 0], key=lambda r: float(r["prob"]))
    fp = [r for r in hl if float(r["prob"]) >= thr][-1:]
    fn = [r for r in br if float(r["prob"]) < thr][:1]
    pick = fp + fn
    if br:
        pick += [br[-1], br[len(br) // 2], br[0]]

    def inband(r):
        a = ages.get(r["id"])
        return a is not None and band[0] <= a <= band[1]
    for want in (True, False):  # one healthy case inside the band, one outside, not already picked
        rs = [r for r in hl if inband(r) == want and r not in pick]
        if rs:
            pick.append(rs[len(rs) // 2])
    seen, out = set(), []
    for r in pick:
        if r["id"] not in seen:
            seen.add(r["id"])
            out.append(r["id"])
    return out[:n]


def overlay(x, bits, det, seg, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    wt = (bits & 4) > 0
    centre = ndi.center_of_mass(wt) if wt.any() else np.unravel_index(int(np.argmax(det)), det.shape)
    c = [int(round(v)) for v in centre]
    views = (("axial", lambda a: a[:, :, c[2]]), ("coronal", lambda a: a[:, c[1], :]),
             ("sagittal", lambda a: a[c[0], :, :]))
    other = x[3] if x[3].any() else x[2]
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i, (name, cut) in enumerate(views):
        for j, (img, heat, lab) in enumerate(((x[0], None, "T1"), (x[0], det, "T1 + detection CAM"),
                                              (other, seg, ("FLAIR" if x[3].any() else "T2") + " + WT seg CAM"))):
            ax = axes[i, j]
            ax.imshow(cut(img).T, origin="lower", cmap="gray", vmin=-2.5, vmax=3)
            if heat is not None:
                ax.imshow(np.ma.masked_less(cut(heat).T, 0.05), origin="lower", cmap="jet", alpha=0.45, vmin=0, vmax=1)
            if wt.any() and j > 0:
                ax.contour(cut(wt).T.astype(float), levels=[0.5], colors="w", linewidths=0.7)
            ax.set_title(f"{name}: {lab}", fontsize=7)
            ax.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def summarise(rows):
    def med(key, rs):
        v = [r[key] for r in rs if key in r and np.isfinite(r[key])]
        return (float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75)), len(v)) if v else None
    out = {}
    for cls, rs in (("tumour", [r for r in rows if r["label"] == 1]), ("healthy", [r for r in rows if r["label"] == 0])):
        keys = sorted({k for r in rs for k in r if k.startswith(("det_", "seg_"))})
        out[cls] = {k: med(k, rs) for k in keys}
    return out


def write_md(summary, path, meta):
    lines = [f"# Grad-CAM++ checks ({meta['run']}, {meta['split']}, n={meta['n']})", "",
             "Median [IQR] per case. *_mass = share of heatmap mass; *_vol = share of brain volume; "
             "*_ratio = mass / volume (1 = no enrichment). shell = outer 3 voxels (6 mm) of the brain. "
             "drop_* = fall in tumour probability after zeroing the top-k% CAM (or random) brain voxels.", ""]
    for cls, d in summary.items():
        lines += [f"## {cls}", "", "| metric | median | IQR | n |", "|---|---|---|---|"]
        for k, v in d.items():
            if v:
                lines.append(f"| {k} | {v[0]:.3f} | {v[1]:.3f} to {v[2]:.3f} | {v[3]} |")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--confirm-test", action="store_true")
    ap.add_argument("--results", help="results_<split>.json from evaluate.py (threshold; required for test)")
    ap.add_argument("--cases-csv", help="cases_<split>.csv from evaluate.py, for case selection")
    ap.add_argument("--cases", nargs="*", default=None, help="explicit case ids for overlays")
    ap.add_argument("--max-cases", type=int, default=0, help="cases for the metrics (0 = all in split)")
    ap.add_argument("--brats-dir")
    ap.add_argument("--ixi-dirs", nargs="*", default=[])
    ap.add_argument("--brats-sdt-dirs", nargs="*", default=[])
    ap.add_argument("--splits")
    ap.add_argument("--ages", default="reports/age_ixi.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    out = Path(args.out)
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    if args.debug:
        import evaluate
        res = evaluate.main(["--debug", "--out", str(out / "eval")])
        d = out / "eval" / "train" / "debug_data"
        args.ckpt, args.splits, args.ages = out / "eval" / "train" / "best.pt", d / "splits.csv", out / "eval" / "ages.csv"
        args.brats_dir, args.ixi_dirs, args.brats_sdt_dirs = d / "brats2mm", [d / "ixi2mm"], [d / "sdt_brats"]
        args.results, args.cases_csv = out / "eval" / "results_val.json", out / "eval" / "cases_val.csv"
        assert res["split"] == "val"
    if args.split == "test":
        assert args.confirm_test and args.results, "test: pass --confirm-test and the run's results_test.json"
        assert json.loads(Path(args.results).read_text())["split"] == "test", "evaluate test before explaining it"
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, st = load_checkpoint(args.ckpt, device)
    cfg = st["config"]
    thr = json.loads(Path(args.results).read_text())["threshold"] if args.results else 0.5
    rows = ds.read_splits(args.splits)
    ages = read_ages(args.ages)
    cohort = ds.Cohort.load(rows, (args.split,), args.brats_dir, args.ixi_dirs, args.brats_sdt_dirs,
                            border=cfg["border"], allow_test=args.split == "test")
    if args.cases is not None:
        chosen = args.cases
    elif args.cases_csv:
        with open(args.cases_csv) as f:
            chosen = select_cases(list(csv.DictReader(f)), ages, thr=thr)
    else:
        chosen = []
    order = list(range(len(cohort)))
    if args.max_cases:
        rng = np.random.default_rng(SEED)
        order = sorted(set(rng.choice(order, min(args.max_cases, len(order)), replace=False).tolist())
                       | {cohort.ids.index(c) for c in chosen if c in cohort.ids})
    do_seg = cfg["loss"]["w_seg"] > 0
    do_det = cfg["loss"]["w_cls"] > 0
    recs = []
    for i in order:
        x = cohort.x[i].astype(np.float32)
        xt = torch.from_numpy(x[None]).to(device)
        brain = (x != 0).any(0)
        tumour = (cohort.y[i] & 4) > 0 if cohort.has_seg[i] else None
        r = dict(id=cohort.ids[i], label=int(cohort.label[i]), site=cohort.site[i])
        det = seg = None
        if do_det:
            det, p = detection_cam(model, xt)
            det = det[0].cpu().numpy()
            r["prob"] = float(p[0])
            r.update({f"det_{k}": v for k, v in mass_stats(det, brain, tumour).items()})
            r.update({f"det_{k}": v for k, v in deletion(model, x, det, brain, device)[1].items()})
        if do_seg and cohort.has_seg[i]:
            seg, _ = segmentation_cam(model, xt, "WT")
            seg = seg[0].cpu().numpy()
            r.update({f"seg_{k}": v for k, v in mass_stats(seg, brain, tumour).items()})
        recs.append(r)
        if r["id"] in chosen:
            np.savez_compressed(out / "overlays" / f"{r['id']}_cam.npz",
                                det=(det if det is not None else np.zeros_like(brain, float)).astype(np.float16),
                                seg=(seg if seg is not None else np.zeros_like(brain, float)).astype(np.float16))
            title = f"{r['id']} ({'tumour' if r['label'] else 'healthy'}, {r['site']})" + \
                    (f", p={r['prob']:.3f}" if "prob" in r else "")
            overlay(x, cohort.y[i], det if det is not None else np.zeros_like(brain, float),
                    seg if seg is not None else np.zeros_like(brain, float), title,
                    out / "overlays" / f"{r['id']}.png")
    with open(out / f"xai_cases_{args.split}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in recs for k in r}))
        w.writeheader()
        w.writerows(recs)
    summary = summarise(recs)
    meta = dict(run=cfg.get("name"), split=args.split, n=len(recs), ckpt=str(args.ckpt), epoch=st["epoch"],
                overlay_cases=chosen, threshold=thr)
    (out / f"xai_summary_{args.split}.json").write_text(json.dumps(dict(meta=meta, summary=summary), indent=2))
    write_md(summary, out / f"xai_metrics_{args.split}.md", meta)
    print(json.dumps(meta), flush=True)
    return dict(meta=meta, summary=summary)


if __name__ == "__main__":
    main()
