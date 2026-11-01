"""PLAN2 P4/P6: evaluate a PLAN2 model on val, or once on test, with the pre-registered checks (tag plan2-prereg).

Val: thresholds chosen on val by the worst-group balanced-accuracy rule, for the detection head (probability)
and the segmentation-derived detector (pass-B WT volume, v*). Test: --confirm-test and --threshold-from
<results_val.json>; test_lock.json is written before any test array is read; a second read is refused.
Pairing files (pairs_<split>.csv) are written by the same seeded function before use.
Outputs: cases_<split>.csv, results_<split>.json, <split>_checks.md, s4/s7/s8 csv, xai/ (with --xai).

  python src/eval_p2.py --ckpt runs/p2-main/best.pt --split val --out runs/p2-main/eval --xai ...
  python src/eval_p2.py --ckpt ... --split test --confirm-test --threshold-from <val>/results_val.json ...
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datasets as ds  # noqa: E402
import hemi  # noqa: E402
import metrics  # noqa: E402
import p2_data  # noqa: E402
import p2_metrics as pm  # noqa: E402
import shortcut_checks as sc  # noqa: E402
from evaluate import HD95_MAX, hd95  # noqa: E402,F401
from p2_model import build  # noqa: E402


def load(path, device):
    st = torch.load(path, map_location=device, weights_only=False)
    model = build(st["config"]).to(device)
    model.load_state_dict(st["model"])
    return model.eval(), st


@torch.no_grad()
def predict(model, ev, device, bs=4):
    recs = []
    for s in range(0, len(ev), bs):
        idx = list(range(s, min(s + bs, len(ev))))
        x = torch.from_numpy(np.stack([np.asarray(ev.x[i], np.float32) for i in idx])).to(device)
        bits = torch.from_numpy(np.stack([ev.bits[i] for i in idx])).to(device)
        target = torch.stack([(bits >> k) & 1 for k in range(3)], 1).bool()
        lg, wt, *_ = model.forward_det(x)
        wtb = torch.sigmoid(wt) > 0.5
        dice_b = metrics.dice_per_case(torch.stack([wtb] * 3, 1), torch.stack([target[:, 2]] * 3, 1))[:, 0]
        brats = [j for j, i in enumerate(idx) if ev.rows[i]["dataset"] == "brats"]
        segA = {}
        if brats:
            pred = metrics.postprocess(model.forward_seg(x[brats]))
            d = metrics.dice_per_case(pred, target[brats])
            for k, j in enumerate(brats):
                segA[j] = (pred[k], d[k])
        for j, i in enumerate(idx):
            r = dict(ev.rows[i])
            r.update(logit=float(lg[j]), prob=float(torch.sigmoid(lg[j])), wt_ml_b=float(wtb[j].sum()) * 8 / 1000)
            if r["label"] == 1:
                r["dsc_WT_passB"] = float(dice_b[j])
            if j in segA:
                p, d = segA[j]
                r["wt_ml_a"] = float(p[2].sum()) * 8 / 1000
                if r["cell"] in ("B+r", "B+p"):
                    for c, name in enumerate(metrics.REGIONS):
                        r[f"dsc_{name}"] = float(d[c])
                    if r["cell"] == "B+r":
                        h = hd95(p.cpu(), target[j].cpu())
                        for c, name in enumerate(metrics.REGIONS):
                            r[f"hd95_{name}"] = float(h[c])
            recs.append(r)
    return recs


def detection_table(recs, key, thr):
    out = {}
    for name, sel in (("all", lambda r: True), ("brats", lambda r: r["dataset"] == "brats"),
                      ("ixi", lambda r: r["dataset"] == "ixi"),
                      ("real (B+r vs B-r, I-r)", lambda r: r["cell"] in ("B+r", "B-r", "I-r"))):
        rs = [r for r in recs if sel(r)]
        if rs and len({r["label"] for r in rs}) == 2:
            d = metrics.detection([r["label"] for r in rs], [r[key] for r in rs], thr)
            d["balanced_accuracy"] = 0.5 * (d["recall"] + d["specificity"])
            out[name] = d
    out["worst_group_ba"] = pm.worst_group_ba(recs, [r[key] for r in recs], thr)
    out["cells"] = {c: dict(n=n, positive_rate=v) for c, (n, v) in pm.cell_rates(recs, [r[key] for r in recs], thr).items()}
    return out


def whole_scan(recs, key, thr):
    """Scan positive if any kept real hemisphere is positive: recall on BraTS subjects with a B+r hemisphere,
    specificity on IXI subjects (both hemispheres I-r). Whole-scan BraTS negatives do not exist."""
    subj = {}
    for r in recs:
        if r["cell"] in ("B+r", "I-r"):
            subj.setdefault((r["dataset"], r["subject"]), []).append(r[key] >= thr)
    br = [any(v) for (d, _), v in subj.items() if d == "brats"]
    ix = [not any(v) for (d, _), v in subj.items() if d == "ixi"]
    return dict(brats_scans=len(br), recall=float(np.mean(br)) if br else float("nan"),
                ixi_scans=len(ix), specificity=float(np.mean(ix)) if ix else float("nan"))


def seg_table(recs):
    out = {}
    for cell in ("B+r", "B+p", "I+p"):
        rs = [r for r in recs if r["cell"] == cell]
        if not rs:
            continue
        o = {}
        for k in ("dsc_ET", "dsc_TC", "dsc_WT", "hd95_ET", "hd95_TC", "hd95_WT", "dsc_WT_passB"):
            v = np.array([r[k] for r in rs if k in r])
            if len(v):
                o[k] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
                            median=float(np.median(v)), n=len(v))
        out[cell] = o
    for cell in ("B-r", "I-r", "B-s", "I-s"):
        v = [r["wt_ml_b"] for r in recs if r["cell"] == cell]
        if v:
            out[f"fp_volume_passB_{cell}"] = dict(median=float(np.median(v)), mean=float(np.mean(v)),
                                                  share_over_1ml=float(np.mean(np.asarray(v) > 1)))
    return out


@torch.no_grad()
def inference_time(model, ev, device, n=10, warmup=3):
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
    xs = [torch.from_numpy(np.asarray(ev.x[i], np.float32))[None].to(device) for i in range(min(n + warmup, len(ev)))]
    ts = []
    for k, x in enumerate(xs):
        sync()
        t = time.perf_counter()
        model.forward_det(x)
        model.forward_seg(x)
        sync()
        if k >= warmup:
            ts.append((time.perf_counter() - t) * 1000)
    return dict(ms_mean=float(np.mean(ts)) if ts else float("nan"), ms_sd=float(np.std(ts)) if ts else float("nan"),
                n=len(ts), passes="detection + pass B, then pass A", device=str(device))


def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)


def checks_md(res, path):
    lines = [f"# PLAN2 checks: {res['run']} on {res['split']}", "",
             f"Pre-registration: tag `plan2-prereg`; checkpoint epoch {res['epoch']}; protocol {res['protocol_hash'][:12]}.",
             f"Thresholds: head {res['threshold_head']:.6g}, segmentation-derived v* {res['threshold_seg']:.6g} mL "
             f"({res['threshold_rule']}).", ""]
    for det in ("head", "seg"):
        v = res["verdict"][det]
        lines += [f"## Detector: {'detection head' if det == 'head' else 'segmentation-derived (pass-B WT volume)'}",
                  "", f"**Verdict: {'shortcut-controlled' if v['shortcut_controlled'] else 'NOT shortcut-controlled'}**"
                  + (f" (failed MUST: {', '.join(v['failed_must'])})" if v["failed_must"] else ""), "",
                  "| Check | Tier | Result | Rule | Value |", "|---|---|---|---|---|"]
        for c in res["checks"][det]:
            mark = "n/a" if c["passed"] is None else ("PASS" if c["passed"] else "FAIL")
            val = json.dumps(c["value"], default=lambda o: round(o, 4) if isinstance(o, float) else str(o))
            lines.append(f"| {c['id']} | {c['tier']} | {mark} | {c['rule']} | `{val}` |")
        lines.append("")
    Path(path).write_text("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--confirm-test", action="store_true")
    ap.add_argument("--threshold-from")
    ap.add_argument("--brats-dir")
    ap.add_argument("--ixi-dirs", nargs="*", default=[])
    ap.add_argument("--brats-sdt-dirs", nargs="*", default=[])
    ap.add_argument("--splits")
    ap.add_argument("--out", required=True)
    ap.add_argument("--xai", action="store_true", help="Grad-CAM++ / occlusion / SHAP / LIME comparison")
    ap.add_argument("--xai-small", action="store_true", help="dry runs: 1 case per cell, 1 sanity case")
    ap.add_argument("--s7-cases", type=int, default=sc.S7_MAX_CASES)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-count-check", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, st = load(args.ckpt, device)
    cfg = st["config"]
    m, width, border = cfg["midline"], cfg["width"], cfg["border"]
    split = args.split
    if split == "test":
        assert args.confirm_test, "test is locked: pass --confirm-test (read once per model)"
        assert args.threshold_from, "test needs --threshold-from <results_val.json>"
        lock = out / "test_lock.json"
        assert not lock.exists(), f"test already evaluated ({lock}); refusing a second read"
        val = json.loads(Path(args.threshold_from).read_text())
        assert val["protocol_hash"] == st["protocol_hash"] and val["epoch"] == st["epoch"], "val results from another checkpoint"
        thr_h, thr_s = val["threshold_head"], val["threshold_seg"]
        lock.write_text(json.dumps(dict(ckpt=str(args.ckpt), protocol_hash=st["protocol_hash"], epoch=st["epoch"],
                                        threshold_head=thr_h, threshold_seg=thr_s,
                                        time=time.strftime("%Y-%m-%d %H:%M:%S")), indent=2))
    rows = ds.read_splits(args.splits)
    ds.check_splits(rows, None if args.no_count_check else ds.EXPECTED)
    load_kw = dict(brats_dir=args.brats_dir, ixi_dirs=args.ixi_dirs, brats_sdt_dirs=args.brats_sdt_dirs, border=border)
    hs = hemi.HemiSet.load(rows, (split,), m, width=width, allow_test=split == "test", **load_kw)
    pairs = p2_data.fixed_pairs(hs, split)
    p2_data.write_pairs(pairs, out / f"pairs_{split}.csv")
    ev = p2_data.EvalSet(hs, pairs)
    print(f"{split}: {len(ev)} hemisphere samples; cells "
          f"{ {c: sum(r['cell'] == c for r in ev.rows) for c in hemi.CELLS} }", flush=True)
    recs = predict(model, ev, device)
    if split == "val":
        thr_h, _ = pm.choose_threshold(recs, [r["prob"] for r in recs])
        thr_s, _ = pm.choose_threshold(recs, [r["wt_ml_b"] for r in recs])
        rule = "worst-group balanced accuracy on val"
    else:
        rule = "frozen from val"

    # full scans of tumour subjects for S4 (mirror on the whole scan, then cut again)
    need = {r["subject"] for r in recs if r["cell"] == "B+r"}
    paths = ds.locate([r for r in rows if r["subject"] in need], args.brats_dir, args.ixi_dirs, args.brats_sdt_dirs)
    scans = {s: ds.load_one(paths[s], border) for s in sorted(need)}
    s4 = sc.s4_mirror(model, recs, scans, m, width, device, thr_h, thr_s)
    del scans
    s8 = sc.s8_erosion(model, ev, recs, device)
    s9 = sc.s9_probe(model, ev, recs, device)
    s7_maps = {}
    s7 = sc.s7_localisation(model, ev, recs, thr_h, device, model.enc[-2], args.s7_cases, maps_out=s7_maps)
    checks = dict(head=sc.decide(recs, thr_h, s4, s7, s8, s9, "head"), seg=sc.decide(recs, thr_s, s4, None, s8, None, "seg"))
    res = dict(split=split, run=cfg.get("name"), ckpt=str(args.ckpt), epoch=st["epoch"], protocol_hash=st["protocol_hash"],
               n=len(recs), cells={c: sum(r["cell"] == c for r in recs) for c in hemi.CELLS},
               hemi_counts=hs.counts, midline=m, width=width, border=border,
               threshold_head=thr_h, threshold_seg=thr_s, threshold_rule=rule,
               detection_head=detection_table(recs, "prob", thr_h), detection_seg=detection_table(recs, "wt_ml_b", thr_s),
               detection_head_at_0_5=detection_table(recs, "prob", 0.5),
               whole_scan_head=whole_scan(recs, "prob", thr_h), whole_scan_seg=whole_scan(recs, "wt_ml_b", thr_s),
               segmentation=seg_table(recs), checks=checks,
               verdict={k: sc.verdict(v) for k, v in checks.items()}, inference=inference_time(model, ev, device))
    write_csv(recs, out / f"cases_{split}.csv")
    write_csv(s4, out / f"s4_{split}.csv")
    write_csv(s7, out / f"s7_{split}.csv")
    write_csv(s8, out / f"s8_{split}.csv")
    if s7_maps:
        np.savez_compressed(out / f"s7_maps_{split}.npz", **{f"{k.replace(':', '_')}__{n}": v
                                                             for k, d in s7_maps.items() for n, v in d.items()})
    if args.xai:
        import p2_xai
        plan = tuple((c, 1) for c in ("B+r", "B+p", "I+p", "B-r", "I-r")) if args.xai_small else None
        res["xai"] = p2_xai.run(model, ev, recs, device, out / "xai", n_sanity=1 if args.xai_small else 5, plan=plan)
    (out / f"results_{split}.json").write_text(json.dumps(res, indent=2, default=float))
    checks_md(res, out / f"{split}_checks.md")
    print(json.dumps(dict(split=split, thresholds=(thr_h, thr_s), verdict=res["verdict"]), default=float), flush=True)
    return res


if __name__ == "__main__":
    main()
