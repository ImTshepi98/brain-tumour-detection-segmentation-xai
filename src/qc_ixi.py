"""Phase 2 QC (local): merge IXI shard rows, flag subjects, draw panels for the owner.

Inputs: one or more IXI output dirs (each with rows/*.json, x/*.npy, failed.jsonl), the
BraTS 2 mm dir (x/*.npy for 12 comparison cases), splits.csv.
Outputs (reports/phase2/): ixi_manifest.csv, ixi_flagged.csv, ixi_failed.txt,
panels_ixi.png, panels_brats.png, histograms.png, qc_summary.json.

Flags (any): mask Dice with template < 0.85; warped brain volume or native brain
volume outside median +/- 3 MAD; T1-template correlation < 0.5; |log affine det| > 0.4;
brain voxels outside the crop box > 0.5%; in the owner's 11-subject exclusion list.
Only the owner signs off visual review.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def mad_flags(vals, k=3.0):
    v = np.asarray(vals, float)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826
    return (np.abs(v - med) > k * mad) if mad > 0 else np.zeros(len(v), bool), med, mad


def flag_rows(recs, owner_excluded):
    reasons = {r["subject"]: [] for r in recs}
    for key in ("warped_brain_ml", "native_brain_ml"):
        fl, med, mad = mad_flags([r[key] for r in recs])
        for r, f in zip(recs, fl):
            if f:
                reasons[r["subject"]].append(f"{key}={r[key]:.0f} (median {med:.0f}, MAD {mad:.0f})")
    for r in recs:
        s = r["subject"]
        if r["mask_dice_template"] < 0.85:
            reasons[s].append(f"mask_dice_template={r['mask_dice_template']:.3f}")
        if r.get("t1_template_corr") is None or r["t1_template_corr"] < 0.5:
            reasons[s].append(f"t1_template_corr={r.get('t1_template_corr')}")
        if abs(np.log(abs(r["affine_det"]))) > 0.4:
            reasons[s].append(f"affine_det={r['affine_det']:.2f}")
        if r["brain_vox_outside_box"] > 0.005 * r["brain_vox_1mm"]:
            reasons[s].append(f"outside_box={r['brain_vox_outside_box']}")
        if s in owner_excluded:
            reasons[s].append("owner_eligibility_excluded")
    return reasons


def panels(arrs, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(arrs)
    fig, axes = plt.subplots(n, 4, figsize=(10, 2.4 * n))
    axes = np.atleast_2d(axes)
    for i, (s, x) in enumerate(arrs):
        x = x.astype(np.float32)
        d, h, w = x.shape[1:]
        for j, (lab, im) in enumerate((("T1 ax", x[0][:, :, w // 2]), ("T2 ax", x[2][:, :, w // 2]),
                                       ("T1 cor", x[0][:, h // 2, :]), ("T1 sag", x[0][d // 2, :, :]))):
            axes[i, j].imshow(im.T, origin="lower", cmap="gray", vmin=-2.5, vmax=3)
            axes[i, j].set_title(f"{s} {lab}", fontsize=7)
            axes[i, j].axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def histograms(ixi, brats, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    bins = np.linspace(-4, 6, 81)
    for ax, ch, name in ((axes[0], 0, "T1"), (axes[1], 2, "T2")):
        for group, col, lab in ((ixi, "tab:blue", "IXI"), (brats, "tab:red", "BraTS")):
            for k, (_, x) in enumerate(group):
                v = x[ch].astype(np.float32)
                v = v[v != 0]
                ax.hist(v, bins=bins, histtype="step", density=True, color=col, alpha=0.4,
                        label=lab if k == 0 else None)
        ax.set_title(f"{name} z-scored brain intensities (2 mm)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def pick(subjects, n, salt):
    return sorted(subjects, key=lambda s: hashlib.sha256((salt + s).encode()).hexdigest())[:n]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ixi-dirs", nargs="+", required=True)
    ap.add_argument("--brats-dir", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--owner-excluded", default="legacy/derived/ixi_excluded_11.json")
    ap.add_argument("--out", default="reports/phase2")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(args.splits) as f:
        split_rows = {r["subject"]: r for r in csv.DictReader(f)}
    expected = sorted(s for s, r in split_rows.items() if r["source"] == "ixi")
    recs, failed, xdir = {}, {}, {}
    for d in map(Path, args.ixi_dirs):
        for p in (d / "rows").glob("*.json"):
            r = json.loads(p.read_text())
            # a row is written only after its array is saved (x_sha256), so arrays may stay on Kaggle;
            # any array present locally is checked against its hash
            xp = d / "x" / f"{r['subject']}.npy"
            if xp.exists():
                x16 = np.load(xp)
                if hashlib.sha256(x16.tobytes()).hexdigest()[:16] != r.get("x_sha256"):
                    raise ValueError(f"x_sha256 mismatch for {r['subject']}")
                xdir[r["subject"]] = d / "x"
            recs[r["subject"]] = r
        if (d / "failed.jsonl").exists():
            for line in (d / "failed.jsonl").read_text().splitlines():
                fr = json.loads(line)
                failed[fr["subject"]] = fr["error"]
    failed = {s: e for s, e in failed.items() if s not in recs}  # later reruns may have fixed it
    missing = [s for s in expected if s not in recs and s not in failed]
    owner_ex = {r["subject"] for r in json.loads(Path(args.owner_excluded).read_text())} \
        if Path(args.owner_excluded).exists() else set()
    rl = [recs[s] for s in sorted(recs)]
    reasons = flag_rows(rl, owner_ex) if rl else {}
    with open(out / "ixi_manifest.csv", "w", newline="") as f:
        cols = ["id", "source", "site", "split", "label", "has_seg", "flagged"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for s in sorted(recs):
            w.writerow(dict(id=s, source="ixi", site=split_rows[s]["site"], split=split_rows[s]["split"],
                            label=0, has_seg=0, flagged=int(bool(reasons[s]))))
    with open(out / "ixi_flagged.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "split", "site", "reasons"])
        for s in sorted(reasons):
            if reasons[s]:
                w.writerow([s, split_rows[s]["split"], split_rows[s]["site"], "; ".join(reasons[s])])
    (out / "ixi_failed.txt").write_text("".join(f"{s}\t{e}\n" for s, e in sorted(failed.items())))
    keys = sorted({k for r in rl for k in r if not isinstance(r[k], (list, dict))})
    with open(out / "ixi_case_qc.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rl)
    # panels: 12 IXI (train/val only) + 12 BraTS (train/val only); test stays unseen
    ixi_tv = [s for s in recs if split_rows[s]["split"] != "test" and s in xdir]
    brats_tv = [p.stem for p in Path(args.brats_dir, "x").glob("*.npy")
                if split_rows.get(p.stem, {}).get("split") in ("train", "val")]
    ixi_pick = [(s, np.load(xdir[s] / f"{s}.npy")) for s in pick(ixi_tv, 12, "ixi")]
    br_pick = [(s, np.load(Path(args.brats_dir, "x", f"{s}.npy"))) for s in pick(brats_tv, 12, "brats")]
    if ixi_pick:
        panels(ixi_pick, "IXI 2 mm (BraTS-like) - middle slices", out / "panels_ixi.png")
    if br_pick:
        panels(br_pick, "BraTS 2 mm - middle slices", out / "panels_brats.png")
    if ixi_pick and br_pick:
        histograms(ixi_pick, br_pick, out / "histograms.png")
    summary = dict(expected=len(expected), passed_arrays=len(recs), failed=len(failed), missing=len(missing),
                   flagged=sum(bool(v) for v in reasons.values()),
                   accept_500=len(recs) >= 500, fail_rate=len(failed) / max(len(expected), 1),
                   median_t_total=float(np.median([r["t_total"] for r in rl])) if rl else None,
                   median_mask_dice=float(np.median([r["mask_dice_template"] for r in rl])) if rl else None,
                   by_split={sp: sum(split_rows[s]["split"] == sp for s in recs) for sp in ("train", "val", "test")},
                   panels_ixi=[s for s, _ in ixi_pick], panels_brats=[s for s, _ in br_pick],
                   visual_review_complete=False)
    (out / "qc_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
