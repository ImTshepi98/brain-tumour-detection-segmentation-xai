"""Pick the shared SynthStrip border from BraTS *train* SDT rows (rule fixed in DECISIONS.md):
the largest b in BORDERS whose median frac_inside (share of the SynthStrip mask inside the
challenge mask) is >= --min-inside. Also reports volumes per border for BraTS train and IXI
train+val (information only; IXI does not enter the rule)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

BORDERS = (1, 0, -1, -2, -3, -4)


def pick_border(rows, min_inside=0.99):
    med = {b: float(np.median([r[f"frac_inside_b{b:+d}"] for r in rows])) for b in BORDERS}
    ok = [b for b in BORDERS if med[b] >= min_inside]
    return (max(ok) if ok else min(BORDERS)), med


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--brats-rows", nargs="+", required=True, help="dirs with rows/*.json from sdt_brats")
    ap.add_argument("--ixi-rows", nargs="+", required=True, help="dirs with rows/*.json from prep_ixi_2mm v2")
    ap.add_argument("--splits", required=True)
    ap.add_argument("--min-inside", type=float, default=0.99)
    ap.add_argument("--out", default="reports/phase3/border_calibration.json")
    args = ap.parse_args(argv)
    with open(args.splits) as f:
        sp = {r["subject"]: r["split"] for r in csv.DictReader(f)}
    load = lambda dirs: [json.loads(p.read_text()) for d in dirs for p in Path(d, "rows").glob("*.json")]
    brats = load(args.brats_rows)
    ixi = load(args.ixi_rows)
    train = [r for r in brats if sp[r["subject"]] == "train"]
    b, med = pick_border(train, args.min_inside)
    vol = {}
    for bb in BORDERS:
        vol[bb] = dict(brats_train_ss_ml=float(np.median([r[f"ss_ml_b{bb:+d}"] for r in train])),
                       brats_train_challenge_covered=float(np.median([r[f"challenge_covered_b{bb:+d}"] for r in train])),
                       ixi_trainval_ml=float(np.median([r[f"brain_ml_b{bb:+d}"] for r in ixi
                                                        if sp[r["subject"]] != "test"])))
    res = dict(rule=f"largest b with median frac_inside >= {args.min_inside} over BraTS train",
               border=b, n_brats_train=len(train), n_brats_rows=len(brats), n_ixi_rows=len(ixi),
               median_frac_inside=med, median_volumes=vol,
               brats_challenge_ml_train=float(np.median([r["challenge_brain_ml"] for r in train])))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
