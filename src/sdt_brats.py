"""SynthStrip signed distance maps for BraTS (CPU, sharded, resumable).

BraTS volumes are already skull-stripped by the challenge. Running the same SynthStrip
(owner's SHA-pinned model, default CSF-inclusive) on the BraTS T1 gives its boundary as a
signed distance map (mm, negative inside); pooled onto the Phase 1 2 mm grid it lets one
SynthStrip boundary be applied to BraTS and IXI alike (common.shared_mask).
Per subject the row records, for each border b, the volume of (sdt < b) and the fraction
of it lying inside the challenge mask; the border is calibrated on train rows only.

Outputs (OUT): sdt/<subject>.npy (96x112x80 fp16), rows/<subject>.json, failed.jsonl.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tempfile
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prep_brats_2mm as pb  # noqa: E402
import prep_ixi_2mm as pi  # noqa: E402
from common import CHANNELS, pool_sdt  # noqa: E402

SIGNATURE = "brats2021-synthstrip-sdt-2mm-v1"
BORDERS = pi.BORDERS


def task(args):
    subject, split, cfg = args
    t0 = time.time()
    try:
        arrs, affine = pb.read_case(subject, CHANNELS)
        support = np.zeros(arrs["t1"].shape, bool)
        for a in arrs.values():
            support |= a != 0
        with tempfile.TemporaryDirectory(dir=cfg["tmp"]) as tmp:
            tmp = Path(tmp)
            t1p = tmp / "t1.nii.gz"
            hdr = nib.Nifti1Header()
            hdr.set_qform(affine, 1)
            hdr.set_sform(affine, 1)
            nib.save(nib.Nifti1Image(arrs["t1"], affine, hdr), t1p)
            pi.skullstrip(t1p, tmp / "mask.nii.gz", tmp, tmp / "sdt.nii.gz")
            sdt = np.asarray(nib.load(tmp / "sdt.nii.gz").dataobj, np.float32)
        assert sdt.shape == support.shape, (subject, sdt.shape)
        rec = dict(subject=subject, split=split, signature=SIGNATURE, t_strip=round(time.time() - t0, 1),
                   challenge_brain_ml=float(support.sum() / 1000.0))
        for b in BORDERS:
            m = sdt < b
            rec[f"ss_ml_b{b:+d}"] = float(m.sum() / 1000.0)
            rec[f"frac_inside_b{b:+d}"] = float((m & support).sum() / max(m.sum(), 1))
            rec[f"challenge_covered_b{b:+d}"] = float((m & support).sum() / max(support.sum(), 1))
        sdt2 = pool_sdt(sdt, cfg["box"], cfg["target"]).astype(np.float16)
        np.save(Path(cfg["out"]) / "sdt" / f"{subject}.npy", sdt2)
        rec["sdt_sha256"] = hashlib.sha256(sdt2.tobytes()).hexdigest()[:16]
        rec["t_total"] = round(time.time() - t0, 1)
        (Path(cfg["out"]) / "rows" / f"{subject}.json").write_text(json.dumps(rec))
        return True, rec
    except Exception as e:  # noqa: BLE001
        return False, dict(subject=subject, error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-1500:])


def _init(tar_path, index, cfg):
    pb._init(tar_path, index)
    pi.CFG.update(cfg)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", default=None)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--crop", required=True)
    ap.add_argument("--tools", default=None)
    ap.add_argument("--skullstrip", choices=["synthstrip", "threshold"], default="synthstrip")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=700)
    args = ap.parse_args(argv)
    t0 = time.time()
    out = Path(args.out)
    for d in ("sdt", "rows"):
        (out / d).mkdir(parents=True, exist_ok=True)
    tar_path = args.tar or str(next(Path("/kaggle/input").rglob("BraTS2021_Training_Data.tar")))
    crop_info = json.loads(Path(args.crop).read_text())
    cfg = dict(box=crop_info["box_1mm"], target=crop_info["target_shape"], out=str(out),
               tmp=args.tmp or tempfile.gettempdir(), skullstrip=args.skullstrip, tools=args.tools)
    if args.skullstrip == "synthstrip":
        tools = args.tools or str(next(Path("/kaggle/input").rglob("synthstrip.1.pt")).parent)
        cfg["tools"] = tools
        for name, h in pi.SYNTHSTRIP_SHA.items():
            assert pi.sha256(Path(tools) / name) == h, f"{name} hash mismatch"
    with open(args.splits) as f:
        rows = [r for r in csv.DictReader(f) if r["source"] == "brats"]
    rows = [r for i, r in enumerate(sorted(rows, key=lambda r: r["subject"])) if i % args.nshards == args.shard]
    if args.limit:
        rows = rows[: args.limit]
    done = {p.stem for p in (out / "rows").glob("*.json") if (out / "sdt" / f"{p.stem}.npy").exists()}
    todo = [(r["subject"], r["split"], cfg) for r in rows if r["subject"] not in done]
    pb.log(f"shard {args.shard}/{args.nshards}: {len(rows)} subjects, {len(todo)} to do")
    index = pb.index_tar(tar_path)
    n_ok = n_fail = 0
    stopped = False
    with Pool(args.workers, initializer=_init, initargs=(tar_path, index, cfg)) as pool:
        for ok, rec in pool.imap_unordered(task, todo):
            if ok:
                n_ok += 1
                pb.log(f"{n_ok + n_fail}/{len(todo)} {rec['subject']} {rec['t_total']}s "
                       f"inside(b-2)={rec['frac_inside_b-2']:.3f}")
            else:
                n_fail += 1
                pb.log("FAIL", rec["subject"], rec["error"])
                with open(out / "failed.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
            if time.time() - t0 > args.max_minutes * 60:
                stopped = True
                pool.terminate()
                break
    summary = dict(signature=SIGNATURE, shard=args.shard, nshards=args.nshards, n_jobs=len(rows),
                   n_done=len(list((out / "rows").glob("*.json"))), n_failed_this_run=n_fail,
                   stopped_early=stopped, elapsed_s=round(time.time() - t0))
    (out / f"summary_shard{args.shard}.json").write_text(json.dumps(summary, indent=2))
    pb.log(summary)
    return summary


if __name__ == "__main__":
    main()
