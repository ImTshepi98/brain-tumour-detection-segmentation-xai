"""Phase 1: BraTS 2021 -> 2 mm whole-volume arrays (CPU).

Pass 1: per-case brain bounding box (union of non-zero voxels over the 4 modalities).
        The crop box is the union over train+val only; test boxes are only measured
        against it (fraction of test brain falling outside is reported, not used).
Pass 2: per case z-score (owner's modality_stats.csv for train/val, same formula
        computed on the case itself for test), crop, pad to even, 2x brain-masked
        average pool (background re-zeroed where brain fraction < 0.5), pool labels
        per region (mean >= 0.5), pad to a multiple of 16, write fp16 x and uint8 bits.

Outputs (OUT): x/<subject>.npy (4,D,H,W) fp16, y/<subject>.npy (D,H,W) uint8 bits,
manifest.csv, crop.json, boxes.json, summary.json, qc/montage.png.
Resumable: cases whose x and y already exist are skipped (row re-read from rows/).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import sys
import tarfile
import time
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CHANNELS, SEED, bbox, crop, pad_box_to_even, pad_to, pool_image,  # noqa: E402
                    pool_labels, round_up, seg_to_bits, union_bbox, zscore_nonzero)

SIGNATURE = "brats2021-2mm-wholevol-v1"
_TAR = None
_INDEX = None


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def index_tar(tar_path):
    """subject -> {mod: (offset_data, size)} from one header walk."""
    idx = {}
    with tarfile.open(tar_path, "r:") as tf:
        for m in tf:
            if not m.isfile():
                continue
            mm = re.search(r"(BraTS2021_\d{5})_(t1|t1ce|t2|flair|seg)\.nii(\.gz)?$", m.name)
            if mm:
                idx.setdefault(mm.group(1), {})[mm.group(2)] = (m.offset_data, m.size, m.name)
    return idx


def _init(tar_path, index):
    global _TAR, _INDEX
    _TAR = open(tar_path, "rb")
    _INDEX = index


def read_case(subject, mods):
    out, affine = {}, None
    for mod in mods:
        off, size, name = _INDEX[subject][mod]
        _TAR.seek(off)
        raw = _TAR.read(size)
        if name.endswith(".gz"):
            raw = gzip.decompress(raw)
        img = nib.Nifti1Image.from_bytes(raw)
        arr = np.asarray(img.dataobj, dtype=np.float32)
        assert np.isfinite(arr).all(), (subject, mod, "nonfinite")
        out[mod] = arr
        if affine is None:
            affine = img.affine
        else:
            assert np.allclose(img.affine, affine, atol=1e-4), (subject, mod, "affine mismatch")
    return out, affine


def task_box(args):
    subject, expected = args
    arrs, affine = read_case(subject, CHANNELS)
    for m, a in arrs.items():
        assert a.shape == tuple(expected), (subject, m, a.shape)
    support = np.zeros(expected, bool)
    for a in arrs.values():
        support |= a != 0
    return subject, dict(box=bbox(support), voxels=int(support.sum()),
                         axcodes="".join(nib.aff2axcodes(affine)), affine=np.round(affine, 4).tolist())


def task_case(args):
    subject, split, stats, box, target, out_dir, expected = args
    arrs, _ = read_case(subject, CHANNELS + ("seg",))
    seg = arrs.pop("seg").astype(np.int16)
    vals = np.unique(seg)
    assert np.isin(vals, [0, 1, 2, 4]).all(), (subject, vals)
    support = np.zeros(expected, bool)
    xs, rec = [], dict(subject=subject, split=split)
    for m in CHANNELS:
        a = arrs[m]
        assert a.shape == tuple(expected), (subject, m, a.shape)
        support |= a != 0
        if stats is not None:
            mean, std = stats[m]
            _, m2, s2 = zscore_nonzero(a)
            rec[f"stat_reldiff_{m}"] = max(abs(m2 - mean) / abs(mean), abs(s2 - std) / std)
            z, _, _ = zscore_nonzero(a, mean, std)
        else:
            z, mean, std = zscore_nonzero(a)
        rec[f"mean_{m}"], rec[f"std_{m}"] = mean, std
        xs.append(z)
    x = np.stack(xs)
    rec["tumour_outside_support"] = int(((seg > 0) & ~support).sum())
    rec["brain_vox_1mm"] = int(support.sum())
    rec["brain_vox_outside_box"] = int(support.sum() - crop(support, box).sum())
    bits = seg_to_bits(seg)
    rec["wt_vox_1mm"] = int((seg > 0).sum())
    rec["wt_vox_outside_box"] = int(rec["wt_vox_1mm"] - (crop(seg, box) > 0).sum())
    xc, sc, bc = crop(x, box), crop(support, box), crop(bits, box)
    xp, brain2 = pool_image(xc, sc)
    bp = pool_labels(bc)
    bp[~brain2] &= 0  # labels only inside the pooled brain
    xp, bp = pad_to(xp, target), pad_to(bp, target)
    assert np.isfinite(xp).all()
    for k, r in enumerate(("et", "tc", "wt")):
        rec[f"{r}_vox_2mm"] = int(((bp >> k) & 1).sum())
    rec["brain_vox_2mm"] = int(brain2.sum())
    x16 = xp.astype(np.float16)
    assert np.isfinite(x16).all(), (subject, "fp16 overflow")
    np.save(Path(out_dir) / "x" / f"{subject}.npy", x16)
    np.save(Path(out_dir) / "y" / f"{subject}.npy", bp.astype(np.uint8))
    rec["x_sha256"] = hashlib.sha256(x16.tobytes()).hexdigest()[:16]
    (Path(out_dir) / "rows" / f"{subject}.json").write_text(json.dumps(rec))
    return rec


def load_splits(path):
    with open(path) as f:
        return [r for r in csv.DictReader(f) if r["source"] == "brats"]


def load_stats(path, id_to_subject):
    """Owner's stats are keyed by hash `id`; the synthetic file by `subject_id`."""
    stats = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            key = r.get("id") or r.get("subject_id")
            stats.setdefault(id_to_subject.get(key, key), {})[r["modality"]] = (float(r["mean"]), float(r["std"]))
    return stats


def montage(out_dir, subjects, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(subjects), 4, figsize=(13, 3.2 * len(subjects)))
    axes = np.atleast_2d(axes)
    colors = {0: "cyan", 1: "red", 2: "yellow"}  # ET, TC, WT
    for i, s in enumerate(subjects):
        x = np.load(Path(out_dir) / "x" / f"{s}.npy").astype(np.float32)
        y = np.load(Path(out_dir) / "y" / f"{s}.npy")
        wt = (y >> 2) & 1
        c = np.array(np.nonzero(wt)).mean(1).astype(int) if wt.any() else np.array(wt.shape) // 2
        views = [("T1ce ax", x[1][:, :, c[2]]), ("FLAIR ax", x[3][:, :, c[2]]),
                 ("T2 cor", x[2][:, c[1], :]), ("T1 sag", x[0][c[0], :, :])]
        labs = [y[:, :, c[2]], y[:, :, c[2]], y[:, c[1], :], y[c[0], :, :]]
        for j, ((title, im), lab) in enumerate(zip(views, labs)):
            ax = axes[i, j]
            ax.imshow(im.T, origin="lower", cmap="gray", vmin=-2, vmax=4)
            for k, col in colors.items():
                m = (lab >> k) & 1
                if m.any():
                    ax.contour(m.T, levels=[0.5], colors=col, linewidths=0.7)
            ax.set_title(f"{s} {title}", fontsize=8)
            ax.axis("off")
    fig.suptitle("BraTS 2 mm arrays; contours ET cyan, TC red, WT yellow (train/val only)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", default=None)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--stats", required=True, help="owner's modality_stats.csv (train/val)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--expected-shape", type=int, nargs=3, default=[240, 240, 155])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N cases per split")
    ap.add_argument("--max-minutes", type=float, default=600)
    ap.add_argument("--pad-multiple", type=int, default=16)
    args = ap.parse_args(argv)
    t0 = time.time()
    np.random.seed(SEED)
    out = Path(args.out)
    for d in ("x", "y", "rows", "qc"):
        (out / d).mkdir(parents=True, exist_ok=True)
    tar_path = args.tar or str(next(Path("/kaggle/input").rglob("BraTS2021_Training_Data.tar")))
    rows = load_splits(args.splits)
    if args.limit:
        keep = []
        for sp in ("train", "val", "test"):
            keep += [r for r in rows if r["split"] == sp][: args.limit]
        rows = keep
    stats = load_stats(args.stats, {r["id"]: r["subject"] for r in rows if r["id"]})
    log("index tar", tar_path)
    index = index_tar(tar_path)
    missing = [r["subject"] for r in rows if r["subject"] not in index or len(index[r["subject"]]) != 5]
    assert not missing, missing[:5]
    for r in rows:
        if r["split"] != "test":
            assert r["subject"] in stats and len(stats[r["subject"]]) == 4, r["subject"]
    log(f"indexed {len(index)} subjects; processing {len(rows)}")

    with Pool(args.workers, initializer=_init, initargs=(tar_path, index)) as pool:
        box_path = out / "boxes.json"
        boxes = json.loads(box_path.read_text()) if box_path.exists() else {}
        todo = [(r["subject"], args.expected_shape) for r in rows if r["subject"] not in boxes]
        for i, (s, b) in enumerate(pool.imap_unordered(task_box, todo, chunksize=2)):
            boxes[s] = b
            if (i + 1) % 50 == 0:
                log(f"pass1 {i + 1}/{len(todo)}")
                box_path.write_text(json.dumps(boxes))
        box_path.write_text(json.dumps(boxes))
        axc = {b["axcodes"] for b in boxes.values()}
        assert len(axc) == 1, axc
        tv = [boxes[r["subject"]]["box"] for r in rows if r["split"] != "test"]
        box = pad_box_to_even(union_bbox(tv), args.expected_shape)
        pooled = [(hi - lo) // 2 for lo, hi in box]
        target = [round_up(n, args.pad_multiple) for n in pooled]
        crop_info = dict(signature=SIGNATURE, box_1mm=box, pooled_shape=pooled, target_shape=target,
                         pad="end", pool="2x brain-masked mean, re-zero if brain fraction < 0.5",
                         label_pool="per-region mean >= 0.5, nesting ET<=TC<=WT enforced",
                         axcodes=axc.pop(), affine_example=next(iter(boxes.values()))["affine"],
                         box_from="union of train+val brain boxes", n_box_cases=len(tv))
        (out / "crop.json").write_text(json.dumps(crop_info, indent=2))
        log("crop", crop_info)

        done = {p.stem for p in (out / "rows").glob("*.json")
                if (out / "x" / f"{p.stem}.npy").exists() and (out / "y" / f"{p.stem}.npy").exists()}
        jobs = [(r["subject"], r["split"], stats.get(r["subject"]) if r["split"] != "test" else None,
                 box, target, str(out), args.expected_shape) for r in rows if r["subject"] not in done]
        log(f"pass2: {len(jobs)} to do, {len(done)} already done")
        stopped = False
        for i, rec in enumerate(pool.imap_unordered(task_case, jobs, chunksize=1)):
            if (i + 1) % 50 == 0:
                log(f"pass2 {i + 1}/{len(jobs)} elapsed {time.time() - t0:.0f}s")
            if (time.time() - t0) / 60 > args.max_minutes:
                log("max-minutes reached; stopping (resume to continue)")
                stopped = True
                pool.terminate()
                break

    recs = [json.loads(p.read_text()) for p in sorted((out / "rows").glob("*.json"))]
    by = {r["subject"]: r for r in recs}
    man = []
    for r in rows:
        if r["subject"] in by:
            man.append(dict(id=r["subject"], owner_id=r["id"], source="brats", site="unknown",
                            split=r["split"], label=1, has_seg=1))
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(man[0]))
        w.writeheader()
        w.writerows(man)
    keys = sorted({k for r in recs for k in r})
    with open(out / "case_qc.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(recs)

    def agg(split_filter):
        rs = [r for r in recs if split_filter(r["split"])]
        if not rs:
            return {}
        return dict(n=len(rs),
                    max_stat_reldiff=max((max(v for k, v in r.items() if k.startswith("stat_reldiff")) for r in rs
                                          if any(k.startswith("stat_reldiff") for k in r)), default=None),
                    tumour_outside_support=sum(r["tumour_outside_support"] for r in rs),
                    cases_brain_outside_box=sum(r["brain_vox_outside_box"] > 0 for r in rs),
                    max_brain_frac_outside_box=max(r["brain_vox_outside_box"] / r["brain_vox_1mm"] for r in rs),
                    cases_wt_outside_box=sum(r["wt_vox_outside_box"] > 0 for r in rs),
                    cases_wt_empty_2mm=sum(r["wt_vox_2mm"] == 0 for r in rs),
                    cases_et_empty_2mm=sum(r["et_vox_2mm"] == 0 for r in rs))

    shapes = {tuple(np.load(out / "x" / f"{s}.npy", mmap_mode="r").shape) for s in by}
    size = sum(p.stat().st_size for d in ("x", "y") for p in (out / d).glob("*.npy"))
    summary = dict(signature=SIGNATURE, n_expected=len(rows), n_done=len(recs), complete=len(recs) == len(rows),
                   stopped_early=stopped, shapes=[list(s) for s in shapes], equal_shapes=len(shapes) == 1,
                   bytes=size, elapsed_s=round(time.time() - t0), train_val=agg(lambda s: s != "test"),
                   test=agg(lambda s: s == "test"), test_note="test arrays/labels stored only; not inspected",
                   visual_review_complete=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    log("summary", json.dumps(summary))
    if summary["complete"]:  # per-case rows are only needed for resume; case_qc.csv keeps them
        for p in (out / "rows").glob("*.json"):
            p.unlink()
        (out / "rows").rmdir()
    tvs = sorted((r["subject"] for r in recs if r["split"] != "test"),
                 key=lambda s: hashlib.sha256(s.encode()).hexdigest())[:6]
    if tvs:
        montage(out, tvs, out / "qc" / "montage.png")
    return summary


if __name__ == "__main__":
    main()
