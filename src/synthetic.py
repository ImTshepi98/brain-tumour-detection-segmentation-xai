"""Tiny synthetic data for local CPU dry runs (PLAN.md rule 3).

Two outputs:
  * make_case / make_cohort: in-memory 4-channel volumes + label bits, shaped like
    the 2 mm arrays the training code consumes (default 4x32x32x32).
  * write_fake_brats_tar: a miniature BraTS2021_Training_Data.tar (NIfTI members,
    same naming as the Kaggle dataset) plus a scope json and modality_stats.csv,
    so prep scripts can be dry-run end to end.

Channel order is fixed: [T1, T1ce, T2, FLAIR]. Label bits: bit0 ET, bit1 TC, bit2 WT.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import tarfile
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import bits_to_channels, seg_to_bits  # noqa: F401  (re-exported for tests)

MODALITIES = ("t1", "t1ce", "t2", "flair")
SEED = 42


def _ellipsoid(shape, centre, radii):
    zz, yy, xx = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    d = sum(((g - c) / r) ** 2 for g, c, r in zip((zz, yy, xx), centre, radii))
    return d <= 1.0


def native_seg(shape, rng, lateral=False):
    """BraTS-native labels inside a brain: 1 NCR, 2 ED, 4 ET (0 elsewhere). lateral=True puts a smaller
    tumour in one hemisphere (axis 0), so the other hemisphere is tumour-free (PLAN2 dry runs)."""
    seg = np.zeros(shape, np.uint8)
    c = [s // 2 + int(rng.integers(-s // 8, s // 8 + 1)) for s in shape]
    r = max(2, min(shape) // 6)
    if lateral:
        r = max(2, min(shape) // 8)
        c[0] = int(round(shape[0] * (0.22 if rng.random() < 0.5 else 0.78)))
    seg[_ellipsoid(shape, c, (r * 1.6,) * 3)] = 2
    seg[_ellipsoid(shape, c, (r * 1.0,) * 3)] = 4
    seg[_ellipsoid(shape, c, (r * 0.5,) * 3)] = 1
    return seg


def make_case(shape=(32, 32, 32), tumour=True, healthy_t1t2_only=None, rng=None, lateral=False):
    """Return (x float16 (4,*shape), bits uint8 shape, brain bool shape).

    healthy_t1t2_only defaults to (not tumour): IXI-like cases have channels 2, 3 zero.
    """
    rng = rng if rng is not None else np.random.default_rng(SEED)
    if healthy_t1t2_only is None:
        healthy_t1t2_only = not tumour
    brain = _ellipsoid(shape, [s / 2 for s in shape], [s * 0.42 for s in shape])
    x = np.zeros((4, *shape), np.float32)
    for ch in range(4):
        v = rng.normal(0.0, 1.0, shape).astype(np.float32)
        x[ch][brain] = v[brain]
    seg = native_seg(shape, rng, lateral) * brain if tumour else np.zeros(shape, np.uint8)
    if tumour:
        x[:, seg > 0] += 2.0
    if healthy_t1t2_only:
        x[1] = 0.0
        x[3] = 0.0
    return x.astype(np.float16), seg_to_bits(seg), brain


def make_cohort(n_brats=6, n_ixi=6, shape=(32, 32, 32), seed=SEED, lateral=False):
    """List of dicts with id/source/label/has_seg/x/bits, BraTS first then IXI."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_brats):
        x, bits, _ = make_case(shape, tumour=True, rng=rng, lateral=lateral)
        out.append(dict(id=f"BraTS2021_{i:05d}", source="brats", label=1, has_seg=1, x=x, bits=bits))
    for i in range(n_ixi):
        x, bits, _ = make_case(shape, tumour=False, rng=rng)
        out.append(dict(id=f"IXI{i:03d}-Guys-{i:04d}", source="ixi", label=0, has_seg=0, x=x, bits=bits))
    return out


def write_fake_brats_tar(out_dir, n_cases=6, shape=(32, 32, 24), n_trainval=4, seed=SEED):
    """Write a miniature BraTS tar + scope json + modality_stats.csv into out_dir.

    The real volumes are (240, 240, 155); prep scripts take the expected shape as a
    parameter so the dry run can use this small one.
    """
    import nibabel as nib

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    ids = [f"BraTS2021_{i:05d}" for i in range(n_cases)]
    stats_rows = []
    tar_path = out_dir / "BraTS2021_Training_Data.tar"
    with tarfile.open(tar_path, "w") as tf:
        for cid in ids:
            brain = _ellipsoid(shape, [s / 2 for s in shape], [s * 0.42 for s in shape])
            seg = native_seg(shape, rng) * brain
            vols = {}
            for m in MODALITIES:
                v = np.zeros(shape, np.float32)
                v[brain] = rng.normal(500.0, 100.0, int(brain.sum()))
                v[seg > 0] += 300.0
                vols[m] = v
                nz = v[v != 0]
                stats_rows.append(dict(subject_id=cid, modality=m, mean=float(nz.mean()), std=float(nz.std())))
            vols["seg"] = seg.astype(np.uint8)
            for m, v in vols.items():
                img = nib.Nifti1Image(v, np.diag([-1.0, -1.0, 1.0, 1.0]))
                buf = io.BytesIO()
                fh = nib.FileHolder(fileobj=buf)
                img.to_file_map({"image": fh, "header": fh})
                import gzip
                data = gzip.compress(buf.getvalue())
                info = tarfile.TarInfo(f"{cid}/{cid}_{m}.nii.gz")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
    scope = {"subjects": ids[:n_trainval]}
    (out_dir / "brats_trainval_scope.json").write_text(json.dumps(scope, indent=2))
    with open(out_dir / "modality_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject_id", "modality", "mean", "std"])
        w.writeheader()
        w.writerows(r for r in stats_rows if r["subject_id"] in scope["subjects"])
    return tar_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_local/synthetic")
    ap.add_argument("--n-cases", type=int, default=6)
    args = ap.parse_args()
    p = write_fake_brats_tar(args.out, n_cases=args.n_cases)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()


def signed_distance(brain, spacing=2.0):
    """SynthStrip-like signed distance map (mm, negative inside) of a boolean mask."""
    from scipy import ndimage as ndi
    return (spacing * (ndi.distance_transform_edt(~brain) - ndi.distance_transform_edt(brain))).astype(np.float16)


def write_fake_2mm(out_dir, n_brats=8, n_ixi=8, shape=(32, 32, 32), seed=SEED, lateral=False):
    """Write a miniature copy of the Kaggle 2 mm layout for training dry runs:
    brats2mm/{x,y}/<id>.npy + crop.json, ixi2mm/{x,sdt}/<id>.npy + template.json,
    sdt_brats/sdt/<id>.npy + summary_shard0.json, splits.csv (train/val/test per cohort).
    Returns dict(brats_dir, ixi_dir, brats_sdt_dir, splits)."""
    out = Path(out_dir)
    bd, idir, sd = out / "brats2mm", out / "ixi2mm", out / "sdt_brats"
    for d in (bd / "x", bd / "y", idir / "x", idir / "sdt", sd / "sdt"):
        d.mkdir(parents=True, exist_ok=True)
    (bd / "crop.json").write_text(json.dumps({"target": list(shape)}))
    (idir / "template.json").write_text("{}")
    (sd / "summary_shard0.json").write_text("{}")
    rows = []
    for i, c in enumerate(make_cohort(n_brats, n_ixi, shape, seed, lateral)):
        split = ("train", "train", "val", "test")[i % 4]
        brain = (c["x"] != 0).any(0)
        if c["source"] == "brats":
            np.save(bd / "x" / f"{c['id']}.npy", c["x"])
            np.save(bd / "y" / f"{c['id']}.npy", c["bits"])
            np.save(sd / "sdt" / f"{c['id']}.npy", signed_distance(brain))
            site = "unknown"
        else:
            np.save(idir / "x" / f"{c['id']}.npy", c["x"])
            np.save(idir / "sdt" / f"{c['id']}.npy", signed_distance(brain))
            site = ("Guys", "HH", "IOP")[i % 3]
        rows.append(dict(subject=c["id"], id=c["id"], source=c["source"], site=site, split=split,
                         label=c["label"], has_seg=c["has_seg"], owner_excluded=0))
    with open(out / "splits.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return dict(brats_dir=str(bd), ixi_dir=str(idir), brats_sdt_dir=str(sd), splits=str(out / "splits.csv"))
