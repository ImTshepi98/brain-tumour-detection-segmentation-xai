"""Phase 8: turn one raw BraTS-format case into the model's input (for the GUI).

Input: skull-stripped NIfTI volumes on the BraTS/SRI24 1 mm grid (240x240x155), as in the
BraTS 2021 release: T1 and T2 required, T1ce and FLAIR optional (missing = zero channel, so
only detection runs), seg optional. Other data must first be brought to this space; that is
not done here.
Steps (same functions as Phase 1 and the shared boundary): per-modality z-score over non-zero
voxels, crop to the train+val box, 2x brain-masked pooling, pad to 96x112x80, then the shared
SynthStrip boundary (`common.shared_mask` at the model's border) from a SynthStrip distance map
(computed with the pinned SynthStrip if --tools is given, or passed with --sdt).
Output: .npz with x (fp16, 4 channels), y (uint8 label bits, zero without seg), meta (json).

  python src/preprocess_one.py --t1 a_t1.nii.gz --t2 a_t2.nii.gz [--t1ce ..] [--flair ..] [--seg ..] \
      --tools <dir with mri_synthstrip.py + synthstrip.1.pt> --border 0 --out data_local/cases/a.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CHANNELS, crop, pad_to, pool_image, pool_labels, pool_sdt, seg_to_bits,  # noqa: E402
                    shared_mask, zscore_nonzero)

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = (240, 240, 155)


def synthstrip_sdt(t1_path, tools):
    import nibabel as nib
    import prep_ixi_2mm as pi
    pi.CFG.update(tools=str(tools), skullstrip="synthstrip")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pi.skullstrip(Path(t1_path).resolve(), tmp / "mask.nii.gz", tmp, tmp / "sdt.nii.gz")
        return np.asarray(nib.load(tmp / "sdt.nii.gz").dataobj, np.float32)


def preprocess(vols, crop_cfg, border=None, sdt=None, seg=None, expected=EXPECTED):
    """vols: dict modality -> 1 mm array (t1 and t2 required). Returns (x fp16, bits uint8, meta)."""
    assert vols.get("t1") is not None and vols.get("t2") is not None, "T1 and T2 are required"
    box, target = crop_cfg["box_1mm"], crop_cfg["target_shape"]
    support = np.zeros(expected, bool)
    xs = []
    for m in CHANNELS:
        a = vols.get(m)
        if a is None:
            xs.append(np.zeros(expected, np.float32))
            continue
        a = np.asarray(a, np.float32)
        assert a.shape == tuple(expected), f"{m}: shape {a.shape}, expected {tuple(expected)} (BraTS 1 mm grid)"
        support |= a != 0
        xs.append(zscore_nonzero(a)[0])
    x = np.stack(xs)
    bits = seg_to_bits(np.asarray(seg)) if seg is not None else np.zeros(expected, np.uint8)
    xp, brain2 = pool_image(crop(x, box), crop(support, box))
    bp = pool_labels(crop(bits, box))
    bp[~brain2] &= 0
    xp, bp = pad_to(xp, target), pad_to(bp, target)
    meta = dict(modalities=[m for m in CHANNELS if vols.get(m) is not None], has_seg=seg is not None,
                shared_boundary=False, border=border)
    if border is not None:
        assert sdt is not None, "the model uses the shared SynthStrip boundary: give --tools or --sdt"
        xp, _ = shared_mask(xp, pool_sdt(sdt, box, target), border, bp if seg is not None else None)
        meta["shared_boundary"] = True
    return xp.astype(np.float16), bp.astype(np.uint8), meta


def main(argv=None):
    import nibabel as nib
    ap = argparse.ArgumentParser()
    for m in ("t1", "t2"):
        ap.add_argument(f"--{m}", required=True)
    for m in ("t1ce", "flair", "seg", "sdt"):
        ap.add_argument(f"--{m}")
    ap.add_argument("--tools", help="dir with mri_synthstrip.py and synthstrip.1.pt (computes the distance map)")
    ap.add_argument("--border", default=None, help="mm; or read from --ckpt")
    ap.add_argument("--ckpt", help="take the border from this checkpoint's config")
    ap.add_argument("--crop", default=str(ROOT / "data_splits/crop_brats2mm.json"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.ckpt:
        import torch
        border = torch.load(args.ckpt, map_location="cpu", weights_only=False)["config"]["border"]
    else:
        border = None if args.border in (None, "null") else float(args.border)
    vols = {m: np.asarray(nib.load(getattr(args, m)).dataobj, np.float32) for m in CHANNELS if getattr(args, m)}
    seg = np.asarray(nib.load(args.seg).dataobj) if args.seg else None
    sdt = None
    if border is not None:
        sdt = (np.asarray(nib.load(args.sdt).dataobj, np.float32) if args.sdt
               else synthstrip_sdt(args.t1, args.tools) if args.tools else None)
    x, y, meta = preprocess(vols, json.loads(Path(args.crop).read_text()), border, sdt, seg)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, x=x, y=y, meta=json.dumps(meta))
    print(json.dumps(meta))
    return meta


if __name__ == "__main__":
    main()
