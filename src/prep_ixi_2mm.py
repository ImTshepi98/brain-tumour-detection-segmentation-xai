"""Phase 2: IXI T1/T2 -> BraTS-like 2 mm arrays (CPU, sharded, resumable).

Per subject (all 577 paired, no age filter):
  1. SynthStrip on native T1 (owner's SHA-pinned tool) -> brain mask.
  2. ANTs rigid T2 -> T1 (whole head), T2 resampled to T1 space, both masked.
  3. ANTs affine T1-brain -> BraTS-space T1 template (built here from BraTS train cases).
     Both registrations are estimated on 2 mm copies (--reg-spacing) and applied at full resolution.
  4. T1, T2 and mask warped onto the BraTS 1 mm grid (240x240x155, BraTS affine).
  5. Owner's z-score over non-zero voxels inside the warped mask; same crop / 2x
     brain-masked pooling / padding as Phase 1 (crop.json); channels [T1, 0, T2, 0], fp16.

Template: mean of brain-mean-normalised T1 over N BraTS *train* cases on their native
(already co-registered SRI24) grid, tumour (WT) voxels excluded from the average;
template mask = voxels that are brain in >= 50% of those cases.

v2: also writes sdt/<IXInnn>.npy, SynthStrip's signed distance map (mm, negative inside)
warped and pooled onto the same 2 mm grid, so one SynthStrip boundary can be applied to
BraTS and IXI alike (common.shared_mask). x is unchanged from v1 (default 1 mm border).

Outputs (OUT): x/<IXInnn>.npy, sdt/<IXInnn>.npy, rows/<IXInnn>.json (QC metrics), failed.jsonl,
template_t1.nii.gz, template_mask.nii.gz, summary_shard<k>.json.
IXI cases have no labels: no y file (datasets.py creates zeros from has_seg=0).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SDT_FILL, SEED, crop, pad_to, pool_image, pool_sdt, seg_to_bits, zscore_nonzero  # noqa: E402,F401

SIGNATURE = "ixi-bratslike-2mm-v2-sdt"
BORDERS = (1, 0, -1, -2, -3, -4)  # mm, SynthStrip boundary offsets reported per subject
SYNTHSTRIP_SHA = {"mri_synthstrip.py": "291c253ab2f7c0cbcb84afa769ae10909549537e727db5b92d268754b7cc7871",
                  "synthstrip.1.pt": "37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33"}
CFG = {}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- template
def build_template(tar_path, subjects, expected_shape, out_dir):
    """Mean of brain-mean-normalised T1 over BraTS train cases, tumour excluded."""
    t_img, m_img = Path(out_dir) / "template_t1.nii.gz", Path(out_dir) / "template_mask.nii.gz"
    if t_img.exists() and m_img.exists():
        return t_img, m_img
    want = {}
    for s in subjects:
        for mod in ("t1", "t2", "t1ce", "flair", "seg"):
            want[f"{s}_{mod}"] = (s, mod)
    vols = {}
    affine = None
    with tarfile.open(tar_path, "r:") as tf:
        for m in tf:
            mm = re.search(r"(BraTS2021_\d{5})_(t1|t1ce|t2|flair|seg)\.nii(\.gz)?$", m.name)
            if not mm or f"{mm.group(1)}_{mm.group(2)}" not in want:
                continue
            raw = tf.extractfile(m).read()
            if m.name.endswith(".gz"):
                raw = gzip.decompress(raw)
            img = nib.Nifti1Image.from_bytes(raw)
            vols.setdefault(mm.group(1), {})[mm.group(2)] = np.asarray(img.dataobj, np.float32)
            affine = img.affine if affine is None else affine
    acc = np.zeros(expected_shape, np.float64)
    cnt = np.zeros(expected_shape, np.float64)
    support = np.zeros(expected_shape, np.float64)
    for s, v in vols.items():
        brain = np.zeros(expected_shape, bool)
        for mod in ("t1", "t2", "t1ce", "flair"):
            brain |= v[mod] != 0
        support += brain
        use = brain & (v["seg"] == 0) & (v["t1"] != 0)
        acc[use] += v["t1"][use] / v["t1"][use].mean()
        cnt[use] += 1
    n = len(vols)
    mask = support >= 0.5 * n
    tpl = np.where(mask & (cnt > 0), acc / np.maximum(cnt, 1), 0.0).astype(np.float32)
    hdr = nib.Nifti1Header()
    hdr.set_qform(affine, 1)
    hdr.set_sform(affine, 1)
    nib.save(nib.Nifti1Image(tpl, affine, hdr), t_img)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine, hdr), m_img)
    meta = dict(n_cases=n, subjects=sorted(vols), mask_voxels=int(mask.sum()),
                axcodes="".join(nib.aff2axcodes(affine)), rule="brain in >=50% of cases; T1/brain-mean, WT excluded")
    (Path(out_dir) / "template.json").write_text(json.dumps(meta, indent=2))
    log("template built", {k: v for k, v in meta.items() if k != "subjects"})
    return t_img, m_img


# ---------------------------------------------------------------- per subject
def skullstrip(t1_path, mask_path, tmp, sdt_path=None):
    """SynthStrip brain mask (default 1 mm border); optionally also its signed distance map."""
    if CFG["skullstrip"] == "threshold":  # debug only (no SynthStrip locally)
        from scipy import ndimage as ndi
        img = nib.load(t1_path)
        a = np.asarray(img.dataobj, np.float32)
        m = a > np.percentile(a[a > 0], 30)
        nib.save(nib.Nifti1Image(m.astype(np.uint8), img.affine), mask_path)
        if sdt_path:
            sp = np.abs(np.diag(img.affine)[:3])
            sdt = ndi.distance_transform_edt(~m, sp) - ndi.distance_transform_edt(m, sp)
            nib.save(nib.Nifti1Image(sdt.astype(np.float32), img.affine), sdt_path)
        return
    tools = Path(CFG["tools"])
    cmd = [sys.executable, str(tools / "mri_synthstrip.py"), "-i", str(t1_path), "-m", str(mask_path),
           "--model", str(tools / "synthstrip.1.pt"), "-t", "1"]
    if sdt_path:
        cmd += ["-d", str(sdt_path)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900, cwd=tmp)
    if r.returncode != 0:
        raise RuntimeError("synthstrip failed: " + r.stderr[-500:])


def lowres(img):
    """Registration runs on a copy resampled to CFG['reg_spacing'] mm (transforms are in
    physical space, so they are then applied to the full-resolution images)."""
    import ants
    sp = CFG.get("reg_spacing", 0)
    if not sp:
        return img
    return ants.resample_image(img, (sp, sp, sp), use_voxels=False, interp_type=0)


def dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    return float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))


def process_subject(job):
    import ants
    subject, t1_path, t2_path = job["subject"], job["t1"], job["t2"]
    out = Path(CFG["out"])
    t0 = time.time()
    rec = dict(subject=subject, site=job["site"], split=job["split"], signature=SIGNATURE)
    with tempfile.TemporaryDirectory(dir=CFG["tmp"]) as tmp:
        tmp = Path(tmp)
        mask_path = tmp / "mask.nii.gz"
        sdt_path = tmp / "sdt.nii.gz"
        skullstrip(t1_path, mask_path, tmp, sdt_path)
        rec["t_skullstrip"] = round(time.time() - t0, 1)
        t1 = ants.image_read(str(t1_path))
        t2 = ants.image_read(str(t2_path))
        mask = ants.image_read(str(mask_path))
        mask = ants.resample_image_to_target(mask, t1, interp_type="nearestNeighbor")
        sdt = ants.resample_image_to_target(ants.image_read(str(sdt_path)), t1, interp_type="linear")
        sp = np.prod(t1.spacing)
        rec["native_brain_ml"] = float(mask.numpy().astype(bool).sum() * sp / 1000.0)
        rec["native_t1_spacing"] = [round(float(s), 3) for s in t1.spacing]
        rec["native_t2_spacing"] = [round(float(s), 3) for s in t2.spacing]
        # rigid T2 -> T1 on whole-head images
        rig = ants.registration(fixed=lowres(t1), moving=lowres(t2), type_of_transform="Rigid",
                                random_seed=SEED, outprefix=str(tmp / "rig_"))
        t2_in_t1 = ants.apply_transforms(fixed=t1, moving=t2, transformlist=rig["fwdtransforms"],
                                         interpolator="linear")
        t1b, t2b = t1 * mask, t2_in_t1 * mask
        tpl = ants.image_read(CFG["template"])
        tpl_mask = ants.image_read(CFG["template_mask"])
        aff = ants.registration(fixed=lowres(tpl), moving=lowres(t1b), type_of_transform="Affine",
                                random_seed=SEED, outprefix=str(tmp / "aff_"))
        rec["t_register"] = round(time.time() - t0 - rec["t_skullstrip"], 1)
        tr = aff["fwdtransforms"]
        w = {}
        for name, img, interp in (("t1", t1b, "linear"), ("t2", t2b, "linear"), ("mask", mask, "nearestNeighbor")):
            wi = ants.apply_transforms(fixed=tpl, moving=img, transformlist=tr, interpolator=interp)
            p = tmp / f"w_{name}.nii.gz"
            ants.image_write(wi, str(p))
            w[name] = np.asarray(nib.load(p).dataobj, np.float32)  # template voxel order = BraTS order
        ws = ants.apply_transforms(fixed=tpl, moving=sdt, transformlist=tr, interpolator="linear",
                                   defaultvalue=SDT_FILL)
        ants.image_write(ws, str(tmp / "w_sdt.nii.gz"))
        w["sdt"] = np.asarray(nib.load(tmp / "w_sdt.nii.gz").dataobj, np.float32)
        tmask = np.asarray(nib.load(CFG["template_mask"]).dataobj) > 0
        brain = w["mask"] > 0.5
        rec["mask_dice_template"] = dice(brain, tmask)
        rec["warped_brain_ml"] = float(brain.sum() / 1000.0)
        mat = ants.read_transform(tr[0]).parameters[:9].reshape(3, 3)
        rec["affine_det"] = float(np.linalg.det(mat))
        tpl_np = np.asarray(nib.load(CFG["template"]).dataobj, np.float32)
        both = brain & tmask
        rec["t1_template_corr"] = float(np.corrcoef(w["t1"][both], tpl_np[both])[0, 1]) if both.sum() > 100 else None
    chans = []
    support = np.zeros(brain.shape, bool)
    for name in ("t1", "t2"):
        a = np.where(brain, np.clip(w[name], 0, None), 0).astype(np.float32)
        z, mean, std = zscore_nonzero(a)
        rec[f"mean_{name}"], rec[f"std_{name}"] = mean, std
        support |= a != 0
        chans.append(z)
    zero = np.zeros(brain.shape, np.float32)
    x = np.stack([chans[0], zero, chans[1], zero])
    box = CFG["box"]
    rec["brain_vox_1mm"] = int(support.sum())
    rec["brain_vox_outside_box"] = int(support.sum() - crop(support, box).sum())
    xp, brain2 = pool_image(crop(x, box), crop(support, box))
    xp = pad_to(xp, CFG["target"])
    rec["brain_vox_2mm"] = int(brain2.sum())
    x16 = xp.astype(np.float16)
    assert np.isfinite(x16).all(), "fp16 overflow"
    np.save(out / "x" / f"{subject}.npy", x16)
    sdt2 = pool_sdt(w["sdt"], box, CFG["target"]).astype(np.float16)
    np.save(out / "sdt" / f"{subject}.npy", sdt2)
    for b in BORDERS:
        rec[f"brain_ml_b{b:+d}"] = float(((w["sdt"] < b) & brain).sum() / 1000.0)
    rec["x_sha256"] = hashlib.sha256(x16.tobytes()).hexdigest()[:16]
    rec["t_total"] = round(time.time() - t0, 1)
    (out / "rows" / f"{subject}.json").write_text(json.dumps(rec))
    return rec


def safe_process(job):
    try:
        return True, process_subject(job)
    except Exception as e:  # noqa: BLE001 - keep going on single failures
        return False, dict(subject=job["subject"], error=f"{type(e).__name__}: {e}",
                           trace=traceback.format_exc()[-1500:])


def _init(cfg):
    CFG.update(cfg)


def list_ixi(root):
    """subject -> {T1, T2: file}. In mdnakib/ixi-dataset (and the HF zip) each
    'IXInnn-Site-nnnn-T1.nii' is a directory holding the actual NIfTI file."""
    files = {}
    for f in sorted(Path(root).rglob("IXI*.nii*")):
        m = re.match(r"(IXI\d{3})-([A-Za-z]+)-\d+-(T1|T2)\.nii", f.name)
        if not m:
            continue
        if f.is_dir():
            inner = sorted(p for p in f.rglob("*.nii*") if p.is_file())
            assert len(inner) == 1, (str(f), [p.name for p in inner])
            f = inner[0]
        files.setdefault(m.group(1), {})[m.group(3)] = str(f)
    return files


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ixi-root", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--crop", required=True, help="Phase 1 crop.json")
    ap.add_argument("--brats-tar", required=True)
    ap.add_argument("--tools", default=None, help="dir with mri_synthstrip.py + synthstrip.1.pt")
    ap.add_argument("--skullstrip", choices=["synthstrip", "threshold"], default="synthstrip")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--template-cases", type=int, default=60)
    ap.add_argument("--expected-shape", type=int, nargs=3, default=[240, 240, 155])
    ap.add_argument("--reg-spacing", type=float, default=2.0, help="mm; 0 = register at native resolution")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--subjects", nargs="*", default=None, help="only these (reruns of failures)")
    ap.add_argument("--max-minutes", type=float, default=660)
    args = ap.parse_args(argv)
    t0 = time.time()
    out = Path(args.out)
    for d in ("x", "sdt", "rows"):
        (out / d).mkdir(parents=True, exist_ok=True)
    tmp = Path(args.tmp or out / "tmp")
    tmp.mkdir(parents=True, exist_ok=True)

    if args.skullstrip == "synthstrip":
        for name, h in SYNTHSTRIP_SHA.items():
            assert sha256(Path(args.tools) / name) == h, f"{name} hash mismatch"
    crop_info = json.loads(Path(args.crop).read_text())
    with open(args.splits) as f:
        rows = list(csv.DictReader(f))
    brats_train = sorted((r["subject"] for r in rows if r["source"] == "brats" and r["split"] == "train"),
                         key=lambda s: hashlib.sha256(s.encode()).hexdigest())[: args.template_cases]
    tpl, tpl_mask = build_template(args.brats_tar, brats_train, tuple(args.expected_shape), out)
    tpl_axc = "".join(nib.aff2axcodes(nib.load(tpl).affine))
    assert tpl_axc == crop_info["axcodes"], (tpl_axc, crop_info["axcodes"])

    files = list_ixi(args.ixi_root)
    ixi = [r for r in rows if r["source"] == "ixi"]
    jobs = []
    for i, r in enumerate(sorted(ixi, key=lambda r: r["subject"])):
        if i % args.nshards != args.shard:
            continue
        if args.subjects and r["subject"] not in args.subjects:
            continue
        f = files.get(r["subject"], {})
        assert "T1" in f and "T2" in f, r["subject"]
        jobs.append(dict(subject=r["subject"], site=r["site"], split=r["split"], t1=f["T1"], t2=f["T2"]))
    if args.limit:
        jobs = jobs[: args.limit]
    done = {p.stem for p in (out / "rows").glob("*.json") if (out / "sdt" / f"{p.stem}.npy").exists()}
    todo = [j for j in jobs if j["subject"] not in done]
    log(f"shard {args.shard}/{args.nshards}: {len(jobs)} subjects, {len(todo)} to do")
    cfg = dict(out=str(out), tmp=str(tmp), tools=args.tools, skullstrip=args.skullstrip,
               template=str(tpl), template_mask=str(tpl_mask),
               box=[tuple(b) for b in crop_info["box_1mm"]], target=crop_info["target_shape"],
               reg_spacing=args.reg_spacing)
    failed = []
    stopped = False
    with Pool(args.workers, initializer=_init, initargs=(cfg,), maxtasksperchild=20) as pool:
        for i, (ok, rec) in enumerate(pool.imap_unordered(safe_process, todo, chunksize=1)):
            if ok:
                log(f"{i + 1}/{len(todo)} {rec['subject']} {rec['t_total']}s dice={rec['mask_dice_template']:.3f}")
            else:
                failed.append(rec)
                log(f"{i + 1}/{len(todo)} FAILED {rec['subject']}: {rec['error']}")
                with open(out / "failed.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
            if (time.time() - t0) / 60 > args.max_minutes:
                log("max-minutes reached; stopping (resume to continue)")
                stopped = True
                pool.terminate()
                break
    shutil.rmtree(tmp, ignore_errors=True)
    recs = [json.loads((out / "rows" / f"{j['subject']}.json").read_text()) for j in jobs
            if (out / "rows" / f"{j['subject']}.json").exists()]
    summary = dict(signature=SIGNATURE, shard=args.shard, nshards=args.nshards, n_jobs=len(jobs),
                   n_done=len(recs), n_failed_this_run=len(failed), stopped_early=stopped,
                   elapsed_s=round(time.time() - t0),
                   median_t_total=float(np.median([r["t_total"] for r in recs])) if recs else None)
    (out / f"summary_shard{args.shard}.json").write_text(json.dumps(summary, indent=2))
    log("summary", json.dumps(summary))
    return summary


if __name__ == "__main__":
    main()
