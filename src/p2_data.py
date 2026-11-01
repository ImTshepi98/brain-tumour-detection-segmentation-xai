"""PLAN2 P1-P2 data: midline, fixed pairing files, pasted cells, group-balanced sampler, P1 report.

Cells (PLAN2 2.2): B+r, B-r, I-r are real hemispheres; B+p, I+p paste a B+r donor's tumour into a real
negative host; B-s, I-s paste the donor's healthy mirror tissue (its other hemisphere) at the same voxels.
Donors and hosts always come from the same split (rule 1.5).
* Val / test: every real negative host (B-r, I-r) gets one donor from a seeded permutation of the split's
  B+r samples (cycled), and is pasted twice with it: tumour (B+p / I+p) and sham (B-s / I-s). The pairing
  is written to pairs_<split>.csv before use.
* Train: each batch holds one draw per group B+, B-, I+, I- (25% each, PLAN2 2.3); within B+ real/pasted
  50/50, within B- and I- real/sham 50/50, I+ always pasted; new host and donor every draw (seed + epoch).

P1 report (CPU kernel): python src/p2_data.py --brats-dir ... --ixi-dirs ... --splits ... --border 0 --out ...
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hemi  # noqa: E402
from common import SEED  # noqa: E402

GROUPS = ("B+", "B-", "I+", "I-")


def fixed_pairs(hs, split, seed=SEED):
    """[dict(cell, host, donor, host_id, donor_id)] for every real negative sample of `hs` (sample indices)."""
    donors = hs.of("B+r")
    if not donors:
        return []
    order = np.random.default_rng(seed).permutation(donors)
    out, k = [], 0
    for host_cell, (pcell, scell) in (("B-r", ("B+p", "B-s")), ("I-r", ("I+p", "I-s"))):
        for h in hs.of(host_cell):
            d = int(order[k % len(order)])
            k += 1
            for cell in (pcell, scell):
                out.append(dict(split=split, cell=cell, host=h, donor=d,
                                host_id=f"{hs.meta(h)['subject']}:{hs.meta(h)['side']}",
                                donor_id=f"{hs.meta(d)['subject']}:{hs.meta(d)['side']}"))
    return out


def write_pairs(pairs, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "cell", "host", "donor", "host_id", "donor_id"])
        w.writeheader()
        w.writerows(pairs)


def make(hs, cell, host, donor):
    """(x float32, target bits uint8, info) for one pasted sample (cell B+p / I+p / B-s / I-s)."""
    sham = cell.endswith("s")
    dp, cp = hs.pool[donor], hs.contra[donor]
    x, t, frac, ml = hemi.paste(hs.x(host), hs.px[dp], hs.pbits[dp],
                                sham_src=hs.px[cp] if sham else None, sham_bits=hs.pbits[cp] if sham else None)
    if not sham and cell == "B+p":  # BraTS host: keep any host labels outside the region (negatives have none)
        t = np.where(t > 0, t, hs.bits(host))
    return x, t, dict(frac=frac, ml=ml)


class EvalSet:
    """Real samples + fixed pasted samples of one split, materialised once (fp16): the training-time val set
    and the evaluation set. rows[i] = dict(cell, label, dataset, id, donor_id, ...)."""

    def __init__(self, hs, pairs):
        self.x, self.bits, self.rows = [], [], []
        for k in range(len(hs)):
            mt = hs.meta(k)
            self.x.append(hs.x(k))
            self.bits.append(hs.bits(k))
            self.rows.append(dict(cell=hs.cell[k], label=hemi.POSITIVE[hs.cell[k]], dataset=hemi.DATASET[hs.cell[k]],
                                  id=f"{mt['subject']}:{mt['side']}", subject=mt["subject"], site=mt["site"],
                                  sample=k, donor_id="", wt_voxels=mt["wt_voxels"]))
        for p in pairs:
            x, t, info = make(hs, p["cell"], p["host"], p["donor"])
            self.x.append(x.astype(np.float16))
            self.bits.append(t)
            self.rows.append(dict(cell=p["cell"], label=hemi.POSITIVE[p["cell"]], dataset=hemi.DATASET[p["cell"]],
                                  id=p["host_id"], subject=p["host_id"].split(":")[0], site=hs.meta(p["host"])["site"],
                                  sample=p["host"], donor_id=p["donor_id"], wt_voxels=int(((t & 4) > 0).sum()), **info))
        self.label = np.array([r["label"] for r in self.rows])

    def __len__(self):
        return len(self.rows)


def sample_batch(hs, rng, n_per_group=1):
    """[(x float32, bits uint8, cell)] with n_per_group draws from each of B+, B-, I+, I- (PLAN2 2.3)."""
    pools = {c: hs.of(c) for c in ("B+r", "B-r", "I-r")}
    donors = pools["B+r"]
    out = []
    for g in GROUPS:
        for _ in range(n_per_group):
            if g == "B+":
                cell = "B+r" if rng.random() < 0.5 or not pools["B-r"] else "B+p"
            elif g == "B-":
                cell = "B-r" if rng.random() < 0.5 else "B-s"
            elif g == "I+":
                cell = "I+p"
            else:
                cell = "I-r" if rng.random() < 0.5 else "I-s"
            if cell in ("B+r", "B-r", "I-r"):
                k = int(rng.choice(pools[cell]))
                out.append((hs.x(k).astype(np.float32), hs.bits(k), cell))
            else:
                host = int(rng.choice(pools["B-r" if cell[0] == "B" else "I-r"]))
                x, t, _ = make(hs, cell, host, int(rng.choice(donors)))
                out.append((x, t, cell))
    return out


def mean_brain_mask(rows, paths, border, workers=4):
    from concurrent.futures import ThreadPoolExecutor
    import datasets as ds
    acc, n = None, 0
    with ThreadPoolExecutor(max(1, workers)) as ex:
        for x, _ in ex.map(lambda r: ds.load_one(paths[r["subject"]], border), rows):
            b = (x != 0).any(0).astype(np.float32)
            acc = b if acc is None else acc + b
            n += 1
    return acc / max(n, 1)


def montage(hs, path, per_cell=6, seed=SEED):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = [c for c in ("B+r", "B-r", "I-r") if hs.of(c)]
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(len(cells), per_cell, figsize=(per_cell * 1.6, len(cells) * 2.4), squeeze=False)
    for r, c in enumerate(cells):
        pick = rng.choice(hs.of(c), min(per_cell, len(hs.of(c))), replace=False)
        for j in range(per_cell):
            a = ax[r, j]
            a.axis("off")
            if j >= len(pick):
                continue
            k = int(pick[j])
            x, b = hs.x(k).astype(np.float32), hs.bits(k)
            z = x.shape[-1] // 2 if not ((b & 4) > 0).any() else int(np.argmax(((b & 4) > 0).sum((0, 1))))
            a.imshow(x[0, :, :, z].T, cmap="gray", origin="lower")
            if ((b & 4) > 0).any():
                a.contour(((b[:, :, z] & 4) > 0).T, levels=[0.5], colors="yellow", linewidths=0.6)
            mt = hs.meta(k)
            a.set_title(f"{c} {mt['subject'][-5:]}:{mt['side']}", fontsize=6)
    fig.suptitle("Hemispheres (T1, axial; midline at the right edge; WT yellow)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def qc_panels(hs, pairs, out_dir, n=12, seed=SEED):
    """n panels of fixed val pastes (half tumour, half sham; both datasets): host, pasted, donor, T1 + T2."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    want = {"I+p": n // 3, "B+p": n // 6, "I-s": n // 4, "B-s": n - n // 3 - n // 6 - n // 4}
    done = []
    for cell, k in want.items():
        cand = [p for p in pairs if p["cell"] == cell]
        for p in [cand[i] for i in rng.choice(len(cand), min(k, len(cand)), replace=False)] if cand else []:
            x, t, info = make(hs, cell, p["host"], p["donor"])
            dbits = hs.pbits[hs.pool[p["donor"]]]
            z = int(np.argmax(((dbits & 4) > 0).sum((0, 1))))
            fig, ax = plt.subplots(2, 3, figsize=(6, 5))
            for r, ch in enumerate((0, 2)):
                for c, (img, title) in enumerate(((hs.x(p["host"]), "host"), (x, cell),
                                                  (hs.px[hs.pool[p["donor"]]], "donor"))):
                    ax[r, c].imshow(np.asarray(img, np.float32)[ch, :, :, z].T, cmap="gray", origin="lower")
                    ax[r, c].contour(((dbits[:, :, z] & 4) > 0).T, levels=[0.5], colors="yellow", linewidths=0.5)
                    ax[r, c].set_title(f"{title} {'T1' if ch == 0 else 'T2'}", fontsize=7)
                    ax[r, c].axis("off")
            fig.suptitle(f"{cell}: host {p['host_id']} donor {p['donor_id']} ({info['ml']:.1f} mL, "
                         f"{info['frac']:.2f} pasted)", fontsize=7)
            fig.tight_layout()
            name = f"{len(done):02d}_{cell.replace('+', 'pos').replace('-', 'neg')}.png"
            fig.savefig(out_dir / name, dpi=100)
            plt.close(fig)
            done.append(dict(file=name, **{k: p[k] for k in ("cell", "host_id", "donor_id")}, **info))
    (out_dir / "panels.json").write_text(json.dumps(dict(panels=done, visual_review_complete=False), indent=1))
    return done


def main(argv=None):
    import datasets as ds
    ap = argparse.ArgumentParser()
    ap.add_argument("--brats-dir")
    ap.add_argument("--ixi-dirs", nargs="*", default=[])
    ap.add_argument("--brats-sdt-dirs", nargs="*", default=[])
    ap.add_argument("--splits")
    ap.add_argument("--border", type=float, default=0)
    ap.add_argument("--width", type=int, default=hemi.WIDTH)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-count-check", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = ds.read_splits(args.splits)
    ds.check_splits(rows, None if args.no_count_check else ds.EXPECTED)
    train_rows = [r for r in rows if r["split"] == "train"]
    paths = ds.locate(train_rows, args.brats_dir, args.ixi_dirs, args.brats_sdt_dirs)
    mm = mean_brain_mask(train_rows, paths, args.border, args.workers)
    m, dice = hemi.find_midline(mm)
    print(f"midline {m} (mirror Dice {dice:.4f}) of axis length {mm.shape[0]}", flush=True)
    np.save(out / "mean_train_mask.npy", mm.astype(np.float32))
    load = dict(brats_dir=args.brats_dir, ixi_dirs=args.ixi_dirs, brats_sdt_dirs=args.brats_sdt_dirs,
                border=args.border, width=args.width, workers=args.workers)
    summary = dict(midline=m, mirror_dice=dice, width=args.width, border=args.border, axis_length=int(mm.shape[0]),
                   min_pos_voxels=hemi.MIN_POS_VOX, margin_voxels=hemi.MARGIN, splits={})
    for split in ("train", "val"):
        hs = hemi.HemiSet.load(rows, (split,), m, **load)
        loss = [mt["crop_loss"] for mt in hs.pmeta[::2]]
        s = dict(counts=hs.counts, cells={c: len(hs.of(c)) for c in ("B+r", "B-r", "I-r")},
                 crop_loss_median=float(np.median(loss)), crop_loss_max=float(np.max(loss)))
        if split == "val":
            pairs = fixed_pairs(hs, "val")
            write_pairs(pairs, out / "pairs_val.csv")
            s["pairs"] = {c: sum(p["cell"] == c for p in pairs) for c in ("B+p", "B-s", "I+p", "I-s")}
            qc = qc_panels(hs, pairs, out / "qc_panels")
            s["paste_ml_median"] = float(np.median([q["ml"] for q in qc if q["cell"].endswith("p")] or [np.nan]))
        montage(hs, out / f"montage_{split}.png")
        summary["splits"][split] = s
        print(split, json.dumps(s), flush=True)
        del hs
    (out / "p1_summary.json").write_text(json.dumps(summary, indent=2))
    print("P1 DONE", flush=True)
    return summary


if __name__ == "__main__":
    main()
