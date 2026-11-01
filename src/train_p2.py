"""PLAN2 P2-P3: train the shortcut-controlled model on hemispheres (PLAN2 sections 2-3).

Each step: one batch with equal draws from the groups B+, B-, I+, I- (pasting on the fly, new host and donor
per draw, seeded per epoch). Augmentation on the GPU: random brain-boundary erosion 0-2 voxels, then PLAN's
flips (not left-right), intensity scale/shift and noise. Losses:
  pass B + detection: one encoder pass on T1 + T2 for every sample -> detection BCE and WT segmentation loss;
  pass A: all 4 channels, BraTS samples (real, pasted, sham, negatives) -> ET/TC/WT segmentation loss;
  L = L_segA + L_segB + w_cls * L_cls (+ w_adv * L_adv, BraTS vs IXI within each label, p2-adv only).
Validation each epoch on the fixed val set (real val hemispheres + the fixed pasted cells, pairs_val.csv):
mean DSC of pass A on real tumour hemispheres (B+r), and the worst-group balanced accuracy of the detection
head at 0.5. best.pt by 0.5 * mean DSC + 0.5 * worst-group BA. The test split is never loaded here.

  python src/train_p2.py --config configs/train/p2_main.yaml --set border=0 midline=<m> --brats-dir ... --out ...
  python src/train_p2.py --debug --out runs/debug-p2        # synthetic CPU dry run
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datasets as ds  # noqa: E402
import hemi  # noqa: E402
import metrics  # noqa: E402
import p2_data  # noqa: E402
import p2_metrics as pm  # noqa: E402
from model import n_params  # noqa: E402
from p2_model import build  # noqa: E402
from train import (Runner, git_commit, host_rss_gb, load_config, lr_lambda, pick_subjects, protocol_hash,  # noqa: E402
                   save, seed_all)

SIGNATURE = "brats-dual-p2-train-v1"


@torch.no_grad()
def predict(net, ev, device, bs=4, seg_a=True):
    """Per-row detection prob/logit, pass-B WT volume (mL) and pass-A DSC (B+r rows)."""
    net.eval()
    recs = []
    for start in range(0, len(ev), bs):
        idx = list(range(start, min(start + bs, len(ev))))
        x = torch.from_numpy(np.stack([np.asarray(ev.x[i], np.float32) for i in idx])).to(device)
        out = net(x, "det")
        logit, wt = out[0], out[1]
        vol = (torch.sigmoid(wt) > 0.5).flatten(1).sum(1).float() * 8 / 1000
        dice = np.full((len(idx), 3), np.nan)
        br = [j for j, i in enumerate(idx) if ev.rows[i]["cell"] == "B+r"]
        if seg_a and br:
            bits = torch.from_numpy(np.stack([ev.bits[idx[j]] for j in br])).to(device)
            target = torch.stack([(bits >> k) & 1 for k in range(3)], 1).bool()
            logits = net(x[br], "seg")
            assert torch.isfinite(logits).all(), "non-finite validation prediction"
            dice[br] = metrics.dice_per_case(metrics.postprocess(logits), target)
        for j, i in enumerate(idx):
            r = dict(ev.rows[i])
            r.update(logit=float(logit[j]), prob=float(torch.sigmoid(logit[j])), wt_ml_b=float(vol[j]))
            if not np.isnan(dice[j, 0]):
                r.update({k: float(dice[j, c]) for c, k in enumerate(metrics.REGIONS)})
            recs.append(r)
    return recs


def summarise(recs):
    seg = [r for r in recs if "WT" in r]
    dsc = {k: float(np.mean([r[k] for r in seg])) for k in metrics.REGIONS} if seg else {}
    mean_dsc = float(np.mean(list(dsc.values()))) if dsc else float("nan")
    probs = [r["prob"] for r in recs]
    ba = pm.balanced_accuracy(recs, probs, 0.5)
    wg = min(ba.values()) if ba else float("nan")
    return dict(dsc=dsc, mean_dsc=mean_dsc, ba=ba, worst_ba=wg, cells=pm.cell_rates(recs, probs, 0.5),
                score=float(0.5 * mean_dsc + 0.5 * wg))


def adv_loss(adv, is_brats, label):
    """BCE of the adversarial head for BraTS vs IXI, averaged over the two labels (within-label confusion)."""
    parts = [F.binary_cross_entropy_with_logits(adv[label == v], is_brats[label == v].float())
             for v in (0, 1) if (label == v).any()]
    return sum(parts) / len(parts)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--brats-dir")
    ap.add_argument("--ixi-dirs", nargs="*", default=[])
    ap.add_argument("--brats-sdt-dirs", nargs="*", default=[])
    ap.add_argument("--splits")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-minutes", type=float, default=600)
    ap.add_argument("--max-rss-gb", type=float, default=0)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-count-check", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    if args.debug:
        import synthetic
        cfg = load_config(args.config or root / "configs/train/p2_debug.yaml", args.set)
        paths = synthetic.write_fake_2mm(out / "debug_data", n_brats=cfg["debug"]["n_brats"],
                                         n_ixi=cfg["debug"]["n_ixi"], shape=tuple(cfg["debug"]["shape"]), lateral=True)
        args.brats_dir, args.ixi_dirs = paths["brats_dir"], [paths["ixi_dir"]]
        args.brats_sdt_dirs, args.splits = [paths["brats_sdt_dir"]], paths["splits"]
        expected = None
    else:
        cfg = load_config(args.config, args.set)
        expected = None if args.no_count_check else ds.EXPECTED
    border, m, width = cfg["border"], cfg["midline"], cfg["width"]
    assert isinstance(border, (int, float)) and isinstance(m, int), f"border/midline must be set: {border!r} {m!r}"
    cfg["max_minutes"], cfg["resume"], cfg["max_rss_gb"] = args.max_minutes, args.resume, args.max_rss_gb
    phash = protocol_hash(cfg)
    seed = cfg.get("seed", 42)
    seed_all(seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    rows = ds.read_splits(args.splits)
    counts = ds.check_splits(rows, expected)
    dcfg = cfg.get("data", {})
    load_kw = dict(brats_dir=args.brats_dir, ixi_dirs=args.ixi_dirs, brats_sdt_dirs=args.brats_sdt_dirs,
                   border=border, width=width, workers=dcfg.get("workers", 4))
    tl = time.time()
    train = hemi.HemiSet.load(rows, ("train",), m, subjects=pick_subjects(rows, "train", dcfg.get("limit_train", 0),
                                                                          seed), **load_kw)
    valh = hemi.HemiSet.load(rows, ("val",), m, subjects=pick_subjects(rows, "val", dcfg.get("limit_val", 0), seed),
                             **load_kw)
    pairs = p2_data.fixed_pairs(valh, "val", seed)
    p2_data.write_pairs(pairs, out / "pairs_val.csv")
    val = p2_data.EvalSet(valh, pairs)
    t_load = time.time() - tl
    cells_train = {c: len(train.of(c)) for c in ("B+r", "B-r", "I-r")}
    cells_val = {c: int(sum(r["cell"] == c for r in val.rows)) for c in hemi.CELLS}
    print(f"loaded train hemispheres {cells_train}, val {cells_val} in {t_load:.0f}s (midline {m}, width {width})",
          flush=True)

    model = build(cfg).to(device)
    net = nn.DataParallel(model) if (device.type == "cuda" and torch.cuda.device_count() > 1
                                     and cfg.get("data_parallel", True)) else model
    net = Runner(model, net)
    oc, lw = cfg["optim"], cfg["loss"]
    epochs, bs = oc["epochs"], cfg["batch_size"]
    assert bs % 4 == 0, "batch size must hold equal draws of the 4 groups"
    steps_per_epoch = dcfg["samples_per_epoch"] // bs
    opt = torch.optim.AdamW(model.parameters(), lr=oc["lr"], weight_decay=oc["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda(epochs * steps_per_epoch, oc.get("warmup_epochs", 0) * steps_per_epoch))

    start_epoch, best, history = 0, -math.inf, []
    resume = out / "last.pt" if args.resume == "auto" else (Path(args.resume) if args.resume else None)
    if resume is not None and resume.exists():
        st = torch.load(resume, map_location=device, weights_only=False)
        assert st["signature"] == SIGNATURE and st["protocol_hash"] == phash, "resume: protocol differs"
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        start_epoch, best, history = st["epoch"] + 1, st["best"], st["history"]
        random.setstate(st["python_rng"])
        np.random.set_state(st["numpy_rng"])
        torch.set_rng_state(st["torch_rng"].cpu())
        print(f"resumed from {resume} at epoch {start_epoch}", flush=True)
    elif args.resume not in (None, "auto"):
        raise FileNotFoundError(resume)

    runtime = dict(signature=SIGNATURE, protocol_hash=phash, git_commit=git_commit(), config=cfg,
                   torch=torch.__version__, device=str(device),
                   gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                   n_params=n_params(model), split_counts=counts, hemi_counts=train.counts | valh.counts,
                   cells_train=cells_train, cells_val=cells_val, steps_per_epoch=steps_per_epoch,
                   data_load_seconds=t_load, start_epoch=start_epoch)
    (out / "runtime.json").write_text(json.dumps(runtime, indent=2))
    print(f"params {runtime['n_params']:,}; {steps_per_epoch} steps/epoch; protocol {phash[:12]}", flush=True)

    budget = args.max_minutes * 60
    durations, t_first_step, stopped = [], None, "epochs_done"
    acfg = dict(cfg.get("augment", {}))
    erode = acfg.pop("erode_max", 0)
    rss = [host_rss_gb()]
    for epoch in range(start_epoch, epochs):
        if durations and time.time() - t0 + 1.1 * max(durations) > budget:
            stopped = "time_budget"
            break
        if args.max_rss_gb and len(rss) > 1 and None not in rss:
            growth = max(b - a for a, b in zip(rss, rss[1:]))
            if rss[-1] + 1.5 * max(growth, 0) > args.max_rss_gb:
                stopped = "memory_budget"
                break
        te = time.time()
        net.train()
        gen = torch.Generator(device=device).manual_seed(seed + epoch)
        rng = np.random.default_rng(seed + 1000 * epoch)
        sums = dict(loss=0.0, segA=0.0, segB=0.0, cls=0.0, adv=0.0)
        for step in range(steps_per_epoch):
            x, target, label, is_brats = pm.to_tensors(p2_data.sample_batch(train, rng, bs // 4), device)
            if erode:
                x = pm.erode_batch(x, target, gen, erode)
            x, target = ds.augment(x, target, gen, **acfg)
            opt.zero_grad(set_to_none=True)
            res = net(x, "det")
            lc = F.binary_cross_entropy_with_logits(res[0], label)
            lb = pm.seg_loss(res[1][:, None], target[:, 2:3])
            la = pm.seg_loss(net(x[is_brats], "seg"), target[is_brats])
            loss = la + lb + lw["w_cls"] * lc
            if len(res) > 2:
                ld = adv_loss(res[2], is_brats, label)
                loss = loss + lw["w_adv"] * ld
                sums["adv"] += float(ld.detach())
            assert torch.isfinite(loss), f"non-finite training loss at epoch {epoch} step {step}"
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), oc["grad_clip"], error_if_nonfinite=True)
            opt.step()
            sched.step()
            for k, v in (("loss", loss), ("segA", la), ("segB", lb), ("cls", lc)):
                sums[k] += float(v.detach())
            if t_first_step is None:
                t_first_step = time.time() - t0
            if step < 2 or (step + 1) % 50 == 0:
                print(f"epoch {epoch} step {step + 1}/{steps_per_epoch} loss {float(loss.detach()):.4f} "
                      f"rss {host_rss_gb() or 0:.2f} GB", flush=True)
        t_train = time.time() - te
        tv = time.time()
        recs = predict(net, val, device, cfg.get("val_batch_size", 4))
        val_m = summarise(recs)
        t_val = time.time() - tv
        durations.append(time.time() - te)
        improved = val_m["score"] > best
        best = max(best, val_m["score"])
        rec = dict(epoch=epoch, lr=sched.get_last_lr()[0], **{f"train_{k}": v / steps_per_epoch for k, v in sums.items()},
                   **{f"val_{k}": v for k, v in val_m.items()}, best=best, improved=improved, train_seconds=t_train,
                   val_seconds=t_val, steps_per_second=steps_per_epoch / t_train,
                   gpu_peak_gb=(torch.cuda.max_memory_allocated() / 2**30) if device.type == "cuda" else None,
                   host_rss_gb=host_rss_gb(), elapsed_seconds=time.time() - t0)
        history.append(rec)
        with open(out / "metrics.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        (out / "val_last.json").write_text(json.dumps(recs))
        state = dict(signature=SIGNATURE, protocol_hash=phash, config=cfg, epoch=epoch, best=best, history=history,
                     model=model.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                     python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state())
        save(out / "last.pt", state)
        if improved:
            save(out / "best.pt", state)
            (out / "val_best.json").write_text(json.dumps(recs))
        print("EPOCH", json.dumps(rec), flush=True)
        rss.append(rec["host_rss_gb"])
        del recs, state
        gc.collect()

    done = dict(stopped=stopped, epochs_completed=len(history), next_epoch=len(history), epochs_planned=epochs,
                start_epoch=start_epoch, best=best, total_seconds=time.time() - t0, data_load_seconds=t_load,
                first_step_seconds=t_first_step, protocol_hash=phash, test_read=False)
    (out / "done.json").write_text(json.dumps(done, indent=2))
    print(f"TRAINING SESSION ENDED ({stopped}); TEST REMAINS LOCKED", json.dumps(done), flush=True)
    return done


if __name__ == "__main__":
    main()
