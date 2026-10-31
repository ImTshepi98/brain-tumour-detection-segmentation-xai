"""Phase 5: train the dual-head model (and its ablations) on the 2 mm arrays.

Each step runs two passes through the shared encoder:
  detection pass: T1ce and FLAIR zeroed, all samples, BCE on the detection logit;
  segmentation pass: all 4 channels, BraTS samples only, squared Dice + BCE.
L = w_seg * L_seg + w_cls * L_cls. AdamW + per-step cosine decay, grad clip with finite checks, FP32.
Validation each epoch on the val split (whole volumes): DSC per region on BraTS, detection
F1/AUROC at threshold 0.5 on all. best.pt by 0.5*mean DSC + 0.5*F1 (single-task runs: their own metric).
The test split is never loaded here.

Outputs in --out: runtime.json, metrics.jsonl, last.pt (every epoch), best.pt, val_last.json,
val_best.json, done.json.

  python src/train.py --config configs/train/main.yaml --brats-dir ... --ixi-dirs ... \
      --brats-sdt-dirs ... --splits ... --out runs/main --max-minutes 600 [--resume auto|path]
  python src/train.py --debug --out runs/debug          # synthetic CPU dry run (PLAN rule 3)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datasets as ds  # noqa: E402
import losses  # noqa: E402
import metrics  # noqa: E402
from model import build, n_params  # noqa: E402

SIGNATURE = "brats-dual-train-v1"
RUNTIME_KEYS = ("max_minutes", "resume")  # changing these does not change the protocol


def load_config(path, overrides=()):
    cfg = yaml.safe_load(Path(path).read_text())
    for o in overrides:
        key, val = o.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(val)
    return cfg


def protocol_hash(cfg):
    core = {k: v for k, v in cfg.items() if k not in RUNTIME_KEYS}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()


def git_commit():
    f = Path(__file__).resolve().parents[1] / "GIT_COMMIT"
    if f.exists():
        return f.read_text().strip()
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).parent).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_subjects(rows, split, per_class, seed):
    """Smoke-test subset: per_class random subjects of each cohort from one split (0 = all)."""
    if not per_class:
        return None
    rng = np.random.default_rng(seed)
    out = set()
    for src in ("brats", "ixi"):
        ids = sorted(r["subject"] for r in rows if r["split"] == split and r["source"] == src)
        out |= set(rng.choice(ids, min(per_class, len(ids)), replace=False).tolist())
    return out


def lr_lambda(total_steps, warmup_steps):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
    return f


class Runner:
    """Calls the model through DataParallel only when every GPU gets at least one sample
    (a seg sub-batch of 1 BraTS case on 2 GPUs would leave replica 1 without input)."""

    def __init__(self, model, net):
        self.model, self.net = model, net
        self.n = len(net.device_ids) if isinstance(net, nn.DataParallel) else 1

    def __call__(self, x, task):
        return (self.net if x.shape[0] >= self.n else self.model)(x, task)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()


@torch.no_grad()
def validate(net, val, cfg, device):
    net.eval()
    w_seg, w_cls = cfg["loss"]["w_seg"], cfg["loss"]["w_cls"]
    recs = []
    bs = cfg.get("val_batch_size", 2)
    for start in range(0, len(val), bs):
        idx = np.arange(start, min(start + bs, len(val)))
        x, target, label, has_seg = val.batch(idx, device)
        prob = torch.sigmoid(net(x, "cls")).cpu().numpy() if w_cls > 0 else np.full(len(idx), np.nan)
        dice = np.full((len(idx), 3), np.nan)
        if w_seg > 0 and has_seg.any():
            logits = net(x[has_seg], "seg")
            assert torch.isfinite(logits).all(), "non-finite validation prediction"
            dice[has_seg.cpu().numpy()] = metrics.dice_per_case(metrics.postprocess(logits), target[has_seg] > 0.5)
        for j, i in enumerate(idx):
            r = dict(id=val.ids[i], source=val.source[i], site=val.site[i], label=int(val.label[i]),
                     prob=float(prob[j]))
            if val.has_seg[i]:
                r.update({k: float(dice[j, c]) for c, k in enumerate(metrics.REGIONS)})
                r.update({f"{k}_present": bool((val.y[i] >> c & 1).any()) for c, k in enumerate(metrics.REGIONS)})
            recs.append(r)
    seg = [r for r in recs if "WT" in r]
    dsc = {k: float(np.mean([r[k] for r in seg])) for k in metrics.REGIONS} if w_seg > 0 and seg else {}
    mean_dsc = float(np.mean(list(dsc.values()))) if dsc else float("nan")
    det = metrics.detection([r["label"] for r in recs], [r["prob"] for r in recs]) if w_cls > 0 else {}
    score = metrics.selection_score(mean_dsc, det.get("f1", float("nan")), w_seg, w_cls)
    return recs, dict(dsc=dsc, mean_dsc=mean_dsc, detection=det, score=score)


def save(path, state):
    tmp = Path(str(path) + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. border=0 optim.epochs=3")
    ap.add_argument("--brats-dir")
    ap.add_argument("--ixi-dirs", nargs="*", default=[])
    ap.add_argument("--brats-sdt-dirs", nargs="*", default=[])
    ap.add_argument("--splits")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-minutes", type=float, default=600)
    ap.add_argument("--resume", default=None, help="path to last.pt, or 'auto' = <out>/last.pt if present")
    ap.add_argument("--debug", action="store_true", help="synthetic data, CPU, tiny (PLAN rule 3)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-count-check", action="store_true", help="skip the real split counts (kernel dry runs)")
    args = ap.parse_args(argv)
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]

    if args.debug:
        import synthetic
        cfg = load_config(args.config or root / "configs/train/debug.yaml", args.set)
        paths = synthetic.write_fake_2mm(out / "debug_data", n_brats=cfg["debug"]["n_brats"],
                                         n_ixi=cfg["debug"]["n_ixi"], shape=tuple(cfg["debug"]["shape"]))
        args.brats_dir, args.ixi_dirs = paths["brats_dir"], [paths["ixi_dir"]]
        args.brats_sdt_dirs, args.splits = [paths["brats_sdt_dir"]], paths["splits"]
        expected = None
    else:
        cfg = load_config(args.config, args.set)
        expected = None if args.no_count_check else ds.EXPECTED
    border = cfg["border"]
    assert border is None or isinstance(border, (int, float)), \
        f"border must be set (mm, from calibrate_border.py) or null, got {border!r}"
    cfg["max_minutes"], cfg["resume"] = args.max_minutes, args.resume
    phash = protocol_hash(cfg)
    seed = cfg.get("seed", 42)
    seed_all(seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    rows = ds.read_splits(args.splits)
    counts = ds.check_splits(rows, expected)
    dcfg = cfg.get("data", {})
    load_kw = dict(brats_dir=args.brats_dir, ixi_dirs=args.ixi_dirs, brats_sdt_dirs=args.brats_sdt_dirs,
                   border=border, workers=dcfg.get("workers", 4), allow_missing=dcfg.get("allow_missing", False))
    tl = time.time()
    train = ds.Cohort.load(rows, ("train",), subjects=pick_subjects(rows, "train", dcfg.get("limit_train", 0), seed),
                           **load_kw)
    val = ds.Cohort.load(rows, ("val",), subjects=pick_subjects(rows, "val", dcfg.get("limit_val", 0), seed),
                         **load_kw)
    t_load = time.time() - tl
    print(f"loaded train {len(train)} val {len(val)} shape {train.shape} in {t_load:.0f}s "
          f"(border {border})", flush=True)

    model = build(cfg).to(device)
    net = nn.DataParallel(model) if (device.type == "cuda" and torch.cuda.device_count() > 1
                                     and cfg.get("data_parallel", True)) else model
    net = Runner(model, net)
    oc = cfg["optim"]
    w_seg, w_cls = cfg["loss"]["w_seg"], cfg["loss"]["w_cls"]
    epochs = oc["epochs"]
    bs = cfg["batch_size"]
    sampler = dcfg.get("sampler", "balanced")
    n_epoch = dcfg.get("samples_per_epoch", 0) or (len(train) if sampler != "brats_only" else int(train.label.sum()))
    steps_per_epoch = n_epoch // bs
    assert steps_per_epoch > 0, (n_epoch, bs)
    opt = torch.optim.AdamW(model.parameters(), lr=oc["lr"], weight_decay=oc["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda(epochs * steps_per_epoch, oc.get("warmup_epochs", 0) * steps_per_epoch))

    start_epoch, best, history = 0, -math.inf, []
    resume = out / "last.pt" if args.resume == "auto" else (Path(args.resume) if args.resume else None)
    if resume is not None and resume.exists():
        st = torch.load(resume, map_location=device, weights_only=False)
        assert st["signature"] == SIGNATURE and st["protocol_hash"] == phash, "resume: protocol differs"
        assert all(torch.isfinite(v).all() for v in st["model"].values()), "resume: non-finite checkpoint"
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
                   n_params=n_params(model), split_counts=counts, n_train=len(train), n_val=len(val),
                   shape=list(train.shape), steps_per_epoch=steps_per_epoch, data_load_seconds=t_load,
                   start_epoch=start_epoch)
    (out / "runtime.json").write_text(json.dumps(runtime, indent=2))
    print(f"params {runtime['n_params']:,}; {steps_per_epoch} steps/epoch; protocol {phash[:12]}", flush=True)

    budget = args.max_minutes * 60
    durations, t_first_step = [], None
    acfg = cfg.get("augment", {})
    stopped = "epochs_done"
    for epoch in range(start_epoch, epochs):
        if durations and time.time() - t0 + 1.1 * max(durations) > budget:
            stopped = "time_budget"
            break
        te = time.time()
        net.train()
        gen = torch.Generator(device=device).manual_seed(seed + epoch)
        order = ds.sample_epoch(train.label, steps_per_epoch * bs, sampler, seed, epoch)
        sums = dict(loss=0.0, seg=0.0, cls=0.0, n_seg=0)
        for step in range(steps_per_epoch):
            idx = order[step * bs:(step + 1) * bs]
            x, target, label, has_seg = train.batch(idx, device)
            x, target = ds.augment(x, target, gen, **acfg)
            opt.zero_grad(set_to_none=True)
            loss = torch.zeros((), device=device)
            if w_cls > 0:
                lc = losses.cls_loss(net(x, "cls"), label)
                loss = loss + w_cls * lc
                sums["cls"] += float(lc.detach())
            if w_seg > 0 and has_seg.any():
                ls = losses.seg_loss(net(x[has_seg], "seg"), target[has_seg])
                loss = loss + w_seg * ls
                sums["seg"] += float(ls.detach())
                sums["n_seg"] += 1
            assert torch.isfinite(loss), f"non-finite training loss at epoch {epoch} step {step}"
            if not loss.requires_grad:
                continue  # seg-only run and a batch with no BraTS case: nothing to learn from
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), oc["grad_clip"], error_if_nonfinite=True)
            opt.step()
            sched.step()
            sums["loss"] += float(loss.detach())
            if t_first_step is None:
                t_first_step = time.time() - t0
            if step < 2 or (step + 1) % 50 == 0:
                print(f"epoch {epoch} step {step + 1}/{steps_per_epoch} loss {float(loss.detach()):.4f}", flush=True)
        t_train = time.time() - te
        tv = time.time()
        recs, val_m = validate(net, val, cfg, device)
        t_val = time.time() - tv
        durations.append(time.time() - te)
        improved = val_m["score"] > best
        best = max(best, val_m["score"])
        rec = dict(epoch=epoch, lr=sched.get_last_lr()[0], train_loss=sums["loss"] / steps_per_epoch,
                   train_cls=sums["cls"] / steps_per_epoch if w_cls > 0 else None,
                   train_seg=sums["seg"] / max(1, sums["n_seg"]) if w_seg > 0 else None,
                   **{f"val_{k}": v for k, v in val_m.items()}, best=best, improved=improved,
                   train_seconds=t_train, val_seconds=t_val, steps_per_second=steps_per_epoch / t_train,
                   gpu_peak_gb=(torch.cuda.max_memory_allocated() / 2**30) if device.type == "cuda" else None,
                   elapsed_seconds=time.time() - t0)
        history.append(rec)
        with open(out / "metrics.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        (out / "val_last.json").write_text(json.dumps(recs, indent=1))
        state = dict(signature=SIGNATURE, protocol_hash=phash, config=cfg, epoch=epoch, best=best, history=history,
                     model=model.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                     python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state())
        save(out / "last.pt", state)
        if improved:
            save(out / "best.pt", state)
            (out / "val_best.json").write_text(json.dumps(recs, indent=1))
        print("EPOCH", json.dumps(rec), flush=True)

    done = dict(stopped=stopped, epochs_completed=len(history), best=best, total_seconds=time.time() - t0,
                data_load_seconds=t_load, first_step_seconds=t_first_step, protocol_hash=phash,
                test_read=False)
    (out / "done.json").write_text(json.dumps(done, indent=2))
    print("TRAINING RUN COMPLETE; TEST REMAINS LOCKED", json.dumps(done), flush=True)
    return done


if __name__ == "__main__":
    main()
