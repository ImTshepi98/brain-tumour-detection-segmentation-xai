"""Phase 6: local CPU inference time (batch 1, after warm-up; detection + segmentation pass).

  python src/cpu_timing.py --ckpt data_local/main/best.pt --cases data_local/cases --out reports/cpu_inference.json
Uses evaluate.inference_time on the exported GUI cases (cycled to warm-up + n passes).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev  # noqa: E402
import infer  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args(argv)
    device = torch.device("cpu")
    model, st = ev.load_checkpoint(args.ckpt, device)
    xs = [infer.load_case(p)[0] for p in sorted(Path(args.cases).glob("*.npz"))]
    assert xs, f"no .npz cases in {args.cases}"

    class Cases:
        x = [xs[k % len(xs)] for k in range(args.n + args.warmup)]

        def __len__(self):
            return len(self.x)
    with torch.no_grad():
        r = ev.inference_time(model, Cases(), device, n=args.n, warmup=args.warmup)
    r.update(hardware=f"CPU: {platform.processor() or platform.machine()}, {os.cpu_count()} logical cores, "
                      f"torch {torch.__version__} ({torch.get_num_threads()} threads)",
             checkpoint_epoch=st["epoch"], protocol_hash=st["protocol_hash"], n_cases=len(xs))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(r, indent=2))
    return r


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
