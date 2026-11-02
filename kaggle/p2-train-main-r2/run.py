"""PLAN2 GPU training: run main-r2 (config p2_main.yaml). Thin wrapper around src/train_p2.py."""
import json, os, shutil, subprocess, sys
from pathlib import Path

CONFIG, BORDER, MAX_MINUTES, EXTRA, CONTINUE = "p2_main.yaml", 0.0, 270.0, ['midline=41'], ['brats-dual-p2-train-main-r1']
# BRATS_DUAL_DRYRUN=<dir>: local dry run against <dir>/input, writing to <dir>/working (PLAN rule 3)
DRY = os.environ.get("BRATS_DUAL_DRYRUN")
if not DRY:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "monai==1.6.1"], check=True)
INP = Path(DRY, "input") if DRY else Path("/kaggle/input")
WORK = Path(DRY, "working") if DRY else Path("/kaggle/working")
CODE = next(p.parent for p in INP.rglob("train.py") if "brats-dual-code" in p.parts)  # the code dataset only (attached kernel outputs carry old src_used/ copies)
DATA = CODE.parent / "data"
print("code", CODE, "commit", (CODE.parent / "GIT_COMMIT").read_text().strip(), flush=True)
shutil.copytree(CODE, WORK / "src_used", dirs_exist_ok=True)
brats = next(p.parent for p in INP.rglob("brats2mm/crop.json"))
ixi = sorted(str(p.parent) for p in INP.rglob("ixi2mm/template.json"))
sdt_b = sorted(str(p.parent) for p in INP.rglob("sdt/summary_shard*.json"))
assert len(ixi) == 3 and len(sdt_b) == 3, (ixi, sdt_b)
# Continue from the furthest checkpoint attached (an earlier completed kernel's output, or a private dataset
# holding a crashed run's run/ folder): train.py resumes at that epoch + 1 and runs to the configured end.
# Any last.pt counts: Kaggle flattens a dataset whose only top-level entry is run/ (main-r2 missed it)
import torch
prev = sorted(((torch.load(p, map_location="cpu", weights_only=False)["epoch"], str(p)) for p in INP.rglob("last.pt")
               if (p.parent / "metrics.jsonl").exists()), reverse=True)
print("attached checkpoints (epoch, path):", prev, flush=True)
assert prev or not CONTINUE, f"built to continue from {CONTINUE} but no last.pt is attached"
out = WORK / "run"
if prev:
    shutil.copytree(Path(prev[0][1]).parent, out, dirs_exist_ok=True)
    print(f"continuing from {prev[0][1]} (epoch {prev[0][0]} done; next epoch {prev[0][0] + 1})", flush=True)
if not DRY:
    subprocess.run(["nvidia-smi"], check=False)
    subprocess.run(["free", "-g"], check=False)
# main v1 was SIGKILLed (host RAM, flat GPU memory) at epoch 15; a fixed mmap threshold makes glibc
# return the ~27 MB per-step batch buffers to the OS instead of fragmenting the heap
env = dict(os.environ, MALLOC_MMAP_THRESHOLD_="1048576", MALLOC_ARENA_MAX="2")
subprocess.run([sys.executable, "-u", str(CODE / "train_p2.py"), "--config", str(CODE.parent / "configs/train" / CONFIG),
                "--set", f"border={json.dumps(BORDER)}", *EXTRA,
                "--brats-dir", str(brats), "--ixi-dirs", *ixi, "--brats-sdt-dirs", *sdt_b,
                "--splits", str(DATA / "splits.csv"), "--out", str(out), "--max-minutes", str(MAX_MINUTES),
                "--max-rss-gb", "0" if DRY else "29",  # Kaggle T4 kernels have ~31 GB; v1/r1 were killed at ~30 GB
                "--resume", "auto"] + (["--no-count-check"] if DRY else []), check=True, env=env if not DRY else None)
