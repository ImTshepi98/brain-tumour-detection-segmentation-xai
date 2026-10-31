"""Phase 5 GPU training: run main-r1 (config main.yaml). Thin wrapper around src/train.py."""
import json, os, shutil, subprocess, sys
from pathlib import Path

CONFIG, BORDER, MAX_MINUTES, EXTRA = "main.yaml", 0.0, 270.0, []
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
prev = sorted(INP.rglob("run/last.pt"))  # attach an earlier version's output to continue
out = WORK / "run"
if prev:
    shutil.copytree(prev[0].parent, out, dirs_exist_ok=True)
    print("continuing from", prev[0], flush=True)
if not DRY:
    subprocess.run(["nvidia-smi"], check=False)
    subprocess.run(["free", "-g"], check=False)
# main v1 was SIGKILLed (host RAM, flat GPU memory) at epoch 15; a fixed mmap threshold makes glibc
# return the ~27 MB per-step batch buffers to the OS instead of fragmenting the heap
env = dict(os.environ, MALLOC_MMAP_THRESHOLD_="1048576", MALLOC_ARENA_MAX="2")
subprocess.run([sys.executable, "-u", str(CODE / "train.py"), "--config", str(CODE.parent / "configs/train" / CONFIG),
                "--set", f"border={json.dumps(BORDER)}", *EXTRA,
                "--brats-dir", str(brats), "--ixi-dirs", *ixi, "--brats-sdt-dirs", *sdt_b,
                "--splits", str(DATA / "splits.csv"), "--out", str(out), "--max-minutes", str(MAX_MINUTES),
                "--resume", "auto"] + (["--no-count-check"] if DRY else []), check=True, env=env if not DRY else None)
