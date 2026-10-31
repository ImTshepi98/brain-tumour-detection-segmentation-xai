"""Phase 5 GPU training: run __NAME__ (config __CONFIG__). Thin wrapper around src/train.py."""
import json, shutil, subprocess, sys
from pathlib import Path

CONFIG, BORDER, MAX_MINUTES, EXTRA = "__CONFIG__", __BORDER__, __MAX_MINUTES__, __EXTRA__
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "monai==1.6.1"], check=True)
INP = Path("/kaggle/input")
CODE = next(p.parent for p in INP.rglob("train.py") if (p.parent / "datasets.py").exists())
DATA = CODE.parent / "data"
print("code", CODE, "commit", (CODE.parent / "GIT_COMMIT").read_text().strip(), flush=True)
shutil.copytree(CODE, "/kaggle/working/src_used", dirs_exist_ok=True)
brats = next(p.parent for p in INP.rglob("brats2mm/crop.json"))
ixi = sorted(str(p.parent) for p in INP.rglob("ixi2mm/template.json"))
sdt_b = sorted(str(p.parent) for p in INP.rglob("sdt/summary_shard*.json"))
assert len(ixi) == 3 and len(sdt_b) == 3, (ixi, sdt_b)
prev = sorted(INP.rglob("run/last.pt"))  # attach an earlier version's output to continue
out = Path("/kaggle/working/run")
if prev:
    shutil.copytree(prev[0].parent, out, dirs_exist_ok=True)
    print("continuing from", prev[0], flush=True)
subprocess.run(["nvidia-smi"], check=False)
subprocess.run([sys.executable, "-u", str(CODE / "train.py"), "--config", str(CODE.parent / "configs/train" / CONFIG),
                "--set", f"border={json.dumps(BORDER)}", *EXTRA,
                "--brats-dir", str(brats), "--ixi-dirs", *ixi, "--brats-sdt-dirs", *sdt_b,
                "--splits", str(DATA / "splits.csv"), "--out", str(out), "--max-minutes", str(MAX_MINUTES),
                "--resume", "auto"], check=True)
