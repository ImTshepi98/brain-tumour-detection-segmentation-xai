"""Phase 2 kernel (shard 0 of 3): IXI -> BraTS-like 2 mm arrays."""
import subprocess, sys, shutil
from pathlib import Path

SHARD, NSHARDS = 0, 3
EXTRA = []
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "antspyx==0.6.3", "surfa==0.6.3"], check=True)
CODE = next(p.parent for p in Path("/kaggle/input").rglob("prep_ixi_2mm.py"))
DATA = CODE.parent / "data"
TOOLS = next(Path("/kaggle/input").rglob("synthstrip.1.pt")).parent
TAR = next(Path("/kaggle/input").rglob("BraTS2021_Training_Data.tar"))
IXI = next(p for p in Path("/kaggle/input").rglob("mdnakib")).joinpath("ixi-dataset")
print("code", CODE, "commit", (CODE.parent / "GIT_COMMIT").read_text().strip(), "tools", TOOLS, "ixi", IXI, flush=True)
shutil.copytree(CODE, "/kaggle/working/src_used", dirs_exist_ok=True)
subprocess.run([sys.executable, "-u", str(CODE / "prep_ixi_2mm.py"),
                "--ixi-root", str(IXI), "--splits", str(DATA / "splits.csv"), "--crop", str(DATA / "crop.json"),
                "--brats-tar", str(TAR), "--tools", str(TOOLS), "--out", "/kaggle/working/ixi2mm",
                "--tmp", "/tmp/ixi_tmp", "--shard", str(SHARD), "--nshards", str(NSHARDS),
                "--workers", "4", "--max-minutes", "650"] + EXTRA, check=True)
