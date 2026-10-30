"""SynthStrip distance maps for BraTS (shard __SHARD__ of __NSHARDS__)."""
import subprocess, sys, shutil
from pathlib import Path

SHARD, NSHARDS = __SHARD__, __NSHARDS__
EXTRA = __EXTRA__
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "antspyx==0.6.3", "surfa==0.6.3"], check=True)
CODE = next(p.parent for p in Path("/kaggle/input").rglob("sdt_brats.py"))
DATA = CODE.parent / "data"
TOOLS = next(Path("/kaggle/input").rglob("synthstrip.1.pt")).parent
TAR = next(Path("/kaggle/input").rglob("BraTS2021_Training_Data.tar"))
print("code", CODE, "commit", (CODE.parent / "GIT_COMMIT").read_text().strip(), "tools", TOOLS, flush=True)
shutil.copytree(CODE, "/kaggle/working/src_used", dirs_exist_ok=True)
subprocess.run([sys.executable, "-u", str(CODE / "sdt_brats.py"),
                "--tar", str(TAR), "--splits", str(DATA / "splits.csv"), "--crop", str(DATA / "crop.json"),
                "--tools", str(TOOLS), "--out", "/kaggle/working/sdt", "--tmp", "/tmp",
                "--shard", str(SHARD), "--nshards", str(NSHARDS), "--workers", "4", "--max-minutes", "650"] + EXTRA,
               check=True)
