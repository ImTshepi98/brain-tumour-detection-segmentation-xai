"""Phase 1 kernel: BraTS -> 2 mm arrays. Thin wrapper around src/prep_brats_2mm.py."""
import json, shutil, subprocess, sys
from pathlib import Path

CODE = next(p.parent for p in Path("/kaggle/input").rglob("prep_brats_2mm.py"))
DATA = CODE.parent / "data"
print("code", CODE, "commit", (CODE.parent / "GIT_COMMIT").read_text().strip(), flush=True)
shutil.copytree(CODE, "/kaggle/working/src_used", dirs_exist_ok=True)
subprocess.run([sys.executable, "-u", str(CODE / "prep_brats_2mm.py"),
                "--splits", str(DATA / "splits.csv"),
                "--stats", str(DATA / "brats_modality_stats_owner.csv"),
                "--out", "/kaggle/working/brats2mm", "--workers", "4", "--max-minutes", "640"], check=True)
