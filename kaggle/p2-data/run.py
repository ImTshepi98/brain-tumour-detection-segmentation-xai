"""PLAN2 P1 kernel (CPU): midline, hemisphere counts per split, montages, fixed val pairing file and 12 QC
panels of pasted samples. Thin wrapper around src/p2_data.py; shared SynthStrip boundary b = 0 (as PLAN)."""
import shutil, subprocess, sys
from pathlib import Path

INP = Path("/kaggle/input")
CODE = next(p.parent for p in INP.rglob("p2_data.py") if "brats-dual-code" in p.parts)
DATA = CODE.parent / "data"
print("code", CODE, "commit", (CODE.parent / "GIT_COMMIT").read_text().strip(), flush=True)
shutil.copytree(CODE, "/kaggle/working/src_used", dirs_exist_ok=True)
brats = next(p.parent for p in INP.rglob("brats2mm/crop.json"))
ixi = sorted(str(p.parent) for p in INP.rglob("ixi2mm/template.json"))
sdt_b = sorted(str(p.parent) for p in INP.rglob("sdt/summary_shard*.json"))
print("brats", brats, "ixi", ixi, "brats sdt", sdt_b, flush=True)
subprocess.run([sys.executable, "-u", str(CODE / "p2_data.py"), "--brats-dir", str(brats), "--ixi-dirs", *ixi,
                "--brats-sdt-dirs", *sdt_b, "--splits", str(DATA / "splits.csv"), "--border", "0",
                "--out", "/kaggle/working/p1", "--workers", "4"], check=True)
