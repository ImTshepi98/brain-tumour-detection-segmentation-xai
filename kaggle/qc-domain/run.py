"""Phase 3b/3c kernel: BraTS vs IXI domain-gap classifier on train+val. Thin wrapper around src/qc_domain.py.
v2: shared SynthStrip boundary (BORDER from src/calibrate_border.py on BraTS train rows)."""
import shutil, subprocess, sys
from pathlib import Path

INP = Path("/kaggle/input")
CODE = next(p.parent for p in INP.rglob("qc_domain.py"))
DATA = CODE.parent / "data"
print("code", CODE, "commit", (CODE.parent / "GIT_COMMIT").read_text().strip(), flush=True)
shutil.copytree(CODE, "/kaggle/working/src_used", dirs_exist_ok=True)
brats = next(p.parent for p in INP.rglob("brats2mm/crop.json"))
ixi = sorted(str(p.parent) for p in INP.rglob("ixi2mm/template.json"))
sdt_b = sorted(str(p.parent) for p in INP.rglob("sdt/summary_shard*.json"))
print("brats", brats, "ixi", ixi, "brats sdt", sdt_b, flush=True)
BORDER = 0
EXTRA = [] if BORDER is None else ["--border", str(BORDER), "--brats-sdt-dirs", *sdt_b]
subprocess.run([sys.executable, "-u", str(CODE / "qc_domain.py"), "--brats-dir", str(brats), "--ixi-dirs", *ixi,
                "--splits", str(DATA / "splits.csv"), "--out", "/kaggle/working/domain", "--workers", "4"] + EXTRA,
               check=True)
