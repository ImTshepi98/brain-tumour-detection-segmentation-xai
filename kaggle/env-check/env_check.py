"""Phase 0 trivial CPU kernel: versions, disk, RAM, /kaggle/input layout.

Also counts IXI T1/T2 files in the attached fallback dataset (mdnakib/ixi-dataset)
so the no-token fallback can be judged without another run. Writes env.json.
"""
import json
import os
import platform
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

out = {"python": sys.version, "platform": platform.platform(), "cpu_count": os.cpu_count()}
for mod in ("numpy", "torch", "monai", "nibabel", "scipy", "sklearn", "skimage", "SimpleITK", "ants", "huggingface_hub"):
    try:
        out[f"ver_{mod}"] = __import__(mod).__version__
    except Exception as e:  # noqa: BLE001
        out[f"ver_{mod}"] = f"missing ({type(e).__name__})"
try:
    import torch
    out["cuda"] = torch.cuda.is_available()
except Exception:  # noqa: BLE001
    out["cuda"] = None

mem = Path("/proc/meminfo").read_text().splitlines()[:3] if Path("/proc/meminfo").exists() else []
out["meminfo"] = mem
for p in ("/kaggle/working", "/kaggle/temp", "/kaggle/tmp", "/tmp", "/kaggle/input"):
    if os.path.exists(p):
        u = shutil.disk_usage(p)
        out[f"disk_{p}"] = {"total_gb": round(u.total / 1e9, 1), "free_gb": round(u.free / 1e9, 1)}

# Layout of /kaggle/input to depth 3, with file counts per directory.
layout = {}
root = Path("/kaggle/input")
for d in sorted(root.rglob("*")) if root.exists() else []:
    rel = d.relative_to(root)
    if d.is_dir() and len(rel.parts) <= 3:
        files = [f for f in d.iterdir() if f.is_file()]
        layout[str(rel)] = {"n_files": len(files), "examples": [f.name for f in files[:5]],
                            "bytes_examples": [f.stat().st_size for f in files[:5]]}
    elif d.is_file() and len(rel.parts) <= 2:
        layout[str(rel)] = {"size": d.stat().st_size}
out["input_layout"] = layout

# IXI fallback check: subjects with both T1 and T2.
mods = {}
for f in root.rglob("IXI*.nii*") if root.exists() else []:
    m = re.match(r"(IXI\d{3})-([A-Za-z]+)-\d+-(T1|T2|PD|MRA|DTI)", f.name)
    if m:
        mods.setdefault(m.group(1), set()).add(m.group(3))
        out.setdefault("ixi_sites", Counter())[m.group(2)] += 1
out["ixi_modality_counts"] = dict(Counter(m for s in mods.values() for m in s))
out["ixi_paired_t1_t2"] = sum(1 for s in mods.values() if {"T1", "T2"} <= s)
out["ixi_other_files"] = [str(f.relative_to(root)) for f in root.rglob("*")
                          if f.is_file() and f.suffix.lower() in (".xls", ".xlsx", ".csv", ".txt", ".tar", ".zip")][:50]

Path("/kaggle/working/env.json").write_text(json.dumps(out, indent=2, default=dict))
print(json.dumps(out, indent=2, default=dict))
