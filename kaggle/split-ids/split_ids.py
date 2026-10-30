"""Phase 0 ID listing (CPU). Reads names only, never image data, so the test split stays locked.

Outputs in /kaggle/working:
  brats_tar_subjects.json  every BraTS2021_xxxxx folder in the tar, with its member names
  ixi_files.json           every IXI NIfTI in mdnakib/ixi-dataset (path, subject, site, modality)
  IXI.xls / ixi_demographics_source.json  official IXI demographics sheet, if reachable
"""
import json
import re
import tarfile
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

OUT = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
t0 = time.time()

tars = list(INPUT.rglob("BraTS2021_Training_Data.tar"))
assert len(tars) == 1, tars
members = defaultdict(list)
with tarfile.open(tars[0], "r:") as tf:  # uncompressed tar: header walk only
    for m in tf:
        parts = Path(m.name).parts
        sub = next((p for p in parts if re.fullmatch(r"BraTS2021_\d{5}", p)), None)
        if sub and m.isfile():
            members[sub].append(m.name)
(OUT / "brats_tar_subjects.json").write_text(json.dumps(dict(sorted(members.items())), indent=1))
print(f"brats subjects in tar: {len(members)} ({time.time() - t0:.0f}s)")

rows = []
for f in sorted(INPUT.rglob("IXI*.nii*")):
    m = re.match(r"(IXI\d{3})-([A-Za-z]+)-(\d+)-(T1|T2|PD|MRA|DTI)", f.name)
    if m:
        rows.append(dict(path=str(f.relative_to(INPUT)), subject=m.group(1), site=m.group(2),
                         scan=m.group(3), modality=m.group(4), size=f.stat().st_size))
(OUT / "ixi_files.json").write_text(json.dumps(rows, indent=1))
subj = defaultdict(set)
for r in rows:
    subj[r["subject"]].add(r["modality"])
print(f"ixi files: {len(rows)}; paired T1+T2 subjects: {sum({'T1', 'T2'} <= s for s in subj.values())}")

# Official IXI demographics (brain-development.org); recorded with its source URL.
src = {}
for url in ("http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI.xls",
            "https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI.xls"):
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
        (OUT / "IXI.xls").write_bytes(data)
        src = dict(url=url, bytes=len(data), fetched_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        break
    except Exception as e:  # noqa: BLE001
        src[url] = f"{type(e).__name__}: {e}"
(OUT / "ixi_demographics_source.json").write_text(json.dumps(src, indent=1))
print(src)
print(f"done in {time.time() - t0:.0f}s")
