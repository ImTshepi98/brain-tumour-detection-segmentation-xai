"""Write one kernel folder per shard: python kaggle/prep-ixi/make_shards.py NSHARDS [extra args as JSON list]."""
import json, sys
from pathlib import Path

here = Path(__file__).parent
n = int(sys.argv[1])
extra = sys.argv[2] if len(sys.argv) > 2 else "[]"
tpl = (here / "run_template.py").read_text()
for k in range(n):
    d = here / f"s{k}"
    d.mkdir(exist_ok=True)
    (d / "run.py").write_text(tpl.replace("__SHARD__", str(k)).replace("__NSHARDS__", str(n)).replace("__EXTRA__", extra))
    meta = {"id": f"leweletp/brats-dual-prep-ixi-s{k}", "title": f"brats-dual-prep-ixi-s{k}", "code_file": "run.py",
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False,
            "enable_tpu": False, "enable_internet": True,
            "dataset_sources": ["dschettler8845/brats-2021-task1", "mdnakib/ixi-dataset", "leweletp/brats-dual-code"],
            "competition_sources": [], "kernel_sources": ["leweletp/fork-of-notebook6794c43f52"]}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(d)
