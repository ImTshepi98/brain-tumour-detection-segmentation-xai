"""Write a GPU training kernel folder.
python kaggle/train/make_kernel.py NAME CONFIG BORDER MAX_MINUTES ['["--set-style", "overrides"]']
e.g. smoke: make_kernel.py smoke main.yaml 0 15 '["data.limit_train=20", "data.limit_val=10", "optim.epochs=1"]'
EXTRA items are appended after --set (key=value overrides)."""
import json, sys
from pathlib import Path

here = Path(__file__).parent
name, config, border, minutes = sys.argv[1:5]
extra = json.loads(sys.argv[5]) if len(sys.argv) > 5 else []
d = here.parent / f"train-{name}"
d.mkdir(exist_ok=True)
tpl = (here / "run_template.py").read_text()
(d / "run.py").write_text(tpl.replace("__NAME__", name).replace("__CONFIG__", config)
                          .replace("__BORDER__", repr(None if border == "null" else float(border)))
                          .replace("__MAX_MINUTES__", str(float(minutes))).replace("__EXTRA__", repr(extra)))
meta = {"id": f"leweletp/brats-dual-train-{name}", "title": f"brats-dual-train-{name}", "code_file": "run.py",
        "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
        "machine_shape": "NvidiaTeslaT4", "enable_tpu": False, "enable_internet": True,
        "dataset_sources": ["leweletp/brats-dual-code"], "competition_sources": [],
        "kernel_sources": ["leweletp/brats-dual-prep-brats-2mm"]
        + [f"leweletp/brats-dual-prep-ixi-s{k}" for k in range(3)]
        + [f"leweletp/brats-dual-sdt-brats-s{k}" for k in range(3)]}
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print(d)
