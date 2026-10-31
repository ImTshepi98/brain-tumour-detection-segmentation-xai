"""Write a GPU training kernel folder.
python kaggle/train/make_kernel.py NAME CONFIG BORDER MAX_MINUTES ['["--set-style", "overrides"]'] [CONTINUE_FROM]
e.g. smoke: make_kernel.py smoke main.yaml 0 15 '["data.limit_train=20", "data.limit_val=10", "optim.epochs=1"]'
EXTRA items are appended after --set (key=value overrides). CONTINUE_FROM is a comma-separated
list of earlier outputs holding run/last.pt: a completed kernel slug (e.g. brats-dual-train-main-r2) or
dataset:<slug> for a crashed run's run/ folder uploaded as a private dataset (Kaggle does not attach an
errored kernel's output). run.py resumes from the highest epoch among them."""
import json, sys
from pathlib import Path

here = Path(__file__).parent
name, config, border, minutes = sys.argv[1:5]
extra = json.loads(sys.argv[5]) if len(sys.argv) > 5 else []
srcs = [c for c in sys.argv[6].split(",") if c] if len(sys.argv) > 6 else []
cont = [f"leweletp/{c}" for c in srcs if not c.startswith("dataset:")]
cont_ds = [f"leweletp/{c[len('dataset:'):]}" for c in srcs if c.startswith("dataset:")]
d = here.parent / f"train-{name}"
d.mkdir(exist_ok=True)
tpl = (here / "run_template.py").read_text()
(d / "run.py").write_text(tpl.replace("__NAME__", name).replace("__CONFIG__", config)
                          .replace("__BORDER__", repr(None if border == "null" else float(border)))
                          .replace("__MAX_MINUTES__", str(float(minutes))).replace("__EXTRA__", repr(extra)))
meta = {"id": f"leweletp/brats-dual-train-{name}", "title": f"brats-dual-train-{name}", "code_file": "run.py",
        "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
        "machine_shape": "NvidiaTeslaT4", "enable_tpu": False, "enable_internet": True,
        "dataset_sources": ["leweletp/brats-dual-code"] + cont_ds, "competition_sources": [],
        "kernel_sources": ["leweletp/brats-dual-prep-brats-2mm"]
        + [f"leweletp/brats-dual-prep-ixi-s{k}" for k in range(3)]
        + [f"leweletp/brats-dual-sdt-brats-s{k}" for k in range(3)] + cont}
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print(d)
