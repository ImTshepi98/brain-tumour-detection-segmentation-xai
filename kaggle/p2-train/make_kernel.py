"""Write a PLAN2 GPU training kernel folder (kaggle/p2-train-<name>); same wrapper as PLAN's training kernels
(resume from the furthest attached last.pt, memory budget), running src/train_p2.py.
python kaggle/p2-train/make_kernel.py NAME CONFIG BORDER MIDLINE MAX_MINUTES ['["overrides"]'] [CONTINUE_FROM]"""
import json, sys
from pathlib import Path

here = Path(__file__).parent
name, config, border, midline, minutes = sys.argv[1:6]
extra = json.loads(sys.argv[6]) if len(sys.argv) > 6 else []
srcs = [c for c in sys.argv[7].split(",") if c] if len(sys.argv) > 7 else []
cont = [f"leweletp/{c}" for c in srcs if not c.startswith("dataset:")]
cont_ds = [f"leweletp/{c[len('dataset:'):]}" for c in srcs if c.startswith("dataset:")]
d = here.parent / f"p2-train-{name}"
d.mkdir(exist_ok=True)
tpl = (here.parent / "train" / "run_template.py").read_text()
assert tpl.count('str(CODE / "train.py")') == 1
tpl = tpl.replace('str(CODE / "train.py")', 'str(CODE / "train_p2.py")').replace(
    '"Phase 5 GPU training: run __NAME__', '"PLAN2 GPU training: run __NAME__').replace("src/train.py", "src/train_p2.py")
(d / "run.py").write_text(tpl.replace("__NAME__", name).replace("__CONFIG__", config)
                          .replace("__BORDER__", repr(float(border))).replace("__MAX_MINUTES__", str(float(minutes)))
                          .replace("__EXTRA__", repr([f"midline={int(midline)}"] + extra)).replace("__CONTINUE__", repr(srcs)))
meta = {"id": f"leweletp/brats-dual-p2-train-{name}", "title": f"brats-dual-p2-train-{name}", "code_file": "run.py",
        "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
        "machine_shape": "NvidiaTeslaT4", "enable_tpu": False, "enable_internet": True,
        "dataset_sources": ["leweletp/brats-dual-code"] + cont_ds, "competition_sources": [],
        "kernel_sources": ["leweletp/brats-dual-prep-brats-2mm"]
        + [f"leweletp/brats-dual-prep-ixi-s{k}" for k in range(3)]
        + [f"leweletp/brats-dual-sdt-brats-s{k}" for k in range(3)] + cont}
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print(d)
