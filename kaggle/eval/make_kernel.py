"""Write an evaluation kernel folder.
python kaggle/eval/make_kernel.py RUN STAGE [--controls] [--export] [--cpu]
STAGE val: attaches brats-dual-train-RUN; STAGE test: also attaches brats-dual-eval-RUN-val.
GPU by default (PLAN 6 times inference on GPU); --cpu for a val stage to save quota."""
import json, sys
from pathlib import Path

here = Path(__file__).parent
run, stage = sys.argv[1], sys.argv[2]
flags = set(sys.argv[3:])
assert stage in ("val", "test")
d = here.parent / f"eval-{run}-{stage}"
d.mkdir(exist_ok=True)
tpl = (here / "run_template.py").read_text()
(d / "run.py").write_text(tpl.replace("__RUN__", run).replace("__STAGE__", stage)
                          .replace("__CONTROLS__", str("--controls" in flags))
                          .replace("__EXPORT__", str("--export" in flags)))
sources = ["leweletp/brats-dual-prep-brats-2mm"] + [f"leweletp/brats-dual-prep-ixi-s{k}" for k in range(3)] \
    + [f"leweletp/brats-dual-sdt-brats-s{k}" for k in range(3)] + [f"leweletp/brats-dual-train-{run}"]
if stage == "test":
    sources.append(f"leweletp/brats-dual-eval-{run}-val")
gpu = "--cpu" not in flags
meta = {"id": f"leweletp/brats-dual-eval-{run}-{stage}", "title": f"brats-dual-eval-{run}-{stage}",
        "code_file": "run.py", "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": gpu, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": ["leweletp/brats-dual-code"], "competition_sources": [], "kernel_sources": sources}
if gpu:
    meta["machine_shape"] = "NvidiaTeslaT4"
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print(d)
