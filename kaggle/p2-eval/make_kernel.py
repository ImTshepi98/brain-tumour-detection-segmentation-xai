"""Write a PLAN2 evaluation kernel folder (kaggle/p2-eval-<run>-<stage>).
python kaggle/p2-eval/make_kernel.py RUN STAGE [--cpu] [--xai-small]
STAGE val attaches brats-dual-p2-train-RUN; STAGE test also attaches brats-dual-p2-eval-RUN-val."""
import json, sys
from pathlib import Path

here = Path(__file__).parent
run, stage = sys.argv[1], sys.argv[2]
flags = set(sys.argv[3:])
assert stage in ("val", "test")
d = here.parent / f"p2-eval-{run}-{stage}"
d.mkdir(exist_ok=True)
(d / "run.py").write_text((here / "run_template.py").read_text().replace("__RUN__", run).replace("__STAGE__", stage)
                          .replace("__XAI_SMALL__", str("--xai-small" in flags or run == "dryrun")))
sources = ["leweletp/brats-dual-prep-brats-2mm"] + [f"leweletp/brats-dual-prep-ixi-s{k}" for k in range(3)] \
    + [f"leweletp/brats-dual-sdt-brats-s{k}" for k in range(3)] + [f"leweletp/brats-dual-p2-train-{run}"]
if stage == "test":
    sources.append(f"leweletp/brats-dual-p2-eval-{run}-val")
gpu = "--cpu" not in flags
meta = {"id": f"leweletp/brats-dual-p2-eval-{run}-{stage}", "title": f"brats-dual-p2-eval-{run}-{stage}",
        "code_file": "run.py", "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": gpu, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": ["leweletp/brats-dual-code"], "competition_sources": [], "kernel_sources": sources}
if gpu:
    meta["machine_shape"] = "NvidiaTeslaT4"
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print(d)
