"""PLAN2 P4/P6 evaluation of run __RUN__, stage __STAGE__. Thin wrapper around src/eval_p2.py.
val:  thresholds chosen on val (worst-group rule), all checks S1-S9 and G1, XAI comparison.
test: needs the val stage's output attached; evaluates test ONCE with the frozen thresholds, the same checks
      and the XAI comparison in the same read."""
import json, os, shutil, subprocess, sys
from pathlib import Path

RUN, STAGE, XAI_SMALL = "__RUN__", "__STAGE__", __XAI_SMALL__
DRY = os.environ.get("BRATS_DUAL_DRYRUN")
if not DRY:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "monai==1.6.1"], check=True)
INP = Path(DRY, "input") if DRY else Path("/kaggle/input")
WORK = Path(DRY, "working") if DRY else Path("/kaggle/working")
CODE = next(p.parent for p in INP.rglob("eval_p2.py") if "brats-dual-code" in p.parts)
DATA = CODE.parent / "data"
print("code", CODE, "commit", (CODE.parent / "GIT_COMMIT").read_text().strip(), flush=True)
shutil.copytree(CODE, WORK / "src_used", dirs_exist_ok=True)
brats = next(p.parent for p in INP.rglob("brats2mm/crop.json"))
ixi = sorted(str(p.parent) for p in INP.rglob("ixi2mm/template.json"))
sdt_b = sorted(str(p.parent) for p in INP.rglob("sdt/summary_shard*.json"))
assert len(ixi) == 3 and len(sdt_b) == 3, (ixi, sdt_b)
ckpt = [p for p in INP.rglob("run/best.pt") if f"p2-train-{RUN}" in str(p)]
assert len(ckpt) == 1, f"attach exactly one brats-dual-p2-train-{RUN} output: {ckpt}"
ckpt = ckpt[0]
ARGS = ["--ckpt", str(ckpt), "--brats-dir", str(brats), "--ixi-dirs", *ixi, "--brats-sdt-dirs", *sdt_b,
        "--splits", str(DATA / "splits.csv"), "--xai"] + (["--no-count-check"] if DRY else []) \
    + (["--xai-small", "--s7-cases", "3"] if XAI_SMALL else [])
out = WORK / "eval"
out.mkdir(parents=True, exist_ok=True)
shutil.copy(ckpt.parent / "metrics.jsonl", out / "train_metrics.jsonl")
if not DRY:
    subprocess.run(["nvidia-smi"], check=False)
if STAGE == "val":
    subprocess.run([sys.executable, "-u", str(CODE / "eval_p2.py"), *ARGS, "--split", "val", "--out", str(out)],
                   check=True)
else:
    assert STAGE == "test"
    assert not list(INP.rglob("eval/test_lock.json")), "an attached output already holds a test evaluation"
    val = [p for p in INP.rglob("eval/results_val.json") if f"p2-eval-{RUN}" in str(p)]
    assert len(val) == 1, f"attach the val stage output (brats-dual-p2-eval-{RUN}-val): {val}"
    shutil.copy(val[0], out / "results_val.json")
    shutil.copy(val[0].parent / "val_checks.md", out / "val_checks.md")
    subprocess.run([sys.executable, "-u", str(CODE / "eval_p2.py"), *ARGS, "--split", "test", "--confirm-test",
                    "--threshold-from", str(out / "results_val.json"), "--out", str(out)], check=True)
