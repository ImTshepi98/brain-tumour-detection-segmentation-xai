"""Phase 6/7 evaluation of run main, stage val. Thin wrapper around evaluate.py / explain.py.
val:  evaluate val (threshold = max F1 on val; controls if CONTROLS), Grad-CAM++ checks on all val cases.
test: needs the val stage's output attached (results_val.json); evaluates test ONCE with the frozen
      threshold, Grad-CAM++ checks and overlays on test, and exports GUI cases if EXPORT."""
import json, os, shutil, subprocess, sys
from pathlib import Path

RUN, STAGE, CONTROLS, EXPORT = "main", "val", True, False
# BRATS_DUAL_DRYRUN=<dir>: local dry run against <dir>/input, writing to <dir>/working (PLAN rule 3)
DRY = os.environ.get("BRATS_DUAL_DRYRUN")
if not DRY:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "monai==1.6.1"], check=True)
INP = Path(DRY, "input") if DRY else Path("/kaggle/input")
WORK = Path(DRY, "working") if DRY else Path("/kaggle/working")
CODE = next(p.parent for p in INP.rglob("evaluate.py") if "brats-dual-code" in p.parts)  # the code dataset only (attached kernel outputs carry old src_used/ copies)
DATA = CODE.parent / "data"
print("code", CODE, "commit", (CODE.parent / "GIT_COMMIT").read_text().strip(), flush=True)
shutil.copytree(CODE, WORK / "src_used", dirs_exist_ok=True)
brats = next(p.parent for p in INP.rglob("brats2mm/crop.json"))
ixi = sorted(str(p.parent) for p in INP.rglob("ixi2mm/template.json"))
sdt_b = sorted(str(p.parent) for p in INP.rglob("sdt/summary_shard*.json"))
assert len(ixi) == 3 and len(sdt_b) == 3, (ixi, sdt_b)
ckpt = [p for p in INP.rglob("run/best.pt") if f"train-{RUN}" in str(p)]
assert len(ckpt) == 1, f"attach exactly one brats-dual-train-{RUN} output: {ckpt}"
ckpt = ckpt[0]
DATA_ARGS = ["--brats-dir", str(brats), "--ixi-dirs", *ixi, "--brats-sdt-dirs", *sdt_b,
             "--splits", str(DATA / "splits.csv"), "--ages", str(DATA / "age_ixi.csv")]
COUNT = ["--no-count-check"] if DRY else []
out = WORK / "eval"
out.mkdir(parents=True, exist_ok=True)
shutil.copy(ckpt.parent / "metrics.jsonl", out / "train_metrics.jsonl")


def run(script, *a):
    subprocess.run([sys.executable, "-u", str(CODE / script), "--ckpt", str(ckpt), *DATA_ARGS, *a], check=True)


if STAGE == "val":
    run("evaluate.py", "--split", "val", "--out", str(out), *COUNT, *(["--controls"] if CONTROLS else []))
    run("explain.py", "--split", "val", "--results", str(out / "results_val.json"),
        "--cases-csv", str(out / "cases_val.csv"), "--out", str(out / "xai"))
else:
    assert STAGE == "test"
    assert not list(INP.rglob("eval/test_lock.json")), "an attached output already holds a test evaluation"
    val = [p for p in INP.rglob("eval/results_val.json") if f"eval-{RUN}" in str(p)]
    assert len(val) == 1, f"attach the val stage output (brats-dual-eval-{RUN}-val): {val}"
    assert json.loads(val[0].read_text())["protocol_hash"] == __import__("torch").load(
        ckpt, map_location="cpu", weights_only=False)["protocol_hash"], "val results are from another checkpoint"
    shutil.copy(val[0], out / "results_val.json")
    run("evaluate.py", "--split", "test", "--confirm-test", "--threshold-from", str(out / "results_val.json"),
        "--out", str(out), *COUNT)
    run("explain.py", "--split", "test", "--confirm-test", "--results", str(out / "results_test.json"),
        "--cases-csv", str(out / "cases_test.csv"), "--out", str(out / "xai"))
    if EXPORT:
        sys.path.insert(0, str(CODE))
        import numpy as np, torch
        import datasets as ds, infer
        cfg = torch.load(ckpt, map_location="cpu", weights_only=False)["config"]
        rows = ds.read_splits(DATA / "splits.csv")
        picked = json.loads((out / "xai" / "xai_summary_test.json").read_text())["meta"]["overlay_cases"]
        test_ids = sorted(r["subject"] for r in rows if r["split"] == "test")
        extra = [s for s in np.random.default_rng(42).permutation(test_ids) if s not in picked][: 10 - len(picked)]
        ids = list(picked) + [str(s) for s in extra]
        cohort = ds.Cohort.load(rows, ("test",), brats, ixi, sdt_b, border=cfg["border"], allow_test=True,
                                subjects=set(ids))
        infer.export_cases(cohort, ids, out / "gui_cases")
        print("exported", ids, flush=True)
print("EVALUATION COMPLETE", RUN, STAGE, flush=True)
