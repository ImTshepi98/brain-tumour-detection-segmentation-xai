"""Local dry run of the training and evaluation kernel wrappers (PLAN rule 3).
Builds a fake /kaggle/input tree from synthetic 2 mm data in the same layout as the real kernel
outputs, then runs: train-dryrun (smoke settings) -> eval-dryrun-val -> eval-dryrun-test.
  python kaggle/dryrun.py [--out runs/kernel-dryrun]"""
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import synthetic  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(ROOT / "runs/kernel-dryrun"))
args = ap.parse_args()
base = Path(args.out)
shutil.rmtree(base, ignore_errors=True)
inp, nb = base / "input", base / "input/notebooks/leweletp"
fake = synthetic.write_fake_2mm(base / "fake", n_brats=12, n_ixi=12)
code = inp / "datasets/leweletp/brats-dual-code"
shutil.copytree(ROOT / "src", code / "src", ignore=shutil.ignore_patterns("__pycache__"))
shutil.copytree(ROOT / "configs", code / "configs")
(code / "data").mkdir()
shutil.copy(fake["splits"], code / "data/splits.csv")
(code / "GIT_COMMIT").write_text("dryrun")
ixi_rows = [l.split(",") for l in Path(fake["splits"]).read_text().splitlines()[1:] if ",ixi," in l]
with open(code / "data/age_ixi.csv", "w") as f:
    f.write("subject,age\n" + "".join(f"{r[0]},{(25, 50, 80)[k % 3]}\n" for k, r in enumerate(ixi_rows)))
shutil.copytree(fake["brats_dir"], nb / "brats-dual-prep-brats-2mm/brats2mm")
ixi_files = sorted(Path(fake["ixi_dir"], "x").glob("*.npy"))
brats_sdt = sorted(Path(fake["brats_sdt_dir"], "sdt").glob("*.npy"))
for k in range(3):
    d = nb / f"brats-dual-prep-ixi-s{k}/ixi2mm"
    for sub in ("x", "sdt"):
        (d / sub).mkdir(parents=True)
    (d / "template.json").write_text("{}")
    for p in ixi_files[k::3]:
        shutil.copy(p, d / "x" / p.name)
        shutil.copy(Path(fake["ixi_dir"], "sdt", p.name), d / "sdt" / p.name)
    s = nb / f"brats-dual-sdt-brats-s{k}/sdt"
    (s / "sdt").mkdir(parents=True)
    (s / f"summary_shard{k}.json").write_text("{}")
    for p in brats_sdt[k::3]:
        shutil.copy(p, s / "sdt" / p.name)

env = dict(os.environ, BRATS_DUAL_DRYRUN=str(base))
py = sys.executable


def kernel(maker, *a):
    folder = Path(subprocess.run([py, str(ROOT / maker), *a], capture_output=True, text=True, check=True).stdout.strip())
    try:
        shutil.rmtree(base / "working", ignore_errors=True)
        subprocess.run([py, str(folder / "run.py")], env=env, check=True)
    finally:
        shutil.rmtree(folder)
    return base / "working"


small = json.dumps(["model.channels=[4,8,16,32,64]", "batch_size=2", "data.workers=1", "optim.epochs=3"])
w = kernel("kaggle/train/make_kernel.py", "dryrun", "main.yaml", "0", "0.0001", small)  # stops after epoch 0
assert json.loads((w / "run/done.json").read_text())["stopped"] == "time_budget"
shutil.copytree(w, nb / "brats-dual-train-dryrun-v1")  # whole output incl. src_used/, as Kaggle attaches it
w = kernel("kaggle/train/make_kernel.py", "dryrun-r1", "main.yaml", "0", "0.0001", small, "brats-dual-train-dryrun-v1")
assert json.loads((w / "run/done.json").read_text())["next_epoch"] == 2  # resumed at 1, stopped after it
resume_ds = inp / "datasets/leweletp/brats-dual-dryrun-resume"  # a crashed run's run/ uploaded as a dataset
shutil.copytree(w / "run", resume_ds / "run")
# both the epoch-0 kernel output and the epoch-1 dataset are attached: must continue from epoch 1
w = kernel("kaggle/train/make_kernel.py", "dryrun-r2", "main.yaml", "0", "5", small,
           "brats-dual-train-dryrun-v1,dataset:brats-dual-dryrun-resume")
assert [json.loads(l)["epoch"] for l in (w / "run/metrics.jsonl").read_text().splitlines()] == [0, 1, 2]
assert json.loads((w / "run/done.json").read_text())["stopped"] == "epochs_done"
shutil.rmtree(nb / "brats-dual-train-dryrun-v1")
shutil.rmtree(resume_ds)
shutil.copytree(w, nb / "brats-dual-train-dryrun")
w = kernel("kaggle/eval/make_kernel.py", "dryrun", "val", "--controls", "--cpu")
assert (w / "eval/results_val.json").exists() and (w / "eval/xai/xai_metrics_val.md").exists()
shutil.copytree(w, nb / "brats-dual-eval-dryrun-val")
w = kernel("kaggle/eval/make_kernel.py", "dryrun", "test", "--export", "--cpu")
res = json.loads((w / "eval/results_test.json").read_text())
assert res["split"] == "test" and res["threshold_rule"] == "frozen_from_val"
assert len(list((w / "eval/gui_cases").glob("*.npz"))) >= 6
print("KERNEL DRY RUN OK", base, flush=True)
