"""Local dry run of the PLAN2 kernel wrappers (rule 1.1: CPU dry run before every push).
Builds a fake /kaggle/input tree from synthetic 2 mm data with one-sided tumours, then runs: p2 training
(stops after epoch 0 on the time budget) -> p2 training continued from that output -> [eval stages when given].
  python kaggle/dryrun_p2.py [--out runs/kernel-dryrun-p2] [--eval]"""
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import synthetic  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(ROOT / "runs/kernel-dryrun-p2"))
ap.add_argument("--eval", action="store_true", help="also run the eval kernels (val, then test)")
args = ap.parse_args()
base = Path(args.out)
shutil.rmtree(base, ignore_errors=True)
inp, nb = base / "input", base / "input/notebooks/leweletp"
fake = synthetic.write_fake_2mm(base / "fake", n_brats=24, n_ixi=12, lateral=True)
code = inp / "datasets/leweletp/brats-dual-code"
shutil.copytree(ROOT / "src", code / "src", ignore=shutil.ignore_patterns("__pycache__"))
shutil.copytree(ROOT / "configs", code / "configs")
(code / "data").mkdir()
shutil.copy(fake["splits"], code / "data/splits.csv")
(code / "GIT_COMMIT").write_text("dryrun")
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


small = json.dumps(["width=16", "model.channels=[4,8,16,32,64]", "data.workers=1", "data.samples_per_epoch=8",
                    "optim.epochs=3"])
w = kernel("kaggle/p2-train/make_kernel.py", "dryrun", "p2_main.yaml", "0", "16", "0.0001", small)
assert json.loads((w / "run/done.json").read_text())["stopped"] == "time_budget"
assert (w / "run/pairs_val.csv").exists()
shutil.copytree(w, nb / "brats-dual-p2-train-dryrun-v1")
w = kernel("kaggle/p2-train/make_kernel.py", "dryrun-r1", "p2_main.yaml", "0", "16", "5", small,
           "brats-dual-p2-train-dryrun-v1")
assert [json.loads(l)["epoch"] for l in (w / "run/metrics.jsonl").read_text().splitlines()] == [0, 1, 2]
assert json.loads((w / "run/done.json").read_text())["stopped"] == "epochs_done"
shutil.rmtree(nb / "brats-dual-p2-train-dryrun-v1")
shutil.copytree(w, base / "keep-train")  # parked: the adv kernel must not see (and resume from) it
adv = json.dumps(json.loads(small) + ["optim.epochs=1"])
w = kernel("kaggle/p2-train/make_kernel.py", "dryrun-adv", "p2_adv.yaml", "0", "16", "5", adv)
assert json.loads((w / "run/metrics.jsonl").read_text().splitlines()[0])["train_adv"] > 0
shutil.move(str(base / "keep-train"), str(nb / "brats-dual-p2-train-dryrun"))
if args.eval:
    w = kernel("kaggle/p2-eval/make_kernel.py", "dryrun", "val", "--cpu")
    assert (w / "eval/val_checks.md").exists()
    shutil.copytree(w, nb / "brats-dual-p2-eval-dryrun-val")
    w = kernel("kaggle/p2-eval/make_kernel.py", "dryrun", "test", "--cpu")
    assert json.loads((w / "eval/results_test.json").read_text())["split"] == "test"
print("P2 KERNEL DRY RUN OK", base, flush=True)
