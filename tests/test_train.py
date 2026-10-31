import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import train  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SMALL = ["model.channels=[4,8,16,32,64]"]
DEBUG_DATA = "debug={n_brats: 8, n_ixi: 8, shape: [32, 32, 32]}"


def _metrics(out):
    return [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]


def test_debug_run_and_resume(tmp_path):
    out = tmp_path / "run"
    d = train.main(["--debug", "--out", str(out), "--set", *SMALL, "optim.epochs=3", "--max-minutes", "0.0001"])
    assert d["stopped"] == "time_budget" and d["epochs_completed"] == 1 and not d["test_read"]
    for f in ("runtime.json", "last.pt", "best.pt", "val_last.json", "val_best.json", "done.json"):
        assert (out / f).exists(), f
    st = torch.load(out / "last.pt", weights_only=False)
    assert st["epoch"] == 0 and st["protocol_hash"] == d["protocol_hash"]
    val = json.loads((out / "val_last.json").read_text())
    assert all(r["id"].startswith(("BraTS", "IXI")) for r in val) and any("WT" in r for r in val)
    # resume continues at epoch 1; max_minutes is not part of the protocol
    d2 = train.main(["--debug", "--out", str(out), "--set", *SMALL, "optim.epochs=3", "--resume", "auto"])
    assert d2["protocol_hash"] == d["protocol_hash"] and [m["epoch"] for m in _metrics(out)] == [0, 1, 2]
    # a changed protocol refuses to resume
    with pytest.raises(AssertionError, match="protocol"):
        train.main(["--debug", "--out", str(out), "--set", *SMALL, "optim.epochs=4", "--resume", "auto"])


@pytest.mark.parametrize("cfg", ["segonly", "clsonly", "noattn"])
def test_variants_run(tmp_path, cfg):
    over = [*SMALL, "border=0", "batch_size=2", "data.workers=1", "optim.epochs=1", DEBUG_DATA]
    d = train.main(["--debug", "--config", str(ROOT / f"configs/train/{cfg}.yaml"), "--out", str(tmp_path),
                    "--set", *over])
    m = _metrics(tmp_path)[0]
    assert d["epochs_completed"] == 1
    if cfg == "segonly":
        assert m["train_cls"] is None and m["val_detection"] == {} and m["val_mean_dsc"] >= 0
    if cfg == "clsonly":
        assert m["train_seg"] is None and m["val_dsc"] == {} and 0 <= m["val_score"] <= 1


def test_border_must_be_set(tmp_path):
    with pytest.raises(AssertionError, match="border"):
        train.main(["--debug", "--config", str(ROOT / "configs/train/main.yaml"), "--out", str(tmp_path),
                    "--set", DEBUG_DATA])
