import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common  # noqa: E402
import prep_brats_2mm as prep_b  # noqa: E402
import sdt_brats  # noqa: E402
import synthetic as syn  # noqa: E402


def test_shared_mask_shrinks_and_keeps_tumour():
    shape = (16, 16, 16)
    rng = np.random.default_rng(0)
    yy = np.indices(shape) - 7.5
    r = np.sqrt((yy ** 2).sum(0))
    brain = r < 7
    x = np.zeros((4, *shape), np.float32)
    x[:, brain] = rng.normal(1, 2, (4, int(brain.sum())))
    sdt = (r - 7) * 2.0  # mm on a 2 mm grid
    y = np.zeros(shape, np.uint8)
    y[(np.abs(yy[0] - 6) < 1) & (np.abs(yy[1]) < 1) & (np.abs(yy[2]) < 1)] = 7  # tumour at the edge
    xn, b = common.shared_mask(x, sdt, -2.0, y)
    assert b.sum() < brain.sum() and (b <= brain).all()
    assert b[(y & 4) > 0].all()
    for c in range(4):
        v = xn[c][xn[c] != 0]
        assert abs(v.mean()) < 1e-4 and abs(v.std() - 1) < 1e-3
    assert not xn[:, ~b].any()


def test_pool_sdt_fill():
    sdt = np.full((8, 8, 8), -5.0, np.float32)
    p = common.pool_sdt(sdt, [(0, 8), (0, 8), (0, 10)], [4, 4, 6])
    assert p.shape == (4, 4, 6)
    assert np.allclose(p[:, :, :4], -5) and np.allclose(p[:, :, 5:], common.SDT_FILL)


def test_sdt_brats_end_to_end(tmp_path):
    shape = (32, 32, 24)
    src = tmp_path / "brats"
    syn.write_fake_brats_tar(src, n_cases=4, shape=shape, n_trainval=3)
    import csv
    splits = tmp_path / "splits.csv"
    with open(splits, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject", "id", "source", "site", "split", "label", "has_seg"])
        w.writeheader()
        for i in range(4):
            s = f"BraTS2021_{i:05d}"
            w.writerow(dict(subject=s, id=s if i < 3 else "", source="brats", site="unknown",
                            split=["train", "train", "val", "test"][i], label=1, has_seg=1))
    tar = str(src / "BraTS2021_Training_Data.tar")
    bout = tmp_path / "b2mm"
    prep_b.main(["--tar", tar, "--splits", str(splits), "--stats", str(src / "modality_stats.csv"),
                 "--out", str(bout), "--expected-shape", *map(str, shape), "--workers", "1"])
    out = tmp_path / "sdt"
    for k in range(2):
        s = sdt_brats.main(["--tar", tar, "--splits", str(splits), "--crop", str(bout / "crop.json"),
                            "--skullstrip", "threshold", "--out", str(out), "--workers", "1",
                            "--shard", str(k), "--nshards", "2"])
        assert s["n_failed_this_run"] == 0
    for i in range(4):
        subj = f"BraTS2021_{i:05d}"
        x = np.load(bout / "x" / f"{subj}.npy")
        d = np.load(out / "sdt" / f"{subj}.npy")
        assert d.shape == x.shape[1:]
        rec = json.loads((out / "rows" / f"{subj}.json").read_text())
        assert rec["ss_ml_b-4"] <= rec["ss_ml_b+1"] and 0 <= rec["frac_inside_b-2"] <= 1
