import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import datasets as ds  # noqa: E402
import synthetic as syn  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _fake(tmp_path):
    p = syn.write_fake_2mm(tmp_path, n_brats=8, n_ixi=8)
    return p, ds.read_splits(p["splits"])


def _load(p, rows, splits=("train",), **kw):
    return ds.Cohort.load(rows, splits, p["brats_dir"], [p["ixi_dir"]], [p["brats_sdt_dir"]], workers=1, **kw)


def test_real_splits_pass_checks():
    counts = ds.check_splits(ds.read_splits(ROOT / "data_splits/splits.csv"))
    assert counts == ds.EXPECTED


def test_split_checks_catch_errors(tmp_path):
    _, rows = _fake(tmp_path)
    with pytest.raises(AssertionError):
        ds.check_splits(rows + [dict(rows[0])], expected=None)
    bad = [dict(r) for r in rows]
    bad[0]["split"] = "holdout"
    with pytest.raises(AssertionError):
        ds.check_splits(bad, expected=None)


def test_test_split_is_refused(tmp_path):
    p, rows = _fake(tmp_path)
    with pytest.raises(AssertionError, match="locked"):
        _load(p, rows, ("train", "test"))
    assert len(_load(p, rows, ("test",), allow_test=True)) == 4


def test_batch_shapes_and_shared_mask(tmp_path):
    p, rows = _fake(tmp_path)
    raw = _load(p, rows)
    masked = _load(p, rows, border=-2.0)
    assert len(raw) == 8 and raw.shape == (4, 32, 32, 32)
    x, target, label, has_seg = masked.batch(np.arange(len(masked)), "cpu")
    assert x.shape == (8, 4, 32, 32, 32) and x.dtype == torch.float32
    assert target.shape == (8, 3, 32, 32, 32)
    assert label.tolist() == [float(s == "brats") for s in masked.source]
    assert has_seg.tolist() == label.bool().tolist()
    # border -2 mm shrinks every brain but keeps all BraTS tumour voxels
    for i in range(len(raw)):
        b_raw, b_new = (raw.x[i] != 0).any(0), (masked.x[i] != 0).any(0)
        assert b_new.sum() < b_raw.sum() and not (b_new & ~b_raw).any()
        assert not ((raw.y[i] & 4) > 0)[~b_new].any()
    # IXI target channels are empty, IXI T1ce/FLAIR stay zero
    ixi = [i for i, s in enumerate(masked.source) if s == "ixi"]
    assert target[ixi].sum() == 0 and x[ixi][:, [1, 3]].abs().sum() == 0


def test_balanced_sampler():
    labels = np.array([1] * 30 + [0] * 10)
    idx = ds.sample_epoch(labels, 2000, "balanced", epoch=3)
    assert 0.45 < labels[idx].mean() < 0.55 and set(idx[labels[idx] == 0]) == set(range(30, 40))
    assert (ds.sample_epoch(labels, 2000, "balanced", epoch=3) == idx).all()  # deterministic per epoch
    assert not (ds.sample_epoch(labels, 2000, "balanced", epoch=4) == idx).all()
    assert set(ds.sample_epoch(labels, 0, "brats_only")) == set(range(30))


def test_augment_keeps_background_and_aligns_labels():
    x, bits, _ = syn.make_case((16, 16, 16), tumour=True, rng=np.random.default_rng(1))
    xi, _, _ = syn.make_case((16, 16, 16), tumour=False, rng=np.random.default_rng(2))
    x = torch.from_numpy(np.stack([x, xi])).float()
    t = torch.from_numpy(np.stack([syn.bits_to_channels(bits), np.zeros((3, 16, 16, 16), np.float32)]))
    xa, ta = ds.augment(x, t, torch.Generator().manual_seed(0))
    brain_a = (xa != 0).any(1)
    assert brain_a.sum() == (x != 0).any(1).sum()             # background stays 0, brain size unchanged
    assert not (ta[:, 2].bool() & ~brain_a).any()             # tumour stays inside the (flipped) brain
    assert xa[1, [1, 3]].abs().sum() == 0                     # absent IXI channels stay zero
    assert ta[0].sum() == t[0].sum() and ta[1].sum() == 0     # flips move labels, never change them


def test_detection_input_zeroes_t1ce_flair():
    d = ds.detection_input(torch.ones(2, 4, 4, 4, 4))
    assert d[:, [0, 2]].eq(1).all() and d[:, [1, 3]].eq(0).all()
