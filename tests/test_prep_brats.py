import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common as C  # noqa: E402
import prep_brats_2mm as prep  # noqa: E402
import synthetic as syn  # noqa: E402


def test_label_mapping_toy_volume():
    seg = np.zeros((4, 4, 4), np.uint8)
    seg[0, 0, 0] = 1  # NCR -> TC, WT
    seg[1, 1, 1] = 2  # ED  -> WT only
    seg[2, 2, 2] = 4  # ET  -> ET, TC, WT
    ch = C.bits_to_channels(C.seg_to_bits(seg))  # [ET, TC, WT]
    assert ch[:, 0, 0, 0].tolist() == [0, 1, 1]
    assert ch[:, 1, 1, 1].tolist() == [0, 0, 1]
    assert ch[:, 2, 2, 2].tolist() == [1, 1, 1]
    assert ch[:, 3, 3, 3].tolist() == [0, 0, 0]
    assert (C.channels_to_bits(ch) == C.seg_to_bits(seg)).all()


def test_pool_image_masked_mean_and_rezero():
    x = np.zeros((1, 2, 2, 4), np.float32)
    brain = np.zeros((2, 2, 4), bool)
    x[0, :, :, :2] = 3.0
    brain[:, :, :2] = True          # block 0: full brain
    x[0, 0, 0, 2] = 5.0
    brain[0, 0, 2] = True           # block 1: brain fraction 1/8 -> re-zeroed
    out, b2 = C.pool_image(x, brain)
    assert out.shape == (1, 1, 1, 2)
    assert out[0, 0, 0, 0] == 3.0 and out[0, 0, 0, 1] == 0.0
    assert b2.tolist() == [[[True, False]]]


def test_pool_labels_threshold_and_nesting():
    seg = np.zeros((2, 2, 2), np.uint8)
    seg.flat[:4] = 4                 # half the block ET -> ET kept (>= 0.5)
    bits = C.pool_labels(C.seg_to_bits(seg))
    assert C.bits_to_channels(bits)[:, 0, 0, 0].tolist() == [1, 1, 1]
    seg.flat[:4] = 0
    seg.flat[:3] = 4                 # 3/8 ET -> dropped
    assert C.pool_labels(C.seg_to_bits(seg))[0, 0, 0] == 0


def test_zscore_matches_owner_formula():
    a = np.array([0, 1, 2, 3, 0], np.float32)
    z, m, s = C.zscore_nonzero(a)
    assert m == 2.0 and abs(s - np.std([1, 2, 3])) < 1e-12
    assert z[0] == 0 and z[4] == 0 and abs(z[2]) < 1e-7


def test_prep_end_to_end_synthetic(tmp_path):
    src = tmp_path / "src"
    syn.write_fake_brats_tar(src, n_cases=6, shape=(32, 32, 24), n_trainval=4)
    ids = [f"BraTS2021_{i:05d}" for i in range(6)]
    splits = tmp_path / "splits.csv"
    with open(splits, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject", "id", "source", "site", "split", "label", "has_seg"])
        w.writeheader()
        for i, s in enumerate(ids):
            w.writerow(dict(subject=s, id=s if i < 4 else "", source="brats", site="unknown",
                            split=["train", "train", "train", "val", "test", "test"][i], label=1, has_seg=1))
    out = tmp_path / "out"
    summ = prep.main(["--tar", str(src / "BraTS2021_Training_Data.tar"), "--splits", str(splits),
                      "--stats", str(src / "modality_stats.csv"), "--out", str(out),
                      "--expected-shape", "32", "32", "24", "--workers", "2"])
    assert summ["complete"] and summ["equal_shapes"] and summ["n_done"] == 6
    assert summ["train_val"]["max_stat_reldiff"] < 1e-5
    assert summ["train_val"]["tumour_outside_support"] == 0
    crop = json.loads((out / "crop.json").read_text())
    assert all(t % 16 == 0 for t in crop["target_shape"])
    x = np.load(out / "x" / "BraTS2021_00000.npy")
    y = np.load(out / "y" / "BraTS2021_00000.npy")
    assert x.dtype == np.float16 and x.shape[0] == 4 and y.dtype == np.uint8 and x.shape[1:] == y.shape
    assert (out / "qc" / "montage.png").exists()
    # resume: second run does nothing new and stays complete
    summ2 = prep.main(["--tar", str(src / "BraTS2021_Training_Data.tar"), "--splits", str(splits),
                       "--stats", str(src / "modality_stats.csv"), "--out", str(out),
                       "--expected-shape", "32", "32", "24", "--workers", "2"])
    assert summ2["complete"] and summ2["n_done"] == 6


def test_even_box_when_brain_fills_axis():
    # regression: z box (0, 155) on a 155-slice volume used to become (-1, 155)
    box = C.pad_box_to_even([(38, 216), (10, 230), (0, 155)], (240, 240, 155))
    assert box == [(38, 216), (10, 230), (0, 156)]
    a = np.ones((2, 4, 4, 5), np.float32)
    c = C.crop(a, C.pad_box_to_even([(0, 4), (0, 4), (0, 5)], (4, 4, 5)))
    assert c.shape == (2, 4, 4, 6) and c[..., 5].sum() == 0 and c[..., :5].min() == 1
