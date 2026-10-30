import sys
import tarfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import synthetic as syn  # noqa: E402


def test_label_bits_mapping():
    seg = np.array([0, 1, 2, 4], np.uint8)
    ch = syn.bits_to_channels(syn.seg_to_bits(seg))  # [ET, TC, WT]
    assert ch[0].tolist() == [0, 0, 0, 1]
    assert ch[1].tolist() == [0, 1, 0, 1]
    assert ch[2].tolist() == [0, 1, 1, 1]


def test_cohort_shapes_and_ixi_zero_channels():
    cohort = syn.make_cohort(2, 2)
    for c in cohort:
        assert c["x"].shape == (4, 32, 32, 32) and c["x"].dtype == np.float16
        assert c["bits"].shape == (32, 32, 32)
    ixi = [c for c in cohort if c["source"] == "ixi"]
    assert all(not c["x"][[1, 3]].any() and not c["bits"].any() for c in ixi)
    brats = [c for c in cohort if c["source"] == "brats"]
    assert all(c["bits"].any() for c in brats)


def test_fake_tar(tmp_path):
    p = syn.write_fake_brats_tar(tmp_path, n_cases=3, n_trainval=2)
    with tarfile.open(p) as tf:
        names = tf.getnames()
    assert len(names) == 15
    assert "BraTS2021_00000/BraTS2021_00000_seg.nii.gz" in names
    assert (tmp_path / "modality_stats.csv").exists()
