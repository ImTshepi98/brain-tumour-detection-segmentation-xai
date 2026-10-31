import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model import AttentionGate, DualUNet, n_params  # noqa: E402


@pytest.mark.parametrize("attention", [True, False])
def test_shapes_full_size(attention):
    torch.manual_seed(0)
    m = DualUNet(attention=attention).eval()
    x = torch.randn(1, 4, 96, 112, 80)  # the real 2 mm array shape
    with torch.no_grad():
        assert m.forward_seg(x).shape == (1, 3, 96, 112, 80)
        assert m.forward_cls(x).shape == (1,)
        assert m(x, task="cls").shape == (1,)


def test_param_counts():
    a, b = n_params(DualUNet(attention=True)), n_params(DualUNet(attention=False))
    assert 4e6 < b < a < 8e6, (a, b)


def test_detection_ignores_t1ce_flair():
    torch.manual_seed(0)
    m = DualUNet().eval()
    x = torch.randn(2, 4, 32, 32, 32)
    y = x.clone()
    y[:, [1, 3]] = torch.randn(2, 2, 32, 32, 32)
    with torch.no_grad():
        assert torch.allclose(m.forward_cls(x), m.forward_cls(y))
        assert not torch.allclose(m.forward_seg(x), m.forward_seg(y))


def test_attention_gate_coefficients():
    g = AttentionGate(8, 16, 4)
    out = g(torch.randn(2, 8, 16, 16, 16), torch.randn(2, 16, 8, 8, 8))
    assert out.shape == (2, 8, 16, 16, 16)
    a = g.last_alpha
    assert a.shape == (2, 1, 16, 16, 16) and (a >= 0).all() and (a <= 1).all()


def test_gradients_reach_both_heads():
    m = DualUNet(channels=(4, 8, 16, 32, 64))
    x = torch.randn(2, 4, 32, 32, 32)
    (m.forward_seg(x).mean() + m.forward_cls(x).mean()).backward()
    assert m.seg_head.weight.grad.abs().sum() > 0 and m.cls_head[-1].weight.grad.abs().sum() > 0
    assert m.enc[0].conv1.weight.grad[:, [1, 3]].abs().sum() > 0  # seg pass trains T1ce/FLAIR filters
