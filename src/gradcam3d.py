"""Phase 7: 3D Grad-CAM++ with forward hooks (our own implementation).

For activations A (B, K, d, h, w) of the hooked layer and gradients g = dS/dA of a scalar target S:
  alpha = g^2 / (2 g^2 + sum_spatial(A) * g^3)      (0 where the denominator is 0)
  w_k   = sum_spatial(alpha * relu(g))
  cam   = relu(sum_k w_k A_k), upsampled trilinearly to the input size, min-max normalised to [0, 1].
Targets:
  detection    - the classification logit; layer = last encoder stage (model.enc[-1]); input as the
                 model sees it (forward_cls zeroes T1ce/FLAIR itself).
  segmentation - sum of the region's logit inside its predicted mask; layer = last decoder stage
                 before the head (model.dec[0]); 4-channel input.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics  # noqa: E402


class GradCAMpp:
    def __init__(self, layer):
        self.act = None
        self.handle = layer.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self.act = out

    def remove(self):
        self.handle.remove()

    def compute(self, score, size):
        """score: scalar tensor computed from the last forward pass. Returns cam (B, *size) in [0, 1]."""
        a = self.act
        g = torch.autograd.grad(score, a, retain_graph=False)[0]
        a, g = a.detach(), g.detach()
        g2, g3 = g.pow(2), g.pow(3)
        den = 2 * g2 + a.sum((2, 3, 4), keepdim=True) * g3
        alpha = torch.where(den != 0, g2 / torch.where(den != 0, den, torch.ones_like(den)), torch.zeros_like(den))
        w = (alpha * F.relu(g)).sum((2, 3, 4), keepdim=True)
        cam = F.relu((w * a).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=tuple(size), mode="trilinear", align_corners=False)[:, 0]
        flat = cam.flatten(1)
        lo, hi = flat.min(1)[0].view(-1, 1, 1, 1), flat.max(1)[0].view(-1, 1, 1, 1)
        return torch.where(hi > lo, (cam - lo) / (hi - lo).clamp(min=1e-12), torch.zeros_like(cam))


def detection_cam(model, x):
    """(cam (B, D, H, W), detection probability (B,))."""
    cam = GradCAMpp(model.enc[-1])
    try:
        with torch.enable_grad():
            logit = model.forward_cls(x)
            maps = cam.compute(logit.sum(), x.shape[2:])
    finally:
        cam.remove()
    return maps, torch.sigmoid(logit.detach())


def segmentation_cam(model, x, region="WT"):
    """(cam (B, D, H, W), predicted mask of that region (B, D, H, W)). Empty prediction -> zero map."""
    r = metrics.REGIONS.index(region)
    cam = GradCAMpp(model.dec[0])
    try:
        with torch.enable_grad():
            logits = model.forward_seg(x)
            pred = metrics.postprocess(logits.detach())[:, r]
            score = (logits[:, r] * pred.float()).sum()
            maps = (cam.compute(score, x.shape[2:]) if pred.any()
                    else torch.zeros(x.shape[:1] + x.shape[2:], device=x.device))
    finally:
        cam.remove()
    return maps, pred
