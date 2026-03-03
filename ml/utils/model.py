"""
CFU Detector Model — CenterNet Architecture
============================================
Architecture:
  Backbone : MobileNetV3-Small (ImageNet pretrained, lightweight for CPU)
  Neck     : Feature Pyramid Network (FPN) with 3 scales
  Head     : CenterNet-style dual output:
               - Heatmap  : [B, 1, H/4, W/4]  colony center probability
               - Size map : [B, 1, H/4, W/4]  colony radius (normalized)

Input  : [B, 3, 512, 512] normalized RGB image
Output : (heatmap, size_map) tensors
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConvBnRelu(nn.Sequential):
    """Conv2d → BatchNorm → ReLU6"""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1, groups: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                      padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )


class DepthwiseSeparable(nn.Module):
    """Depthwise-separable conv block for lightweight processing."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = ConvBnRelu(in_ch, in_ch, kernel=3, stride=stride, groups=in_ch)
        self.pw = ConvBnRelu(in_ch, out_ch, kernel=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class FPNNeck(nn.Module):
    """
    Feature Pyramid Network neck that fuses multi-scale backbone features
    into a single feature map at stride-4 resolution.
    """
    def __init__(self, in_channels: list[int], out_channels: int = 128):
        super().__init__()
        self.laterals = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels
        ])
        self.outputs = nn.ModuleList([
            ConvBnRelu(out_channels, out_channels) for _ in in_channels
        ])

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        laterals = [l(f) for l, f in zip(self.laterals, features)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest"
            )
        out = self.outputs[0](laterals[0])
        return out


class DetectionHead(nn.Module):
    """
    CenterNet-style detection head.
    Produces:
      heatmap : [B, 1, H, W]   — sigmoid-activated center probability
      size_map: [B, 1, H, W]   — ReLU-activated radius (in heatmap pixels)
    """
    def __init__(self, in_ch: int):
        super().__init__()
        self.shared = nn.Sequential(
            ConvBnRelu(in_ch, in_ch),
            ConvBnRelu(in_ch, in_ch // 2),
        )
        self.hm = nn.Sequential(
            ConvBnRelu(in_ch // 2, 64),
            nn.Conv2d(64, 1, 1),
        )
        self.sz = nn.Sequential(
            ConvBnRelu(in_ch // 2, 64),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, x: torch.Tensor):
        shared = self.shared(x)
        hm = torch.sigmoid(self.hm(shared))
        sz = F.relu(self.sz(shared))
        return hm, sz


class CFUDetectorModel(nn.Module):
    """
    Lightweight colony detector. Runs comfortably on CPU for inference.

    Backbone: MobileNetV3-Small — ~2.5M params
    Total   : ~4M params
    FPS     : ~5-15 fps on CPU (512×512 input)
    """
    INPUT_SIZE = 512
    STRIDE = 4

    def __init__(self, pretrained: bool = True):
        super().__init__()

        mobilenet = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.layer1 = mobilenet.features[:4]    # stride 8,  ch=24
        self.layer2 = mobilenet.features[4:9]   # stride 16, ch=48
        self.layer3 = mobilenet.features[9:13]  # stride 16, ch=96

        in_chs = [24, 48, 96]

        self.upsample = nn.Sequential(
            ConvBnRelu(24, 128),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBnRelu(128, 128),
        )

        self.neck = FPNNeck(in_chs, out_channels=128)
        self.head = DetectionHead(in_ch=128)

    def forward(self, x: torch.Tensor):
        c3 = self.layer1(x)
        c4 = self.layer2(c3)
        c5 = self.layer3(c4)

        c3_up = self.upsample(c3)

        c4_up = F.interpolate(c4, size=c3_up.shape[-2:], mode="bilinear", align_corners=False)
        c4_proj = nn.functional.conv2d(
            c4_up,
            weight=self.neck.laterals[1].weight,
            bias=self.neck.laterals[1].bias,
        )
        c3_proj = nn.functional.conv2d(
            c3_up,
            weight=self.neck.laterals[0].weight,
            bias=self.neck.laterals[0].bias,
        )
        fused = self.neck.outputs[0](c3_proj + c4_proj)

        heatmap, size_map = self.head(fused)
        return heatmap, size_map

    def save(self, path: str, extra_info: dict | None = None) -> None:
        payload = {
            "model_state": self.state_dict(),
            "architecture": "CFUDetectorModel-v1",
            "input_size": self.INPUT_SIZE,
            "stride": self.STRIDE,
            "info": extra_info or {},
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "CFUDetectorModel":
        checkpoint = torch.load(path, map_location=device)
        model = cls(pretrained=False)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        return model


class FocalLoss(nn.Module):
    """Modified Focal Loss for heatmap regression (CenterNet / CornerNet style)."""
    def __init__(self, alpha: float = 2.0, beta: float = 4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pos_mask = target.eq(1).float()
        neg_mask = 1.0 - pos_mask

        pos_loss = (
            torch.log(pred + 1e-7)
            * ((1 - pred) ** self.alpha)
            * pos_mask
        )
        neg_loss = (
            torch.log(1 - pred + 1e-7)
            * (pred ** self.alpha)
            * ((1 - target) ** self.beta)
            * neg_mask
        )

        num_pos = pos_mask.sum().clamp(min=1)
        loss = -(pos_loss.sum() + neg_loss.sum()) / num_pos
        return loss


class SizeRegLoss(nn.Module):
    """Smooth L1 loss for radius regression, only at positive positions."""
    def forward(
        self,
        pred: torch.Tensor,
        target_size: torch.Tensor,
        target_hm: torch.Tensor,
    ) -> torch.Tensor:
        pos_mask = target_hm.eq(1).float()
        num_pos = pos_mask.sum().clamp(min=1)
        loss = F.smooth_l1_loss(pred * pos_mask, target_size * pos_mask, reduction="sum")
        return loss / num_pos


class CFUDetectorLoss(nn.Module):
    """Combined heatmap focal loss + size regression loss."""
    def __init__(self, size_weight: float = 0.1):
        super().__init__()
        self.focal = FocalLoss()
        self.size_reg = SizeRegLoss()
        self.size_weight = size_weight

    def forward(
        self,
        hm_pred: torch.Tensor,
        sz_pred: torch.Tensor,
        hm_target: torch.Tensor,
        sz_target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        focal = self.focal(hm_pred, hm_target)
        size  = self.size_reg(sz_pred, sz_target, hm_target)
        total = focal + self.size_weight * size
        return {"total": total, "focal": focal, "size": size}


def gaussian_radius(det_size: tuple[float, float], min_overlap: float = 0.7) -> float:
    """Compute the Gaussian radius for a colony given its size on the heatmap."""
    h, w = det_size
    a1 = 1
    b1 = h + w
    c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (h + w)
    c2 = (1 - min_overlap) * w * h
    sq2 = math.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (h + w)
    c3 = (min_overlap - 1) * w * h
    sq3 = math.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)


def draw_gaussian(heatmap: torch.Tensor, cx: int, cy: int, radius: float) -> None:
    """Draw a 2D Gaussian at (cx, cy) on the heatmap tensor in-place."""
    diameter = int(2 * radius + 1)
    sigma = radius / 3.0

    x = torch.arange(0, diameter, dtype=torch.float32)
    y = torch.arange(0, diameter, dtype=torch.float32)
    gy, gx = torch.meshgrid(y, x, indexing="ij")
    gaussian = torch.exp(
        -((gx - radius) ** 2 + (gy - radius) ** 2) / (2 * sigma ** 2)
    )

    H, W = heatmap.shape[-2:]
    x0 = int(cx - radius)
    y0 = int(cy - radius)
    x1 = int(cx + radius + 1)
    y1 = int(cy + radius + 1)

    gx0 = max(0, -x0)
    gy0 = max(0, -y0)
    gx1 = gx0 + min(x1, W) - max(x0, 0)
    gy1 = gy0 + min(y1, H) - max(y0, 0)

    hx0 = max(x0, 0)
    hy0 = max(y0, 0)
    hx1 = hx0 + gx1 - gx0
    hy1 = hy0 + gy1 - gy0

    if hx1 <= hx0 or hy1 <= hy0 or gx1 <= gx0 or gy1 <= gy0:
        return

    patch = gaussian[gy0:gy1, gx0:gx1]
    existing = heatmap[..., hy0:hy1, hx0:hx1]
    torch.maximum(existing, patch, out=existing)


if __name__ == "__main__":
    model = CFUDetectorModel(pretrained=False)
    dummy = torch.randn(2, 3, 512, 512)
    hm, sz = model(dummy)
    print(f"Heatmap shape : {hm.shape}")   # [2, 1, 128, 128]
    print(f"Size map shape: {sz.shape}")   # [2, 1, 128, 128]
    print("Model params  :", sum(p.numel() for p in model.parameters()) / 1e6, "M")