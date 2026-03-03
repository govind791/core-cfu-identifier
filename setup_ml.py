"""
Run this script from your project root to create all ml/ files automatically.
Usage: python setup_ml.py
"""
import os

os.makedirs("ml", exist_ok=True)

# ── model.py ──────────────────────────────────────────────────────────────────
with open("ml/model.py", "w") as f:
    f.write('''from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1, groups=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )


class FPNNeck(nn.Module):
    def __init__(self, in_channels, out_channels=128):
        super().__init__()
        self.laterals = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels])
        self.outputs = nn.ModuleList([ConvBnRelu(out_channels, out_channels) for _ in in_channels])

    def forward(self, features):
        laterals = [l(f) for l, f in zip(self.laterals, features)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest"
            )
        return self.outputs[0](laterals[0])


class DetectionHead(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.shared = nn.Sequential(ConvBnRelu(in_ch, in_ch), ConvBnRelu(in_ch, in_ch // 2))
        self.hm = nn.Sequential(ConvBnRelu(in_ch // 2, 64), nn.Conv2d(64, 1, 1))
        self.sz = nn.Sequential(ConvBnRelu(in_ch // 2, 64), nn.Conv2d(64, 1, 1))

    def forward(self, x):
        shared = self.shared(x)
        return torch.sigmoid(self.hm(shared)), F.relu(self.sz(shared))


class CFUDetectorModel(nn.Module):
    INPUT_SIZE = 512
    STRIDE = 4

    def __init__(self, pretrained=True):
        super().__init__()
        mobilenet = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.layer1 = mobilenet.features[:4]
        self.layer2 = mobilenet.features[4:9]
        self.layer3 = mobilenet.features[9:13]
        self.upsample = nn.Sequential(
            ConvBnRelu(24, 128),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBnRelu(128, 128),
        )
        self.neck = FPNNeck([24, 48, 96], out_channels=128)
        self.head = DetectionHead(in_ch=128)

    def forward(self, x):
        c3 = self.layer1(x)
        c4 = self.layer2(c3)
        c3_up = self.upsample(c3)
        c4_up = F.interpolate(c4, size=c3_up.shape[-2:], mode="bilinear", align_corners=False)
        c3_proj = F.conv2d(c3_up, weight=self.neck.laterals[0].weight, bias=self.neck.laterals[0].bias)
        c4_proj = F.conv2d(c4_up, weight=self.neck.laterals[1].weight, bias=self.neck.laterals[1].bias)
        fused = self.neck.outputs[0](c3_proj + c4_proj)
        return self.head(fused)

    def save(self, path, extra_info=None):
        torch.save({
            "model_state": self.state_dict(),
            "architecture": "CFUDetectorModel-v1",
            "input_size": self.INPUT_SIZE,
            "stride": self.STRIDE,
            "info": extra_info or {},
        }, path)

    @classmethod
    def load(cls, path, device="cpu"):
        checkpoint = torch.load(path, map_location=device)
        model = cls(pretrained=False)
        model.load_state_dict(checkpoint["model_state"])
        return model.to(device)


class FocalLoss(nn.Module):
    def __init__(self, alpha=2.0, beta=4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        pos_mask = target.eq(1).float()
        neg_mask = 1.0 - pos_mask
        pos_loss = torch.log(pred + 1e-7) * ((1 - pred) ** self.alpha) * pos_mask
        neg_loss = torch.log(1 - pred + 1e-7) * (pred ** self.alpha) * ((1 - target) ** self.beta) * neg_mask
        return -(pos_loss.sum() + neg_loss.sum()) / pos_mask.sum().clamp(min=1)


class SizeRegLoss(nn.Module):
    def forward(self, pred, target_size, target_hm):
        pos_mask = target_hm.eq(1).float()
        return F.smooth_l1_loss(pred * pos_mask, target_size * pos_mask, reduction="sum") / pos_mask.sum().clamp(min=1)


class CFUDetectorLoss(nn.Module):
    def __init__(self, size_weight=0.1):
        super().__init__()
        self.focal = FocalLoss()
        self.size_reg = SizeRegLoss()
        self.size_weight = size_weight

    def forward(self, hm_pred, sz_pred, hm_target, sz_target):
        focal = self.focal(hm_pred, hm_target)
        size = self.size_reg(sz_pred, sz_target, hm_target)
        return {"total": focal + self.size_weight * size, "focal": focal, "size": size}


def gaussian_radius(det_size, min_overlap=0.7):
    h, w = det_size
    b1 = h + w
    c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(b1 ** 2 - 4 * c1)) / 2
    b2 = 2 * (h + w)
    c2 = (1 - min_overlap) * w * h
    r2 = (b2 + math.sqrt(b2 ** 2 - 16 * c2)) / 2
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (h + w)
    c3 = (min_overlap - 1) * w * h
    r3 = (b3 + math.sqrt(b3 ** 2 - 4 * a3 * c3)) / (2 * a3)
    return min(r1, r2, r3)


def draw_gaussian(heatmap, cx, cy, radius):
    diameter = int(2 * radius + 1)
    sigma = radius / 3.0
    x = torch.arange(0, diameter, dtype=torch.float32)
    y = torch.arange(0, diameter, dtype=torch.float32)
    gy, gx = torch.meshgrid(y, x, indexing="ij")
    gaussian = torch.exp(-((gx - radius) ** 2 + (gy - radius) ** 2) / (2 * sigma ** 2))
    H, W = heatmap.shape[-2:]
    x0, y0 = int(cx - radius), int(cy - radius)
    x1, y1 = int(cx + radius + 1), int(cy + radius + 1)
    gx0, gy0 = max(0, -x0), max(0, -y0)
    gx1 = gx0 + min(x1, W) - max(x0, 0)
    gy1 = gy0 + min(y1, H) - max(y0, 0)
    hx0, hy0 = max(x0, 0), max(y0, 0)
    hx1 = hx0 + gx1 - gx0
    hy1 = hy0 + gy1 - gy0
    if hx1 <= hx0 or hy1 <= hy0 or gx1 <= gx0 or gy1 <= gy0:
        return
    torch.maximum(heatmap[..., hy0:hy1, hx0:hx1], gaussian[gy0:gy1, gx0:gx1],
                  out=heatmap[..., hy0:hy1, hx0:hx1])
''')
print("✓ ml/model.py written")


# ── evaluate.py ───────────────────────────────────────────────────────────────
with open("ml/evaluate.py", "w") as f:
    f.write('''from __future__ import annotations
import math
import numpy as np
import torch
import cv2
from tqdm import tqdm


def circle_iou(cx1, cy1, r1, cx2, cy2, r2):
    d = math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return (math.pi * min(r1, r2) ** 2) / (math.pi * max(r1, r2) ** 2)
    a = (r1 ** 2) * math.acos((d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1))
    b = (r2 ** 2) * math.acos((d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2))
    c = 0.5 * math.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
    intersection = a + b - c
    union = math.pi * (r1 ** 2 + r2 ** 2) - intersection
    return intersection / max(union, 1e-7)


def decode_heatmap_predictions(heatmap, size_map, stride=4, score_thresh=0.3, nms_radius=3):
    hm = heatmap[0]
    sz = size_map[0]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (nms_radius * 2 + 1,) * 2)
    local_max = cv2.dilate(hm, kernel)
    peaks = (hm == local_max) & (hm > score_thresh)
    ys, xs = np.where(peaks)
    detections = []
    for y, x in zip(ys, xs):
        score = float(hm[y, x])
        radius = max(2.0, float(sz[y, x]) * stride)
        detections.append(((x + 0.5) * stride, (y + 0.5) * stride, radius, score))
    return detections


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = total_focal = total_size = 0.0
    n_batches = 0
    count_errors = []

    for batch in tqdm(loader, desc="  val  ", leave=False):
        img = batch["image"].to(device)
        hm_tgt = batch["heatmap"].to(device)
        sz_tgt = batch["size_map"].to(device)
        gt_counts = batch["n_colonies"].numpy()

        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            hm_pred, sz_pred = model(img)
            losses = criterion(hm_pred, sz_pred, hm_tgt, sz_tgt)

        total_loss += losses["total"].item()
        total_focal += losses["focal"].item()
        total_size += losses["size"].item()
        n_batches += 1

        hm_np = hm_pred.cpu().numpy()
        sz_np = sz_pred.cpu().numpy()
        for i in range(len(img)):
            preds = decode_heatmap_predictions(hm_np[i], sz_np[i])
            count_errors.append(abs(len(preds) - int(gt_counts[i])))

    n = max(1, n_batches)
    return {
        "loss": total_loss / n,
        "focal": total_focal / n,
        "size": total_size / n,
        "count_mae": float(np.mean(count_errors)) if count_errors else 0.0,
    }
''')
print("✓ ml/evaluate.py written")


# ── dataset.py ────────────────────────────────────────────────────────────────
with open("ml/dataset.py", "w") as f:
    f.write('''from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ml.model import gaussian_radius, draw_gaussian

INPUT_SIZE = 512
HEATMAP_SIZE = INPUT_SIZE // 4


class CFUDataset(Dataset):
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, data_dir, split="train", transform=None, max_colonies=500):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.max_colonies = max_colonies

        index_file = self.data_dir / "dataset.json"
        if index_file.exists():
            index = json.loads(index_file.read_text())
            if split == "train":
                self.ids = index["train"]
            elif split == "val":
                self.ids = index["val"]
            else:
                self.ids = index["train"] + index["val"]
        else:
            self.ids = [p.stem for p in (self.data_dir / "images").glob("*.png")]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        bgr = cv2.imread(str(self.data_dir / "images" / f"{image_id}.png"))
        if bgr is None:
            raise FileNotFoundError(f"Image not found: {image_id}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        ann_path = self.data_dir / "labels" / f"{image_id}.json"
        ann = json.loads(ann_path.read_text()) if ann_path.exists() else {"colonies": [], "is_tntc": False}

        colonies = ann.get("colonies", [])
        is_tntc = ann.get("is_tntc", False)
        if is_tntc:
            colonies = []

        keypoints = [(c["cx"], c["cy"]) for c in colonies]
        radii = [c["radius"] for c in colonies]

        if self.transform:
            rgb, keypoints, radii = self._apply_augment(rgb, keypoints, radii)

        heatmap, size_map = self._encode_targets(keypoints, radii, rgb.shape[:2])

        img_f = rgb.astype(np.float32) / 255.0
        img_f = (img_f - self.MEAN) / self.STD
        img_t = torch.from_numpy(img_f.transpose(2, 0, 1))

        return {
            "image": img_t, "heatmap": heatmap, "size_map": size_map,
            "n_colonies": len(keypoints), "is_tntc": int(is_tntc), "image_id": image_id,
        }

    def _apply_augment(self, rgb, keypoints, radii):
        result = self.transform(image=rgb, keypoints=[(kp[0], kp[1]) for kp in keypoints])
        rgb_out = result["image"]
        kps_out = result["keypoints"]
        h, w = rgb_out.shape[:2]
        surviving, surviving_r = [], []
        for i, (x, y) in enumerate(kps_out):
            if 0 <= x < w and 0 <= y < h:
                surviving.append((float(x), float(y)))
                if i < len(radii):
                    surviving_r.append(radii[i])
        return rgb_out, surviving, surviving_r

    def _encode_targets(self, keypoints, radii, img_shape):
        H, W = img_shape
        stride_x = W / HEATMAP_SIZE
        stride_y = H / HEATMAP_SIZE
        heatmap = torch.zeros(1, HEATMAP_SIZE, HEATMAP_SIZE)
        size_map = torch.zeros(1, HEATMAP_SIZE, HEATMAP_SIZE)
        for i, ((cx, cy), r) in enumerate(zip(keypoints, radii)):
            if i >= self.max_colonies:
                break
            hm_cx = cx / stride_x
            hm_cy = cy / stride_y
            hm_r = r / max(stride_x, stride_y)
            gauss_r = max(1.0, gaussian_radius((hm_r * 2, hm_r * 2), min_overlap=0.7))
            draw_gaussian(heatmap, int(hm_cx), int(hm_cy), gauss_r)
            hx, hy = int(hm_cx), int(hm_cy)
            if 0 <= hx < HEATMAP_SIZE and 0 <= hy < HEATMAP_SIZE:
                size_map[0, hy, hx] = hm_r
        return heatmap, size_map


def build_dataloaders(data_dir, batch_size=8, num_workers=4, train_transform=None, val_transform=None):
    train_ds = CFUDataset(data_dir, split="train", transform=train_transform)
    val_ds = CFUDataset(data_dir, split="val", transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
''')
print("✓ ml/dataset.py written")


# ── generate_synthetic_data.py ────────────────────────────────────────────────
with open("ml/generate_synthetic_data.py", "w") as f:
    f.write('''from __future__ import annotations
import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

IMAGE_SIZE = 512
PLATE_COLORS = [(200,210,180),(210,200,175),(185,195,165),(220,215,190)]
COLONY_PALETTES = [
    {"mean":(255,255,255),"std":(15,15,15)},
    {"mean":(230,230,230),"std":(20,20,20)},
    {"mean":(200,215,200),"std":(15,20,15)},
    {"mean":(240,230,215),"std":(10,10,10)},
    {"mean":(255,230,200),"std":(10,20,15)},
    {"mean":(200,200,220),"std":(15,15,20)},
]


def _generate_plate(rng, size=IMAGE_SIZE):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = np.array([rng.integers(20,60), rng.integers(20,60), rng.integers(20,60)], dtype=np.uint8)
    cx = size//2 + int(rng.integers(-20,20))
    cy = size//2 + int(rng.integers(-20,20))
    radius = int(size * rng.uniform(0.35, 0.46))
    plate_color = list(rng.choice(PLATE_COLORS))
    noise = rng.normal(0, 8, (size, size, 3)).astype(np.int16)
    plate_img = np.clip(np.array(plate_color, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    for c in range(3):
        img[:,:,c] = np.where(mask > 0, plate_img[:,:,c], img[:,:,c])
    shadow = np.zeros_like(mask)
    cv2.circle(shadow, (cx, cy), radius, 255, 3)
    img[shadow > 0] = np.clip(img[shadow > 0].astype(np.int16) - 30, 0, 255).astype(np.uint8)
    return img, (cx, cy, radius)


def _add_glare(rng, img, cx, cy, radius):
    if rng.random() > 0.3:
        return img
    overlay = img.copy()
    gcx = cx + int(rng.integers(-radius//2, radius//2))
    gcy = cy + int(rng.integers(-radius//2, 0))
    gr = int(rng.integers(20, 70))
    cv2.ellipse(overlay, (gcx, gcy), (gr, gr//2), angle=rng.integers(0,180),
                startAngle=0, endAngle=360, color=(255,255,255), thickness=-1)
    return cv2.addWeighted(overlay, 0.3, img, 0.7, 0)


def _sample_colonies(rng, cx, cy, radius, n):
    colonies = []
    min_r = max(3, int(radius*0.012))
    max_r = max(8, int(radius*0.075))
    for _ in range(n * 20):
        if len(colonies) >= n:
            break
        angle = rng.uniform(0, 2*math.pi)
        r_frac = rng.uniform(0, 0.88)
        px = int(cx + r_frac*radius*math.cos(angle))
        py = int(cy + r_frac*radius*math.sin(angle))
        pr = int(rng.integers(min_r, max_r+1))
        if not any(math.sqrt((px-ox)**2+(py-oy)**2) < (pr+orr)*0.7 for ox,oy,orr in colonies):
            colonies.append((px, py, pr))
    return colonies


def _draw_colony(rng, img, cx, cy, radius, palette):
    color = tuple(int(np.clip(rng.normal(palette["mean"][c], palette["std"][c]), 0, 255)) for c in range(3))
    cv2.circle(img, (cx, cy), radius, color, -1)
    cv2.circle(img, (cx, cy), max(1, radius-2), tuple(min(255,c+20) for c in color), -1)
    cv2.circle(img, (cx, cy), radius, tuple(max(0,c-40) for c in color), 1)


def generate_dataset(output_dir, n_images=5000, seed=42, val_fraction=0.1, tntc_fraction=0.05):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    output_path = Path(output_dir)
    (output_path / "images").mkdir(parents=True, exist_ok=True)
    (output_path / "labels").mkdir(parents=True, exist_ok=True)
    train_ids, val_ids, all_ann = [], [], []
    n_val = int(n_images * val_fraction)

    for i in tqdm(range(n_images), desc="Generating"):
        image_id = f"syn_{i:06d}"
        is_val = i < n_val
        is_tntc = rng.random() < tntc_fraction
        img, (plate_cx, plate_cy, plate_r) = _generate_plate(rng)
        n_colonies = int(rng.integers(301,600)) if is_tntc else int(10**rng.uniform(0,math.log10(301)))
        palette = rng.choice(COLONY_PALETTES)
        colonies = _sample_colonies(rng, plate_cx, plate_cy, plate_r, n_colonies)
        for cx,cy,r in colonies:
            _draw_colony(rng, img, cx, cy, r, palette)
        img = _add_glare(rng, img, plate_cx, plate_cy, plate_r)
        img = np.clip(img.astype(np.float32)*rng.uniform(0.8,1.2)+rng.integers(-15,15), 0, 255).astype(np.uint8)
        if rng.random() < 0.15:
            img = cv2.GaussianBlur(img, (rng.choice([3,5,7]),)*2, 0)
        cv2.imwrite(str(output_path/"images"/f"{image_id}.png"), img)
        ann = {
            "image_id": image_id, "image_size": IMAGE_SIZE,
            "plate": {"cx": plate_cx, "cy": plate_cy, "radius": plate_r},
            "colonies": [{"cx":cx,"cy":cy,"radius":r,"nx":cx/IMAGE_SIZE,"ny":cy/IMAGE_SIZE,"nr":r/IMAGE_SIZE}
                         for cx,cy,r in (colonies[:300] if is_tntc else colonies)],
            "colony_count": len(colonies), "is_tntc": is_tntc,
        }
        (output_path/"labels"/f"{image_id}.json").write_text(json.dumps(ann, indent=2))
        all_ann.append(ann)
        (val_ids if is_val else train_ids).append(image_id)

    (output_path/"dataset.json").write_text(json.dumps({
        "n_images": n_images, "n_train": len(train_ids), "n_val": len(val_ids),
        "seed": seed, "train": train_ids, "val": val_ids,
    }, indent=2))
    print(f"Done! {n_images} images in {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/synthetic")
    parser.add_argument("--n-images", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()
    generate_dataset(args.output_dir, args.n_images, args.seed, args.val_fraction)
''')
print("✓ ml/generate_synthetic_data.py written")


# ── train.py ──────────────────────────────────────────────────────────────────
with open("ml/train.py", "w") as f:
    f.write('''from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from ml.augmentations import get_train_transform, get_val_transform
from ml.dataset import build_dataloaders
from ml.evaluate import evaluate_epoch
from ml.model import CFUDetectorModel, CFUDetectorLoss


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = total_focal = total_size = 0.0
    n_batches = 0
    for batch in tqdm(loader, desc="  train", leave=False):
        img = batch["image"].to(device)
        hm_tgt = batch["heatmap"].to(device)
        sz_tgt = batch["size_map"].to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            hm_pred, sz_pred = model(img)
            losses = criterion(hm_pred, sz_pred, hm_tgt, sz_tgt)
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += losses["total"].item()
        total_focal += losses["focal"].item()
        total_size += losses["size"].item()
        n_batches += 1
    n = max(1, n_batches)
    return {"loss": total_loss/n, "focal": total_focal/n, "size": total_size/n}


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    train_loader, val_loader = build_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size,
        num_workers=0,
        train_transform=get_train_transform(), val_transform=get_val_transform(),
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = CFUDetectorModel(pretrained=not args.no_pretrain).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device)["model_state"])

    criterion = CFUDetectorLoss(size_weight=args.size_weight)
    backbone_params = list(model.layer1.parameters()) + list(model.layer2.parameters()) + list(model.layer3.parameters())
    head_params = list(model.upsample.parameters()) + list(model.neck.parameters()) + list(model.head.parameters())
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_metrics = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        log = {
            "epoch": epoch, "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"], "val_mae": val_metrics.get("count_mae", -1),
            "lr": optimizer.param_groups[1]["lr"], "elapsed_s": time.time()-t0,
        }
        history.append(log)
        print(f"Epoch {epoch:3d}/{args.epochs} | train={log[\'train_loss\']:.4f} | val={log[\'val_loss\']:.4f} | mae={log[\'val_mae\']:.2f}")
        model.save(str(output_dir/"latest.pt"), extra_info=log)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            model.save(str(output_dir/"best.pt"), extra_info=log)
            print(f"  New best saved (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping.")
                break

    (output_dir/"history.json").write_text(json.dumps(history, indent=2))
    print(f"Done. Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="runs/exp1")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--size-weight", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--no-pretrain", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    main(args)
''')
print("✓ ml/train.py written")


# ── __init__.py ───────────────────────────────────────────────────────────────
with open("ml/__init__.py", "w") as f:
    f.write("")
print("✓ ml/__init__.py written")

print("\n✅ All ml/ files created successfully!")
print("Next steps:")
print("  1. pip install torch torchvision opencv-python-headless numpy tqdm albumentations")
print("  2. python -m ml.generate_synthetic_data --output-dir data/synthetic --n-images 5000")
print("  3. python -m ml.train --data-dir data/synthetic --output-dir runs/exp1 --epochs 50")