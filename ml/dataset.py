from __future__ import annotations
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
