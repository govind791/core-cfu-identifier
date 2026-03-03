"""
Synthetic Colony Data Generator
================================
Creates realistic-looking TSA plate images for training the CFU detector.

Usage:
  python -m ml.generate_synthetic_data \
      --output-dir data/synthetic \
      --n-images 5000 \
      --seed 42

Output layout:
  data/synthetic/
    images/       ← PNG plate images (512×512)
    labels/       ← JSON annotation per image
    dataset.json  ← index file listing all images + labels
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm


IMAGE_SIZE = 512
HEATMAP_STRIDE = 4

PLATE_COLORS = [
    (200, 210, 180),
    (210, 200, 175),
    (185, 195, 165),
    (220, 215, 190),
]

COLONY_PALETTES = [
    {"mean": (255, 255, 255), "std": (15, 15, 15)},
    {"mean": (230, 230, 230), "std": (20, 20, 20)},
    {"mean": (200, 215, 200), "std": (15, 20, 15)},
    {"mean": (240, 230, 215), "std": (10, 10, 10)},
    {"mean": (255, 230, 200), "std": (10, 20, 15)},
    {"mean": (200, 200, 220), "std": (15, 15, 20)},
]


def _generate_plate(rng: np.random.Generator, size: int = IMAGE_SIZE) -> tuple[np.ndarray, tuple]:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    bg_color = np.array([
        rng.integers(20, 60), rng.integers(20, 60), rng.integers(20, 60),
    ], dtype=np.uint8)
    img[:] = bg_color

    cx = size // 2 + int(rng.integers(-20, 20))
    cy = size // 2 + int(rng.integers(-20, 20))
    radius = int(size * rng.uniform(0.35, 0.46))

    plate_color = list(rng.choice(PLATE_COLORS))
    noise = rng.normal(0, 8, (size, size, 3)).astype(np.int16)
    plate_img = np.clip(np.array(plate_color, dtype=np.int16) + noise, 0, 255).astype(np.uint8)

    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)

    for c in range(3):
        img[:, :, c] = np.where(mask > 0, plate_img[:, :, c], img[:, :, c])

    shadow_ring = np.zeros_like(mask)
    cv2.circle(shadow_ring, (cx, cy), radius, 255, 3)
    img[shadow_ring > 0] = np.clip(
        img[shadow_ring > 0].astype(np.int16) - 30, 0, 255
    ).astype(np.uint8)

    return img, (cx, cy, radius)


def _add_glare(rng, img, cx, cy, radius):
    if rng.random() > 0.3:
        return img
    overlay = img.copy()
    glare_cx = cx + int(rng.integers(-radius // 2, radius // 2))
    glare_cy = cy + int(rng.integers(-radius // 2, 0))
    glare_r  = int(rng.integers(20, 70))
    cv2.ellipse(overlay, (glare_cx, glare_cy), (glare_r, glare_r // 2),
                angle=rng.integers(0, 180), startAngle=0, endAngle=360,
                color=(255, 255, 255), thickness=-1)
    return cv2.addWeighted(overlay, 0.3, img, 0.7, 0)


def _sample_colonies(rng, cx, cy, radius, n_colonies):
    colonies = []
    min_r = max(3, int(radius * 0.012))
    max_r = max(8, int(radius * 0.075))
    max_attempts = n_colonies * 20

    for _ in range(max_attempts):
        if len(colonies) >= n_colonies:
            break
        angle = rng.uniform(0, 2 * math.pi)
        r_frac = rng.uniform(0, 0.88)
        px = int(cx + r_frac * radius * math.cos(angle))
        py = int(cy + r_frac * radius * math.sin(angle))
        pr = int(rng.integers(min_r, max_r + 1))

        overlap = False
        for (ox, oy, orr) in colonies:
            dist = math.sqrt((px - ox) ** 2 + (py - oy) ** 2)
            if dist < (pr + orr) * 0.7:
                overlap = True
                break

        if not overlap:
            colonies.append((px, py, pr))

    return colonies


def _draw_colony(rng, img, cx, cy, radius, palette):
    color = tuple(int(np.clip(
        rng.normal(palette["mean"][c], palette["std"][c]), 0, 255
    )) for c in range(3))
    cv2.circle(img, (cx, cy), radius, color, -1)
    inner_r = max(1, radius - 2)
    brighter = tuple(min(255, c + 20) for c in color)
    cv2.circle(img, (cx, cy), inner_r, brighter, -1)
    cv2.circle(img, (cx, cy), radius, tuple(max(0, c - 40) for c in color), 1)
    if rng.random() > 0.6:
        shadow_x = cx + max(1, radius // 4)
        shadow_y = cy + max(1, radius // 4)
        cv2.circle(img, (shadow_x, shadow_y), radius + 1,
                   (max(0, color[0] - 50), max(0, color[1] - 50), max(0, color[2] - 50)), 1)


def _make_annotation(image_id, img_size, colonies, plate_cx, plate_cy, plate_r):
    return {
        "image_id": image_id,
        "image_size": img_size,
        "plate": {"cx": plate_cx, "cy": plate_cy, "radius": plate_r},
        "colonies": [
            {
                "cx": cx, "cy": cy, "radius": r,
                "nx": cx / img_size, "ny": cy / img_size, "nr": r / img_size,
            }
            for cx, cy, r in colonies
        ],
        "colony_count": len(colonies),
    }


def generate_dataset(
    output_dir: str,
    n_images: int = 5000,
    seed: int = 42,
    val_fraction: float = 0.1,
    tntc_fraction: float = 0.05,
) -> None:
    rng = np.random.default_rng(seed)
    random.seed(seed)

    output_path = Path(output_dir)
    (output_path / "images").mkdir(parents=True, exist_ok=True)
    (output_path / "labels").mkdir(parents=True, exist_ok=True)

    train_ids = []
    val_ids   = []
    all_annotations = []

    n_val = int(n_images * val_fraction)

    for i in tqdm(range(n_images), desc="Generating synthetic images"):
        image_id = f"syn_{i:06d}"
        is_val = i < n_val
        is_tntc = rng.random() < tntc_fraction

        img, (plate_cx, plate_cy, plate_r) = _generate_plate(rng)

        if is_tntc:
            n_colonies = int(rng.integers(301, 600))
        else:
            n_colonies = int(10 ** rng.uniform(0, math.log10(301)))

        palette = rng.choice(COLONY_PALETTES)
        colonies = _sample_colonies(rng, plate_cx, plate_cy, plate_r, n_colonies)

        for (cx, cy, r) in colonies:
            _draw_colony(rng, img, cx, cy, r, palette)

        img = _add_glare(rng, img, plate_cx, plate_cy, plate_r)

        alpha = rng.uniform(0.80, 1.20)
        beta  = rng.integers(-15, 15)
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        if rng.random() < 0.15:
            ksize = rng.choice([3, 5, 7])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)

        img_path = output_path / "images" / f"{image_id}.png"
        cv2.imwrite(str(img_path), img)

        ann = _make_annotation(image_id, IMAGE_SIZE, colonies[:300] if is_tntc else colonies,
                               plate_cx, plate_cy, plate_r)
        ann["is_tntc"] = is_tntc

        ann_path = output_path / "labels" / f"{image_id}.json"
        ann_path.write_text(json.dumps(ann, indent=2))

        all_annotations.append(ann)
        if is_val:
            val_ids.append(image_id)
        else:
            train_ids.append(image_id)

    index = {
        "n_images": n_images,
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "seed": seed,
        "train": train_ids,
        "val": val_ids,
    }
    (output_path / "dataset.json").write_text(json.dumps(index, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/synthetic")
    parser.add_argument("--n-images", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()
    generate_dataset(output_dir=args.output_dir, n_images=args.n_images,
                     seed=args.seed, val_fraction=args.val_fraction)