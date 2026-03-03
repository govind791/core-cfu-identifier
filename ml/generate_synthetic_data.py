from __future__ import annotations
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
