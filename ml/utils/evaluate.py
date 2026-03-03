"""
Evaluation Metrics
==================
Computes:
  - Validation loss (focal + size)
  - Colony Count MAE
  - Detection AP@0.5 IoU (using circle IoU)
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import cv2
from tqdm import tqdm


def circle_iou(cx1, cy1, r1, cx2, cy2, r2) -> float:
    d = math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        smaller = min(r1, r2) ** 2
        larger  = max(r1, r2) ** 2
        return (math.pi * smaller) / (math.pi * larger)

    a = (r1 ** 2) * math.acos((d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1))
    b = (r2 ** 2) * math.acos((d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2))
    c = 0.5 * math.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
    intersection = a + b - c
    union = math.pi * (r1 ** 2 + r2 ** 2) - intersection
    return intersection / max(union, 1e-7)


def decode_heatmap_predictions(
    heatmap: np.ndarray,
    size_map: np.ndarray,
    stride: int = 4,
    score_thresh: float = 0.3,
    nms_radius: int = 3,
) -> list[tuple[float, float, float, float]]:
    hm = heatmap[0]
    sz = size_map[0]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (nms_radius * 2 + 1,) * 2)
    local_max = cv2.dilate(hm, kernel)
    peaks = (hm == local_max) & (hm > score_thresh)

    ys, xs = np.where(peaks)
    detections = []
    for y, x in zip(ys, xs):
        score  = float(hm[y, x])
        radius = float(sz[y, x]) * stride
        cx_px  = (x + 0.5) * stride
        cy_px  = (y + 0.5) * stride
        radius = max(2.0, radius)
        detections.append((cx_px, cy_px, radius, score))

    return detections


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device: str) -> dict[str, float]:
    from ml.model import CFUDetectorLoss

    model.eval()
    total_loss = total_focal = total_size = 0.0
    n_batches = 0
    count_errors = []

    for batch in tqdm(loader, desc="  val  ", leave=False):
        img    = batch["image"].to(device)
        hm_tgt = batch["heatmap"].to(device)
        sz_tgt = batch["size_map"].to(device)
        gt_counts = batch["n_colonies"].numpy()

        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            hm_pred, sz_pred = model(img)
            losses = criterion(hm_pred, sz_pred, hm_tgt, sz_tgt)

        total_loss  += losses["total"].item()
        total_focal += losses["focal"].item()
        total_size  += losses["size"].item()
        n_batches   += 1

        hm_np = hm_pred.cpu().numpy()
        sz_np = sz_pred.cpu().numpy()
        for i in range(len(img)):
            preds = decode_heatmap_predictions(hm_np[i], sz_np[i])
            count_errors.append(abs(len(preds) - int(gt_counts[i])))

    n = max(1, n_batches)
    return {
        "loss":      total_loss  / n,
        "focal":     total_focal / n,
        "size":      total_size  / n,
        "count_mae": float(np.mean(count_errors)) if count_errors else 0.0,
    }


def compute_detection_ap(all_preds, all_gts, iou_threshold=0.5) -> float:
    tp_list  = []
    fp_list  = []
    scores   = []
    n_gt_total = sum(len(g) for g in all_gts)

    for preds, gts in zip(all_preds, all_gts):
        matched = [False] * len(gts)
        sorted_preds = sorted(preds, key=lambda x: x[3], reverse=True)

        for cx, cy, r, score in sorted_preds:
            scores.append(score)
            best_iou = 0.0
            best_idx = -1
            for j, (gcx, gcy, gr) in enumerate(gts):
                if matched[j]:
                    continue
                iou = circle_iou(cx, cy, r, gcx, gcy, gr)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j

            if best_iou >= iou_threshold and best_idx >= 0:
                tp_list.append(1)
                fp_list.append(0)
                matched[best_idx] = True
            else:
                tp_list.append(0)
                fp_list.append(1)

    if not scores:
        return 0.0

    order = np.argsort(scores)[::-1]
    tp = np.array(tp_list)[order]
    fp = np.array(fp_list)[order]

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    precision = cum_tp / (cum_tp + cum_fp + 1e-7)
    recall    = cum_tp / max(n_gt_total, 1)

    ap = 0.0
    for t in np.linspace(0, 1, 101):
        mask = recall >= t
        ap += precision[mask].max() if mask.any() else 0.0
    return ap / 101