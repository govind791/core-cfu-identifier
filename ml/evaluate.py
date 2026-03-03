from __future__ import annotations
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
