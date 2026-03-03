from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ── Neural Network Architecture ───────────────────────────────────────────

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
        c4_up = F.interpolate(c4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.neck.outputs[0](self.neck.laterals[0](c3) + self.neck.laterals[1](c4_up))
        fused = F.interpolate(fused, scale_factor=2, mode="bilinear", align_corners=False)
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


# ── Result Dataclasses ────────────────────────────────────────────────────

@dataclass
class Detection:
    x: float
    y: float
    radius_px: float
    score: float


@dataclass
class QualityMetrics:
    plate_found: bool
    focus_score: float
    glare_score: float
    overgrowth_detected: bool


@dataclass
class PipelineResult:
    cfu_count_total: int
    detections: List[Detection]
    quality: QualityMetrics
    overall_confidence: float
    needs_review: bool
    reason_codes: List[str]
    annotated_image_bytes: Optional[bytes]
    model_name: str
    model_version: str
    pipeline_hash: str
    processing_time_ms: float


# ── Main Pipeline ─────────────────────────────────────────────────────────

class CFUPipeline:
    MODEL_NAME = "cfu-detector"
    MODEL_VERSION = "1.0.0"

    # Detection thresholds
    HM_THRESHOLD = 0.45       # heatmap confidence threshold (neural net mode)
    NMS_DISTANCE = 8          # min pixel distance between detections
    FOCUS_THRESHOLD = 0.15    # below this = LOW_FOCUS flag

    def __init__(self, model_path: str = "models/cfu_detector.pt"):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._nn_model: Optional[CFUDetectorModel] = None
        self._use_nn = False

        # Try loading the trained neural network model
        try:
            import os
            if os.path.exists(model_path):
                self._nn_model = CFUDetectorModel.load(model_path, device=self.device)
                self._nn_model.eval()
                self._use_nn = True
                print(f"[CFUPipeline] Loaded neural network model from {model_path}")
            else:
                print(f"[CFUPipeline] No model file at {model_path}, using OpenCV fallback")
        except Exception as e:
            print(f"[CFUPipeline] Failed to load neural network: {e}, using OpenCV fallback")

        self.pipeline_hash = hashlib.md5(
            f"{self.MODEL_NAME}-{self.MODEL_VERSION}-{'nn' if self._use_nn else 'cv'}".encode()
        ).hexdigest()[:8]

    def run(self, image_bytes: bytes) -> PipelineResult:
        t0 = time.time()

        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode image bytes")

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Quality metrics
        focus_score = float(min(1.0, cv2.Laplacian(gray, cv2.CV_64F).var() / 500.0))
        glare_score = float(np.sum(gray > 240) / gray.size)

        # Find plate boundary
        plate_mask, plate_circle = self._find_plate(gray)
        plate_found = plate_mask is not None
        if not plate_found:
            plate_mask = np.ones(gray.shape, dtype=np.uint8) * 255

        # Run detection
        if self._use_nn and self._nn_model is not None:
            detections = self._detect_nn(bgr, plate_mask)
            model_name = f"{self.MODEL_NAME}-neural"
        else:
            detections = self._detect_opencv(gray, plate_mask)
            model_name = f"{self.MODEL_NAME}-opencv"

        # Overgrowth check
        plate_area = float(np.sum(plate_mask > 0))
        colony_area = sum(3.14159 * d.radius_px ** 2 for d in detections)
        overgrowth = plate_area > 0 and (colony_area / plate_area) > 0.60

        quality = QualityMetrics(
            plate_found=plate_found,
            focus_score=round(focus_score, 3),
            glare_score=round(glare_score, 3),
            overgrowth_detected=overgrowth,
        )

        # Flags
        reason_codes: List[str] = []
        if not plate_found:
            reason_codes.append("PLATE_NOT_FOUND")
        if focus_score < self.FOCUS_THRESHOLD:
            reason_codes.append("LOW_FOCUS")
        if glare_score > 0.5:
            reason_codes.append("HIGH_GLARE")
        if overgrowth:
            reason_codes.append("OVERGROWTH")

        conf = 1.0
        if not plate_found:
            conf *= 0.5
        conf *= max(0.5, focus_score)
        conf *= max(0.5, 1.0 - glare_score)
        overall_confidence = float(min(1.0, conf))
        needs_review = bool(reason_codes) or overall_confidence < 0.6

        annotated_bytes = self._draw_annotations(bgr, detections, plate_circle)
        processing_ms = round((time.time() - t0) * 1000, 1)

        return PipelineResult(
            cfu_count_total=len(detections),
            detections=detections,
            quality=quality,
            overall_confidence=round(overall_confidence, 3),
            needs_review=needs_review,
            reason_codes=reason_codes,
            annotated_image_bytes=annotated_bytes,
            model_name=model_name,
            model_version=self.MODEL_VERSION,
            pipeline_hash=self.pipeline_hash,
            processing_time_ms=processing_ms,
        )

    # ── Neural Network Detection ──────────────────────────────────────────

    def _detect_nn(self, bgr: np.ndarray, plate_mask: np.ndarray) -> List[Detection]:
        h_orig, w_orig = bgr.shape[:2]
        size = CFUDetectorModel.INPUT_SIZE

        # Preprocess
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (size, size))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        tensor = tensor.unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            hm, sz = self._nn_model(tensor)

        hm_np = hm[0, 0].cpu().numpy()
        sz_np = sz[0, 0].cpu().numpy()

        # Scale factors back to original image
        scale_x = w_orig / hm_np.shape[1]
        scale_y = h_orig / hm_np.shape[0]

        # Peak extraction with NMS
        detections = []
        peaks = np.argwhere(hm_np > self.HM_THRESHOLD)

        for yx in peaks:
            cy_hm, cx_hm = yx
            score = float(hm_np[cy_hm, cx_hm])
            radius_hm = float(sz_np[cy_hm, cx_hm])

            cx_orig = cx_hm * scale_x
            cy_orig = cy_hm * scale_y
            radius_orig = radius_hm * ((scale_x + scale_y) / 2)

            # Check inside plate mask
            ix, iy = int(cx_orig), int(cy_orig)
            if 0 <= iy < plate_mask.shape[0] and 0 <= ix < plate_mask.shape[1]:
                if plate_mask[iy, ix] == 0:
                    continue

            # Simple NMS
            too_close = False
            for d in detections:
                dist = math.sqrt((d.x - cx_orig) ** 2 + (d.y - cy_orig) ** 2)
                if dist < self.NMS_DISTANCE:
                    too_close = True
                    break
            if too_close:
                continue

            detections.append(Detection(
                x=round(cx_orig, 1),
                y=round(cy_orig, 1),
                radius_px=round(max(3.0, radius_orig), 1),
                score=round(score, 3),
            ))

        return detections

    # ── OpenCV Fallback Detection ─────────────────────────────────────────

    def _detect_opencv(self, gray: np.ndarray, plate_mask: np.ndarray) -> List[Detection]:
        masked = cv2.bitwise_and(gray, gray, mask=plate_mask)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(masked)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh = cv2.bitwise_and(thresh, thresh, mask=plate_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        h, w = gray.shape
        min_area = (h * w) * 0.00005
        max_area = (h * w) * 0.05

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * 3.14159 * area / (perimeter ** 2)
            if circularity < 0.35:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            score = min(1.0, circularity * (area / max_area) ** 0.1)
            detections.append(Detection(
                x=round(float(cx), 1),
                y=round(float(cy), 1),
                radius_px=round(float(radius), 1),
                score=round(float(score), 3),
            ))

        return detections

    # ── Plate Detection ───────────────────────────────────────────────────

    def _find_plate(self, gray: np.ndarray):
        h, w = gray.shape
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=min(h, w) * 0.5, param1=60, param2=40,
            minRadius=int(min(h, w) * 0.25), maxRadius=int(min(h, w) * 0.55),
        )
        if circles is None:
            return None, None
        cx, cy, r = np.round(circles[0][0]).astype(int)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, (cx, cy), int(r * 0.95), 255, -1)
        return mask, (cx, cy, r)

    # ── Annotation ────────────────────────────────────────────────────────

    def _draw_annotations(self, bgr: np.ndarray, detections: List[Detection], plate_circle) -> bytes:
        out = bgr.copy()
        if plate_circle is not None:
            cx, cy, r = plate_circle
            cv2.circle(out, (cx, cy), r, (0, 255, 0), 2)
        for d in detections:
            cx, cy = int(d.x), int(d.y)
            r = max(3, int(d.radius_px))
            cv2.circle(out, (cx, cy), r, (0, 0, 255), 1)
            cv2.circle(out, (cx, cy), 2, (0, 0, 255), -1)
        cv2.putText(out, f"CFU: {len(detections)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        _, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return buf.tobytes()