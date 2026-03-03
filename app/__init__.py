from __future__ import annotations
import hashlib, time
from dataclasses import dataclass
from typing import List, Optional
import cv2
import numpy as np

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

class CFUPipeline:
    MODEL_NAME = "opencv-cfu-detector"
    MODEL_VERSION = "1.0.0"

    def __init__(self, model_path="models/cfu_detector.pt"):
        self.model_path = model_path
        self.pipeline_hash = hashlib.md5(
            f"{self.MODEL_NAME}-{self.MODEL_VERSION}".encode()
        ).hexdigest()[:8]

    def run(self, image_bytes):
        t0 = time.time()
        nparr = np.frombuffer(image_bytes, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode image bytes")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        focus_score = float(min(1.0, cv2.Laplacian(gray, cv2.CV_64F).var() / 500.0))
        glare_score = float(np.sum(gray > 240) / gray.size)
        plate_mask, plate_circle = self._find_plate(gray)
        plate_found = plate_mask is not None
        if not plate_found:
            plate_mask = np.ones(gray.shape, dtype=np.uint8) * 255
        detections = self._detect_colonies(gray, plate_mask)
        plate_area = float(np.sum(plate_mask > 0))
        colony_area = sum(3.14159 * d.radius_px ** 2 for d in detections)
        overgrowth = plate_area > 0 and (colony_area / plate_area) > 0.60
        quality = QualityMetrics(
            plate_found=plate_found,
            focus_score=round(focus_score, 3),
            glare_score=round(glare_score, 3),
            overgrowth_detected=overgrowth,
        )
        reason_codes = []
        if not plate_found: reason_codes.append("PLATE_NOT_FOUND")
        if focus_score < 0.3: reason_codes.append("LOW_FOCUS")
        if glare_score > 0.5: reason_codes.append("HIGH_GLARE")
        if overgrowth: reason_codes.append("OVERGROWTH")
        conf = 1.0
        if not plate_found: conf *= 0.5
        conf *= max(0.3, focus_score)
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
            model_name=self.MODEL_NAME,
            model_version=self.MODEL_VERSION,
            pipeline_hash=self.pipeline_hash,
            processing_time_ms=processing_ms,
        )

    def _find_plate(self, gray):
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

    def _detect_colonies(self, gray, plate_mask):
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
                x=round(float(cx), 1), y=round(float(cy), 1),
                radius_px=round(float(radius), 1), score=round(float(score), 3),
            ))
        return detections

    def _draw_annotations(self, bgr, detections, plate_circle):
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