import torch
import cv2
import numpy as np
from pathlib import Path


class CFUPipeline:
    def __init__(self):
        model_path = Path("app/ml/models/cfu_detector.pt")

        self.model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=str(model_path),
            source="local"
        )

        self.model.conf = 0.15
        self.model.iou = 0.45

    def run(self, image_bytes: bytes) -> dict:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = self.model(image)

        detections = results.xyxy[0]

        cfu_count = 0
        detection_list = []

        for *box, conf, cls in detections:
            x1, y1, x2, y2 = map(int, box)
            cfu_count += 1

            detection_list.append({
                "x": int((x1 + x2) / 2),
                "y": int((y1 + y2) / 2),
                "radius_px": int(max(x2 - x1, y2 - y1) / 2),
                "score": float(conf)
            })

        return {
            "cfu_count": int(cfu_count),
            "detections": detection_list
        }