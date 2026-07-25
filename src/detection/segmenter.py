from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from ultralytics import YOLO

@dataclass
class SegmentDetection:
    class_name: str
    confidence: float
    mask: np.ndarray
    bbox: tuple

    @property
    def center_xy(self):
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2, (y1 + y2) / 2


class RoadObjectSegmenter:

    def __init__(self, model_name: str = "yolov8n-seg.pt", confidence: float = 0.4,
                 target_classes: Optional[List[str]] = None):

        self.model = YOLO(model_name)
        self.confidence = confidence
        self.target_classes = target_classes if target_classes is not None else []

    def segment(self, frame_bgr: np.ndarray) -> List[SegmentDetection]:

        h, w = frame_bgr.shape[:2]
        results = self.model.predict(frame_bgr, conf=self.confidence, verbose=False)[0]

        detection = []
        if results.masks is None:
            return detection

        mask_data = results.masks.data.cpu().numpy()

        for i, box in enumerate(results.boxes):
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]

            if self.target_classes and class_name not in self.target_classes:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])

            mask = mask_data[i]
            if mask.shape != (h, w):
                import cv2
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            mask = mask.astype(bool)

            detection.append(SegmentDetection(
                class_name=class_name,
                confidence=conf,
                mask=mask,
                bbox=(x1, y1, x2, y2)
            ))

        return detection

