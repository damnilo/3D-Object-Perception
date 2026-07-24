from dataclasses import dataclass
from typing import List

import numpy as np
from ultralytics import YOLO

@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: tuple

    @property
    def center_xy(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

class RoadObjectDetector:

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float=0.4, target_classes: List[str]=None):

        self.model = YOLO(model_name)
        self.confidence = confidence
        self.target_classes = target_classes if target_classes is not None else []

    def detect(self, frame: np.ndarray) -> List[Detection]:

        results = self.model.predict(frame, conf=self.confidence, verbose=False)[0]

        detections = []

        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]

            if self.target_classes and class_name not in self.target_classes:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])

            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2)
                )
            )

        return detections