import numpy as np
import torch
from PIL import Image
from transformers import pipeline

class DepthEstimator:

    def __init__(self, model_name: str = "depth-anything/Depth-Anything-V2-Small-hf", device = "cuda"):

        self.device = device if torch.cuda.is_available() else "cpu"
        self.pipeline = pipeline(task="depth-estimation", model=model_name, device=self.device)

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:

        rgb = frame_bgr[..., ::-1]
        image = Image.fromarray(rgb)

        result = self.pipeline(image)
        depth = np.array(result["depth"], dtype=np.float32)

        return depth

    def depth_in_bbox(self, depth_map: np.ndarray, bbox_xyxy: tuple, center_fraction: float = 0.3) -> float:
        x1, y1, x2, y2 = bbox_xyxy
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        hw, hh = w * center_fraction / 2, h * center_fraction / 2
        patch = depth_map[int(cy - hh):int(cy + hh), int(cx - hw):int(cx + hw)]

        return float(np.median(patch)) if patch.size > 0 else 0.0