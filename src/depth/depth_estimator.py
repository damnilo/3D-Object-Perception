import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import pipeline

class DepthEstimator:

    def __init__(self, model_name: str = "depth-anything/Depth-Anything-V2-Small-hf", device = "cuda"):

        self.device = device if torch.cuda.is_available() else "cpu"
        self.pipeline = pipeline(task="depth-estimation", model=model_name, device=self.device)

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:

        rgb = frame_bgr[..., ::-1]
        image = Image.fromarray(rgb)
        h, w = frame_bgr.shape[:2]

        result = self.pipeline(image)

        raw = result["predicted_depth"]
        if raw.dim() == 2:

            raw = raw.unsqueeze(0).unsqueeze(0)
        elif raw.dim() == 3:

            raw = raw.unsqueeze(0)

        raw = F.interpolate(raw, size=(h, w), mode="bicubic", align_corners=False)
        depth = raw.squeeze().cpu().numpy().astype(np.float32)

        depth = np.clip(depth, 1e-6, None)
        depth = 1.0 / depth

        return depth

    def depth_in_bbox(self, depth_map: np.ndarray, bbox_xyxy: tuple, center_fraction: float = 0.3) -> float:
        x1, y1, x2, y2 = bbox_xyxy
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        hw, hh = w * center_fraction / 2, h * center_fraction / 2
        patch = depth_map[int(cy - hh):int(cy + hh), int(cx - hw):int(cx + hw)]

        return float(np.median(patch)) if patch.size > 0 else 0.0

    def depth_at_point(self, depth_map: np.ndarray, x: int, y: int, patch_radius: int=3) -> float:

        h, w = depth_map.shape
        xi, yi = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))

        x0, x1 = max(0, xi - patch_radius), min(w, xi + patch_radius + 1)
        y0, y1 = max(0, yi - patch_radius), min(h, yi + patch_radius + 1)
        patch = depth_map[y0:y1, x0:x1]

        return float(np.median(patch)) if patch.size > 0 else 0.0