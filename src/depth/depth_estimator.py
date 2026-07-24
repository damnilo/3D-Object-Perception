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

    def depth_at(self, depth_map: np.ndarray, x: float, y: float) -> float:

        h, w = depth_map.shape
        xi, yi = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
        return float(depth_map[yi, xi])