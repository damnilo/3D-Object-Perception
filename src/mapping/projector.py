from dataclasses import dataclass
from typing import List, Dict, TYPE_CHECKING

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from src.slam.visual_odometry import CameraPose


def sample_patch_depth(depth_map: np.ndarray, x: float, y: float, patch_radius: int = 3) -> float:

    h, w = depth_map.shape
    xi, yi = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))

    x0, x1 = max(0, xi - patch_radius), min(w, xi + patch_radius + 1)
    y0, y1 = max(0, yi - patch_radius), min(h, yi + patch_radius + 1)
    patch = depth_map[y0:y1, x0:x1]

    return float(np.median(patch)) if patch.size > 0 else 0.0

def backproject_build(pixel_xy: tuple, depth: float, intrinsics: Dict, pose: CameraPose
                      ) -> np.ndarray:

    x, y = pixel_xy
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']

    x_cam = (x - cx) * depth / fx
    y_cam = (y - cy) * depth / fy
    z_cam = depth
    point_cam = np.array([x_cam, y_cam, z_cam])

    point_world = pose.rotation @ point_cam + pose.translation

    return point_world

def compute_depth(sparse_points, depth_map, pose, depth_at_fn, min_points: int = 8,
                  max_reasonable_ratio: float = 100.0) -> float:

    ratios = []
    for x, y, point_world in sparse_points:
        point_cam = pose.rotation.T @ (point_world - pose.translation)
        colmap_depth = point_cam[2]

        raw_depth = depth_at_fn(depth_map, x, y)
        if colmap_depth > 0 and raw_depth > 0:
            ratio = colmap_depth / raw_depth
            if ratio < max_reasonable_ratio:
                ratios.append(ratio)

    if  len(ratios) < min_points:
        print("Warning: No valid depth ratios found. Returning None.")
        return None

    return float(np.median(ratios))