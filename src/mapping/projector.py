from dataclasses import dataclass
from typing import List, Dict, TYPE_CHECKING

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from src.slam.visual_odometry import CameraPose

if TYPE_CHECKING:
    from src.detection.detector import Detection

@dataclass
class ObjectSighting:
    class_name: str
    confidence: float
    position_3d: np.ndarray
    frame_name: str

@dataclass
class MapObject:
    class_name: str
    positions_3d: np.ndarray
    points_3d: np.ndarray
    frame_names: List[str]
    num_sightings: int
    avg_confidence: float

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

def collect_sightings(
        frame_detections: Dict[str, List['Detection']],
        depth_maps: Dict[str, np.ndarray], poses: Dict[str, CameraPose], intrinsics: Dict
) -> List[ObjectSighting]:

    sightings = []
    
    for frame_name, detections in frame_detections.items():
        if frame_name not in depth_maps or frame_name not in poses:
            continue

        pose = poses[frame_name]
        depth_map = depth_maps[frame_name]

        for det in detections:
            x, y = det.center_xy
            h, w = depth_map.shape
            xi, yi = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
            depth = float(depth_map[yi, xi])

            if depth <= 0:
                continue

            position_3d = backproject_build((x, y), depth, intrinsics, pose)

            sightings.append(ObjectSighting(
                class_name=det.class_name,
                confidence=det.confidence,
                position_3d=position_3d,
                frame_name=frame_name
            ))

    return sightings

def cluster_sightings(
        sightings: List[ObjectSighting], eps_meters: float = 1.5, min_samples: int = 2
) -> List[MapObject]:

    map_objects = []

    by_class: Dict[str, List[ObjectSighting]] = {}

    for s in sightings:
        by_class.setdefault(s.class_name, []).append(s)

    for class_name, clusters in by_class.items():

        if len(clusters) < min_samples:
            continue

        positions = np.array([s.position_3d for s in clusters])

        if len(positions) == 1:
            labels = np.array([1])
        else:
            Z = linkage(positions, method='single')
            labels = fcluster(Z, t=eps_meters, criterion='distance')

        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            cluster_sightings = [s for s, m in zip(clusters, mask) if m]

            if len(cluster_sightings) < min_samples:
                continue

            cluster_points = np.array([s.position_3d for s in cluster_sightings])
            centroid = np.mean(cluster_points, axis=0)
            avg_confidence = float(np.mean([s.confidence for s in cluster_sightings]))

            map_objects.append(MapObject(
                class_name=class_name,
                positions_3d=centroid,
                points_3d=cluster_points,
                frame_names=[s.frame_name for s in cluster_sightings],
                num_sightings=len(cluster_sightings),
                avg_confidence=avg_confidence
            ))

    return map_objects

def compute_depth(sparse_points, depth_map, pose, depth_at_fn, min_points: int = 5) -> float:

    ratios = []
    for x, y, point_world in sparse_points:
        point_cam = pose.rotation.T @ (point_world - pose.translation)
        colmap_depth = point_cam[2]

        raw_depth = depth_at_fn(depth_map, x, y)
        if colmap_depth > 0 and raw_depth > 0:
            ratios.append(colmap_depth / raw_depth)

    if  len(ratios) < min_points:
        print("Warning: No valid depth ratios found. Returning scale factor of 1.0.")
        return None

    return float(np.median(ratios))