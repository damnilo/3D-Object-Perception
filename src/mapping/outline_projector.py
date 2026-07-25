from dataclasses import dataclass, field
from typing import List, Dict, TYPE_CHECKING

import cv2
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from src.slam.colmap_runner import CameraPose
from src.mapping.projector import backproject_build

if TYPE_CHECKING:
    from src.detection.segmenter import SegmentDetection

@dataclass
class ObjectOutlineSighting:
    class_name: str
    confidence: float
    frame_name: str
    outline_3d: np.ndarray
    centroid: np.ndarray

@dataclass
class OutlineMapObject:
    class_name: str
    centroid: np.ndarray
    outlines: List[np.ndarray] = field(default_factory=list)
    num_sightings: int = 0
    avg_confidence: float = 0.0

def extract_contour(mask: np.ndarray, stride: int=2) -> np.ndarray:

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.empty((0, 2))

    largest = max(contours, key=cv2.contourArea)
    points = largest.reshape(-1, 2)

    if stride > 1:
        points = points[::stride]

    return points

def backproject_outline(contour: np.ndarray, depth_map: np.ndarray, 
                        intrinsics: Dict, pose: CameraPose) -> np.ndarray:

    h, w = depth_map.shape
    pts_3d = []

    for x, y in contour:
        xi = int(np.clip(x, 0, w - 1))
        yi = int(np.clip(y, 0, h - 1))
        depth = float(depth_map[yi, xi])

        if depth <= 0:
            continue

        pts_3d.append(backproject_build((x, y), depth, intrinsics, pose))

    return np.array(pts_3d) if pts_3d else np.empty((0, 3))

def collect_outline_sightings(
        frame_segment: Dict[str, List['SegmentDetection']],
        depth_maps: Dict[str, np.ndarray], 
        poses: Dict[str, CameraPose], 
        intrinsics: Dict,
        contour_stride: int=2
) -> List[ObjectOutlineSighting]:

    sigthings = []

    for frame_name, detections in frame_segment.items():
        if frame_name not in depth_maps or frame_name not in poses:
            continue

        pose = poses[frame_name]
        depth_map = depth_maps[frame_name]

        for det in detections:
            contour = extract_contour(det.mask, stride=contour_stride)
            if len(contour) < 3:
                continue

            outline_3d = backproject_outline(contour, depth_map, intrinsics, pose)
            if len(outline_3d) < 3:
                continue

            sigthings.append(ObjectOutlineSighting(
                class_name=det.class_name,
                confidence=det.confidence,
                frame_name=frame_name,
                outline_3d=outline_3d,
                centroid=outline_3d.mean(axis=0)
            ))
    return sigthings

def cluster_outline(sightings: List[ObjectOutlineSighting], 
                    eps_meters: float = 1.5, min_samples: int = 2) -> List[OutlineMapObject]:

    map_objects = []
    by_class: Dict[str, List[ObjectOutlineSighting]] = {}

    for s in sightings:
        by_class.setdefault(s.class_name, []).append(s)

    for class_name, clusters in by_class.items():

        if len(clusters) < min_samples:
            continue

        centroids = np.array([s.centroid for s in clusters])
        if len(centroids) == 1:
            labels = np.array([1])
        else:
            Z = linkage(centroids, method='single')
            labels = fcluster(Z, t=eps_meters, criterion='distance')

        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            cluster_sightings = [s for s, l in zip(clusters, labels) if l == label]

            if len(cluster_sightings) < min_samples:
                continue

            centroid = np.mean([s.centroid for s in cluster_sightings], axis=0)
            avg_conf = float(np.mean([s.confidence for s in cluster_sightings]))

            map_objects.append(OutlineMapObject(
                class_name=class_name,
                centroid=centroid,
                outlines=[s.outline_3d for s in cluster_sightings],
                num_sightings=len(cluster_sightings),
                avg_confidence=avg_conf
            ))

    return map_objects