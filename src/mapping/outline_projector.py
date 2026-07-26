from dataclasses import dataclass, field
from typing import List, Dict, TYPE_CHECKING

import cv2
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from src.slam.visual_odometry import CameraPose
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

def extract_contour(mask: np.ndarray, stride: int=2,
                    erode_px: int=3, simplify_eps: float=2.0) -> np.ndarray:

    mask_u8 = mask.astype(np.uint8)

    if erode_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px, erode_px))
        eroded = cv2.erode(mask_u8, kernel)

        if eroded.sum() > 0:
            mask_u8 = eroded
        
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.empty((0, 2))

    largest = max(contours, key=cv2.contourArea)

    if simplify_eps > 0:
        param = cv2.arcLength(largest, True)
        largest = cv2.approxPolyDP(largest, epsilon=simplify_eps, closed=True)
        if len(largest) < 3 and param > 0:
            largest = max(contours, key=cv2.contourArea)

    points = largest.reshape(-1, 2)

    if stride > 1:
        points = points[::stride]

    return points

def _reject_outliers(pts_3d: List[np.ndarray], mad_thresh: float=3.0) -> np.ndarray:

    if len(pts_3d) == 0:
        return np.empty((0, 3))

    pts = np.array(pts_3d)

    if len(pts) < 4:
        return pts

    centroid = np.median(pts, axis=0)
    dists = np.linalg.norm(pts - centroid, axis=1)

    median_dist = np.median(dists)
    mad = np.median(np.abs(dists - median_dist))

    if mad == 0:
        return pts

    modified_z = 0.6745 * (dists - median_dist) / mad
    keep = np.abs(modified_z) < mad_thresh

    filtered = pts[keep]

    if len(filtered) < 3:
        return pts

    return filtered 
    

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

    pts_3d = _reject_outliers(pts_3d)

    return np.array(pts_3d)

def collect_outline_sightings(
        frame_segment: Dict[str, List['SegmentDetection']],
        depth_maps: Dict[str, np.ndarray], 
        poses: Dict[str, CameraPose], 
        intrinsics: Dict,
        contour_stride: int=2
) -> List[ObjectOutlineSighting]:

    sightings = []

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

            sightings.append(ObjectOutlineSighting(
                class_name=det.class_name,
                confidence=det.confidence,
                frame_name=frame_name,
                outline_3d=outline_3d,
                centroid=outline_3d.mean(axis=0)
            ))
    return sightings

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
            Z = linkage(centroids, method='complete')
            labels = fcluster(Z, t=eps_meters, criterion='distance')

        unique_labels = np.unique(labels)
        for label in unique_labels:
            cluster_sightings = [s for s, l in zip(clusters, labels) if l == label]

            if len(cluster_sightings) < min_samples:
                continue

            centroid = np.mean([s.centroid for s in cluster_sightings], axis=0)
            spread = np.mean([np.linalg.norm(s.centroid - centroid) for s in cluster_sightings])
            avg_conf = float(np.mean([s.confidence for s in cluster_sightings]))

            if spread > eps_meters:
                print(f"Note: {class_name} cluster has {len(cluster_sightings)} "
                      f"sightings with spread {spread:.2f}m, which exceeds eps_meters {eps_meters}.")

            map_objects.append(OutlineMapObject(
                class_name=class_name,
                centroid=centroid,
                outlines=[s.outline_3d for s in cluster_sightings],
                num_sightings=len(cluster_sightings),
                avg_confidence=avg_conf
            ))

    return map_objects