from dataclasses import dataclass, field
from typing import List, Dict, TYPE_CHECKING

import cv2
import numpy as np

from src.slam.visual_odometry import CameraPose
from src.mapping.projector import backproject_build, sample_patch_depth

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
    frame_names: List[str] = field(default_factory=list)

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
                        intrinsics: Dict, pose: CameraPose,
                        depth_band_m: float=2.5, max_extent: float=12.0) -> np.ndarray:

    pts_3d = []
    depths = []

    for x, y in contour:
        depth = sample_patch_depth(depth_map, x, y)

        if not np.isfinite(depth) or depth <= 0:
            continue

        depths.append(depth)
        pts_3d.append(backproject_build((float(x), float(y)), depth, intrinsics, pose))

    if len(pts_3d) < 3:
        return np.empty((0, 3))

    depths = np.asarray(depths, dtype=float)
    pts = np.asarray(pts_3d, dtype=float)
    median_depth = float(np.median(depths))
    if median_depth <= 0:
        return np.empty((0, 3))

    # A vehicle occupies one distance. Background mixed into the mask
    # stretches the contour into a ray along the camera's view.
    band = max(depth_band_m, 0.15 * median_depth)
    same_distance = np.abs(depths - median_depth) <= band
    pts = _reject_outliers(list(pts[same_distance]))

    if len(pts) < 3:
        return np.empty((0, 3))

    if float(np.max(np.ptp(pts, axis=0))) > max_extent:
        return np.empty((0, 3))

    return np.asarray(pts, dtype=float)

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

def _closed_outline(points: np.ndarray) -> np.ndarray:

    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return pts

    flat = pts.copy()
    flat[:, 1] = float(np.median(pts[:, 1]))
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(flat[:, [0, 2]])
        loop = flat[hull.vertices]
    except Exception:
        loop = flat

    return _smooth_ring(loop)

def _smooth_ring(points: np.ndarray, samples: int=28) -> np.ndarray:

    ring = np.asarray(points, dtype=float)
    if len(ring) < 3:
        return ring

    closed = np.vstack([ring, ring[0]])
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    keep = np.concatenate([[True], lengths > 1e-3])
    closed = closed[keep]
    if len(closed) < 4:
        return ring

    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    distance = np.concatenate([[0.0], np.cumsum(lengths)])
    total = float(distance[-1])
    if total < 1e-3:
        return ring

    distance /= total
    targets = np.linspace(0.0, 1.0, samples, endpoint=False)
    return np.stack([np.interp(targets, distance, closed[:, axis]) for axis in range(3)], axis=1)

def _tracks_for_class(sightings: List[ObjectOutlineSighting], eps_meters: float):

    ordered = sorted(sightings, key=lambda s: s.frame_name)
    tracks: List[List[ObjectOutlineSighting]] = []
    centers: List[np.ndarray] = []

    for sighting in ordered:
        if not centers:
            tracks.append([sighting])
            centers.append(sighting.centroid.copy())
            continue

        distances = [float(np.linalg.norm(sighting.centroid - center)) for center in centers]
        nearest = int(np.argmin(distances))
        if distances[nearest] <= eps_meters:
            tracks[nearest].append(sighting)
            centers[nearest] = np.median([item.centroid for item in tracks[nearest]], axis=0)
        else:
            tracks.append([sighting])
            centers.append(sighting.centroid.copy())

    return tracks

def cluster_outline(sightings: List[ObjectOutlineSighting], 
                    eps_meters: float = 1.5, min_samples: int = 2) -> List[OutlineMapObject]:

    map_objects = []
    by_class: Dict[str, List[ObjectOutlineSighting]] = {}

    for s in sightings:
        by_class.setdefault(s.class_name, []).append(s)

    for class_name, class_sightings in by_class.items():

        if len(class_sightings) < min_samples:
            continue

        for cluster_sightings in _tracks_for_class(class_sightings, eps_meters):

            if len(cluster_sightings) < min_samples:
                continue

            member_centroids = np.array([s.centroid for s in cluster_sightings])
            centroid = np.median(member_centroids, axis=0)
            avg_conf = float(np.mean([s.confidence for s in cluster_sightings]))
            central = min(
                cluster_sightings,
                key=lambda s: float(np.linalg.norm(s.centroid - centroid))
            )

            map_objects.append(OutlineMapObject(
                class_name=class_name,
                centroid=centroid,
                outlines=[_closed_outline(central.outline_3d)],
                num_sightings=len(cluster_sightings),
                avg_confidence=avg_conf,
                frame_names=[s.frame_name for s in cluster_sightings]
            ))

    return map_objects