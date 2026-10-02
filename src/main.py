import argparse
from pathlib import Path

import pickle
import cv2
import numpy as np
import yaml
from tqdm import tqdm

from src.mapping.projector import compute_depth, sample_patch_depth
from src.mapping.outline_projector import collect_outline_sightings, cluster_outline
from src.slam.visual_odometry import VisualOdometry
from src.depth.depth_estimator import DepthEstimator
from src.visualization.viewer import render_flythrough, save_map
from src.visualization.outline_viewer import render_scene
from src.detection.segmenter import RoadObjectSegmenter

SCALE_OUTLIER_RATIO = 4.0
CACHE_VERSION = 2
SLAM_CACHE_VERSION = 5
MOVABLE_CLASSES = {"person", "bicycle", "motorcycle", "car", "truck", "bus"}

def _cache_path(config: dict, filename: str, version: int = None) -> Path:

    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_version = CACHE_VERSION if version is None else version
    return output_dir / f"cache_v{cache_version}_{filename}.pkl"

def save_cache(config: dict, filename: str, data, version: int = None):

    with open(_cache_path(config, filename, version), 'wb') as f:
        pickle.dump(data, f)

def load_cache(config: dict, filename: str, version: int = None):

    cache_file = _cache_path(config, filename, version)
    if not cache_file.exists():
        return None

    with open(cache_file, 'rb') as f:
        return pickle.load(f)

def build_dynamic_masks(detections_per_frame: dict) -> dict:

    mask = {}
    for frame_name, detections in detections_per_frame.items():

        combined = None
        for det in detections:

            if det.class_name not in MOVABLE_CLASSES:
                continue

            if combined is None:
                combined = np.zeros_like(det.mask, dtype=np.uint8)

            combined |= det.mask.astype(np.uint8)

        if combined is not None:
            mask[frame_name] = combined

    return mask

def sample_raw_depth(depth_map, x, y):

    h, w = depth_map.shape
    xi, yi = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))

    return float(depth_map[yi, xi])

def _parse_intrinsics(intrinsics_raw: dict) -> dict:

    params = intrinsics_raw.get('params', {})
    model = intrinsics_raw["model"]

    if model == "SIMPLE_RADIAL":
        f, cx, cy = params[0], params[1], params[2]
        if len(params) > 3 and abs(float(params[3])) > 1e-8:
            print("Warning: SIMPLE_RADIAL distortion is ignored. Using pinhole intrinsics.")
        return {"fx": f, "fy": f, "cx": cx, "cy": cy}

    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params[0], params[1], params[2]
        return {"fx": f, "fy": f, "cx": cx, "cy": cy}

    elif model == "PINHOLE":
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}

    else:
        raise ValueError(f"Unsupported camera model: {model}")

def run_slam(config: dict, frames_dir: Path, dynamic_masks: dict = None):

    print("Running visual odometry for camera pose estimation...")
    slam = VisualOdometry(
        frames_dir=str(frames_dir),
        intrinsics=config.get("slam", {}).get("intrinsics"),
        dynamic_masks=dynamic_masks,
        max_consecutive_failures=config.get("slam", {}).get("max_consecutive_failures", 1),
        max_change_ratio=config.get("slam", {}).get("max_change_ratio", 3.0),
        max_forward_angle_deg=config.get("slam", {}).get("max_forward_angle_deg", 45.0),
        ego_mask_y=config.get("slam", {}).get("ego_mask_y", 0.70),
        backend=config.get("slam", {}).get("backend", "lk_vo"),
    )
    slam.run()

    poses = slam.read_poses()
    frame_order = sorted(poses.keys())
    print(f"First 5 frames in order: {frame_order[:5]}")
    print(f"Last 5 frames in order: {frame_order[-5:]}")
    print(f"Total poses: {len(poses)}")
    intrinsics = _parse_intrinsics(slam.read_intrinsics())
    env_points = slam.read_sparse_points()
    sparse_correspondences = slam.read_sparse_correspondences()

    print(f"Recovered {len(poses)} camera poses and intrinsics")
    return slam, poses, intrinsics, env_points, sparse_correspondences

def run_detection_and_depth(config: dict, frame_paths):

    print("Loading detection and depth models...")
    detection = RoadObjectSegmenter(
        model_name=config["detection"]["model"],
        confidence=config["detection"]["confidence"],
        target_classes=config["detection"]["target_classes"],
    )
    depth_estimator = DepthEstimator(
        model_name=config["depth"]["model"],
        device=config["depth"]["device"],
    )

    detections_per_frame = {}
    depth_maps_per_frame = {}

    print("Running detection and depth estimation on frames...")
    for frame_path in tqdm(frame_paths):

        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"Warning: Could not read frame {frame_path}. Skipping.")
            continue

        frame_name = frame_path.name
        detections_per_frame[frame_name] = detection.segment(frame)
        depth_maps_per_frame[frame_name] = depth_estimator.estimate(frame)

    return detections_per_frame, depth_maps_per_frame

def compute_frame_scales(depth_maps, poses, sparse_correspondences):

    frame_scales = {}

    for frame_name in depth_maps:
        if frame_name not in poses:
            print(f"Warning: No pose found for frame {frame_name}. Skipping scale computation.")
            frame_scales[frame_name] = None
            continue

        sparse_pts = sparse_correspondences.get(frame_name, [])
        frame_scales[frame_name] = compute_depth(
            sparse_pts, depth_maps[frame_name], poses[frame_name], sample_patch_depth, frame_name=frame_name
        )

    return frame_scales

def _locally_consistent_scales(frame_scales, outlier_ratio: float, half_window: int=5):

    names = sorted(name for name, scale in frame_scales.items() if scale is not None and scale > 0)
    scales = np.array([frame_scales[name] for name in names], dtype=float)
    keep = set()

    for i, name in enumerate(names):
        lo = max(0, i - half_window)
        hi = min(len(scales), i + half_window + 1)
        local = float(np.median(scales[lo:hi]))
        if local <= 0:
            continue
        ratio = scales[i] / local
        if 1.0 / outlier_ratio <= ratio <= outlier_ratio:
            keep.add(name)
        else:
            print(f"Warning: Scale for frame {name} is an outlier. Dropping frame as unreliable.")

    return keep

def apply_scales(depth_maps, detection_per_frame, frame_scales, outlier_ratio: float=SCALE_OUTLIER_RATIO):

    if not any(scale is not None and scale > 0 for scale in frame_scales.values()):
        print("Warning: No positive depth scales. Dropping all frames.")
        return {}, {}

    reliable_frames = _locally_consistent_scales(frame_scales, outlier_ratio)
    scaled_depth_maps = {
        name: depth_maps[name] * frame_scales[name]
        for name in reliable_frames
        if name in depth_maps
    }

    filtered_detections = {
        name: dets for name, dets in detection_per_frame.items() if name in reliable_frames
    }

    print(f"Kept {len(reliable_frames)} frames after scale filtering. Dropped {len(depth_maps) - len(reliable_frames)} frames.")

    return scaled_depth_maps, filtered_detections

def project_and_cluster(config: dict, detections_per_frame, depth_maps,
                        poses, intrinsics):

    print("Projecting detections into 3D space...")
    sightings = collect_outline_sightings(detections_per_frame, depth_maps, poses, intrinsics)

    print(f"Poses recovered: {len(poses)} / {len(depth_maps)} usable frames")
    print(f"Collected {len(sightings)} 3D sightings from detections.")

    map_objects = cluster_outline(
        sightings,
        min_samples=config["mapping"]["cluster_min_samples"],
        eps_meters=config["mapping"]["cluster_eps_meters"]
    )

    print(f"Found {len(map_objects)} clustered map objects from sightings.")

    return map_objects

def save_and_render(config: dict, map_objects, poses, env_points):

    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    vis = config.get("visualization", {})
    save_map(env_points, str(output_dir / "map.ply"), map_objects=map_objects)
    print(f"Saved map point cloud to {output_dir / 'map.ply'}")
    print(f"Objects: {len(map_objects)}, Poses: {len(poses)}")

    render_scene(
        map_objects,
        poses,
        output_path=str(output_dir / "scene.png"),
        show_trajectory=vis.get("show_camera_trajectory", True),
        show_axes=vis.get("show_axes", True),
    )

    if vis.get("flythrough", False):
        render_flythrough(
            map_objects,
            poses,
            env_points,
            output_path=str(output_dir / "flythrough.mp4"),
            point_size=float(vis.get("point_size", 3.0)),
        )

def run_pipeline(config: dict, use_cached: bool = True):

    frames_dir = Path(config["paths"]["frames_dir"])
    frame_paths = sorted(frames_dir.glob("*.jpg"))

    if not frame_paths:
        raise ValueError(f"No frames found in {frames_dir}. Please check the path.")

    cached = load_cache(config, "pipeline_cache") if use_cached else None
    if cached is not None:
        print("Loaded cached results.")
        detections_per_frame, depth_maps = cached
    else:
        detections_per_frame, depth_maps = run_detection_and_depth(config, frame_paths)
        save_cache(config, "pipeline_cache", (detections_per_frame, depth_maps))

    dynamic_masks = build_dynamic_masks(detections_per_frame)
    print(f"Built dynamic masks for {len(dynamic_masks)} frames based on detections.")

    cached_slam = load_cache(config, "slam_cache", version=SLAM_CACHE_VERSION) if use_cached else None
    if cached_slam is not None:
        print("Loaded cached SLAM results.")
        poses, intrinsics, env_points, sparse_correspondences = cached_slam
    else:
        slam, poses, intrinsics, env_points, sparse_correspondences = run_slam(config, frames_dir, dynamic_masks)
        save_cache(
            config, "slam_cache",
            (poses, intrinsics, env_points, sparse_correspondences),
            version=SLAM_CACHE_VERSION,
        )

    frame_scales = compute_frame_scales(depth_maps, poses, sparse_correspondences)
    depth_maps, detections_per_frame = apply_scales(depth_maps, detections_per_frame, frame_scales)
    map_objects = project_and_cluster(config, detections_per_frame, depth_maps, poses, intrinsics)
    save_and_render(config, map_objects, poses, env_points)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Object Perception Pipeline")
    parser.add_argument("--config", type=str, default="configs/pipeline.yaml")
    parser.add_argument("--flythrough", action="store_true", help="Enable flythrough rendering")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching of intermediate results")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.flythrough:
        config.setdefault("visualization", {})["flythrough"] = True

    run_pipeline(config, use_cached=not args.no_cache)
        