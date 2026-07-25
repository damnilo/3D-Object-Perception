import argparse
from pathlib import Path

import cv2
import yaml
import numpy as np
from tqdm import tqdm

from src.mapping.projector import collect_sightings, cluster_sightings, compute_depth
from src.slam.colmap_runner import ColmapRunner
from src.visualization.viewer import render_map, save_map
from src.depth.depth_estimator import DepthEstimator
from src.detection.detector import RoadObjectDetector

def sample_raw_depth(depth_map, x, y):
    h, w = depth_map.shape
    xi, yi = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))

    return float(depth_map[yi, xi])

def run_pipeline(config: dict):

    frames_dir = Path(config["paths"]["frames_dir"])
    frame_paths = sorted(frames_dir.glob("*.jpg"))

    if not frame_paths:
        raise ValueError(f"No frames found in {frames_dir}")

    print("Running COLMAP for camera pose estimation...")
    colmap = ColmapRunner(workspace=config["paths"]["colmap_workspace"], frames_dir=str(frames_dir), matcher=config["slam"]["matcher"])

    colmap.run()
    poses = colmap.read_poses()
    intrinsics_raw = colmap.read_intrinsics()
    intrinsics = _parse_intrinsics(intrinsics_raw)
    print(f"Recovered {len(poses)} camera poses and intrinsics.")

    print("Loading detection and depth models...")
    detector = RoadObjectDetector(
        confidence = config["detection"]["confidence"],
        target_classes = config["detection"]["target_classes"]
    )
    depth_estimator = DepthEstimator(
        device=config["depth"]["device"]
    )

    detections_by_frame = {}
    depth_maps = {}

    print("Running detection and depth estimation on frames...")
    for frame_path in tqdm(frame_paths):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"Warning: Could not read frame {frame_path}. Skipping.")
            continue

        frame_name = frame_path.name
        detections_by_frame[frame_name] = detector.detect(frame)
        depth_maps[frame_name] = depth_estimator.estimate(frame)

    sparse_correspondences = colmap.read_sparse_correspondences()
    
    for frame_name, depth_map in depth_maps.items():
        if frame_name not in poses:
            print(f"Warning: No pose found for frame {frame_name}. Skipping depth scaling.")
            continue

        sparse_pts = sparse_correspondences.get(frame_name, [])
        scale = compute_depth(sparse_pts, depth_map, poses[frame_name], sample_raw_depth)
        depth_maps[frame_name] = depth_map * scale

    print("Projecting detections into 3D space...")
    sightings = collect_sightings(detections_by_frame, depth_maps, poses, intrinsics)
    print(f"Poses recovered: {len(poses)} / {len(frame_paths)} frames")
    print(f"Raw sightings before filtering: {len(sightings)}")
    map_objects = cluster_sightings(
        sightings,
        eps_meters=config["mapping"]["cluster_eps_meters"],
        min_samples=config["mapping"]["cluster_min_samples"]
    )

    print(f"Found {len(map_objects)} unique objects in the scene.")

    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_map(map_objects, poses, str(output_dir / "map.ply"))
    print(f"Saved 3D map to {output_dir / 'map.ply'}")

    render_map(map_objects, poses, 
               show_trajectory=config["visualization"]["show_camera_trajectory"], 
               show_axes=config["visualization"]["show_axes"], 
               point_size=config["visualization"]["point_size"],
               output_path=str(output_dir / "map_render.png")
    )

def _parse_intrinsics(intrinsics_raw: dict) -> dict:

    params = intrinsics_raw.get("params", [])
    if intrinsics_raw["model"] in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
        f, cx, cy = params[0], params[1], params[2]
        return {"fx": f, "fy": f, "cx": cx, "cy": cy}
    elif intrinsics_raw["model"] == "PINHOLE":
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
    else:
        raise ValueError(f"Unsupported camera model: {intrinsics_raw['model']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pipeline.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_pipeline(config)