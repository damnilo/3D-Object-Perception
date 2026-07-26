from typing import Dict, List

import numpy as np
import open3d as o3d

from src.mapping.outline_projector import OutlineMapObject
from src.slam.visual_odometry import CameraPose

CLASS_COLORS = {
    "car": [0.2, 0.4, 0.9],
    "truck": [0.1, 0.2, 0.6],
    "bus": [0.3, 0.6, 0.9],
    "motorcycle": [0.9, 0.5, 0.1],
    "bicycle": [0.9, 0.7, 0.2],
    "person": [0.9, 0.1, 0.1],
    "traffic light": [0.1, 0.9, 0.2],
    "stop sign": [0.9, 0.1, 0.6]
}
DEFAULT_COLOR = [0.8, 0.8, 0.8]

def build_outline(outline_3d: np.ndarray, color) -> o3d.geometry.LineSet:

    n = len(outline_3d)
    lines = [[i, (i + 1) % n] for i in range(n)]

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(outline_3d)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([color for _ in range(n)])
    return ls

def build_scene(map_objects: List[OutlineMapObject], poses: Dict[str, CameraPose],
                show_trajectory: bool=True):

    geometries = []

    for obj in map_objects:
        color = CLASS_COLORS.get(obj.class_name, DEFAULT_COLOR)
        for outline in obj.outlines:
            ls = build_outline(outline, color)
            geometries.append(ls)

    if show_trajectory and poses:
        sorted_poses = sorted(poses.values(), key=lambda x: x.frame_name)
        points = np.array([pose.translation for pose in sorted_poses])
        lines = [[i, i + 1] for i in range(len(points) - 1)]
        traj = o3d.geometry.LineSet()
        traj.points = o3d.utility.Vector3dVector(points)
        traj.lines = o3d.utility.Vector2iVector(lines)
        traj.colors = o3d.utility.Vector3dVector([[0.9, 0.1, 0.1] for _ in range(len(lines))])
        geometries.append(traj)

    return geometries

def render_scene(map_objects: List[OutlineMapObject],
                 poses: Dict[str, CameraPose], output_path: str):

    geometries = build_scene(map_objects, poses)

    if output_path:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Outline Map", width=1280, height=720, visible=False)
        for g in geometries:
            vis.add_geometry(g)

        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(output_path)
        vis.destroy_window()

    else:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Outline Map", width=1280, height=720)
        for g in geometries:
            vis.add_geometry(g)

        vis.run()
        vis.destroy_window()