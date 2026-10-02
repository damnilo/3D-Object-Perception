from typing import Dict, List

import numpy as np
import open3d as o3d

from src.mapping.outline_projector import OutlineMapObject
from src.slam.visual_odometry import CameraPose
from src.visualization.viewer import CLASS_COLORS, _fit_view

DEFAULT_COLOR = [0.8, 0.8, 0.8]

def _rotation_from_z(direction: np.ndarray) -> np.ndarray:

    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, direction)
    c = float(np.dot(z, direction))
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))

def polyline_tube(points: np.ndarray, radius: float, color, resolution: int=8) -> o3d.geometry.TriangleMesh:

    points = np.asarray(points, dtype=float)
    parts = []
    for a, b in zip(points[:-1], points[1:]):
        delta = b - a
        length = float(np.linalg.norm(delta))
        if length < 1e-4:
            continue
        direction = delta / length
        cyl = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius, height=length + radius, resolution=resolution, split=1
        )
        cyl.rotate(_rotation_from_z(direction), center=np.zeros(3))
        cyl.translate((a + b) / 2.0)
        parts.append(cyl)

    tube = o3d.geometry.TriangleMesh()
    if parts:
        tube = parts[0]
        for part in parts[1:]:
            tube += part
    tube.paint_uniform_color(color)
    tube.compute_vertex_normals()
    return tube

def _trajectory_points(poses: Dict[str, CameraPose]) -> np.ndarray:

    ordered = sorted(poses.values(), key=lambda pose: pose.frame_name)
    return np.array([pose.translation.reshape(3) for pose in ordered], dtype=float)

def _spread_for_display(outline: np.ndarray, path: np.ndarray,
                        shape_scale: float=3.5, min_offset: float=8.0) -> np.ndarray:

    outline = np.asarray(outline, dtype=float)
    centroid = outline.mean(axis=0)
    nearest = int(np.argmin(np.linalg.norm(path - centroid, axis=1)))
    i0 = max(0, nearest - 1)
    i1 = min(len(path) - 1, nearest + 1)
    tangent = path[i1] - path[i0]
    tangent[1] = 0.0
    norm = float(np.linalg.norm(tangent))
    if norm < 1e-6:
        tangent = np.array([1.0, 0.0, 0.0])
    else:
        tangent /= norm
    lateral = np.array([-tangent[2], 0.0, tangent[0]])

    offset = centroid - path[nearest]
    offset[1] = 0.0
    side = float(np.dot(offset, lateral))
    side = 1.0 if side >= 0.0 else -1.0
    distance = abs(float(np.dot(offset, lateral)))
    pushed = max(distance * shape_scale, min_offset)
    center = path[nearest] + lateral * side * pushed
    center[1] = centroid[1]
    return (outline - centroid) * shape_scale + center

def build_outline(outline_3d: np.ndarray, color, radius: float=0.15) -> o3d.geometry.TriangleMesh:

    ring = np.asarray(outline_3d, dtype=float)
    if len(ring) > 2 and np.linalg.norm(ring[0] - ring[-1]) > 1e-6:
        ring = np.vstack([ring, ring[0]])
    return polyline_tube(ring, radius, color, resolution=6)

def build_scene(map_objects: List[OutlineMapObject], poses: Dict[str, CameraPose],
                show_trajectory: bool=True, show_axes: bool=False):

    geometries = []
    path = _trajectory_points(poses) if len(poses) >= 2 else np.zeros((0, 3))

    if show_trajectory and len(path) >= 2:
        geometries.append(polyline_tube(path, radius=0.55, color=[0.9, 0.05, 0.05]))

    for obj in map_objects:
        color = CLASS_COLORS.get(obj.class_name, DEFAULT_COLOR)
        for outline in obj.outlines:
            if len(outline) < 2:
                continue
            drawn = _spread_for_display(outline, path) if len(path) >= 2 else outline
            geometries.append(build_outline(drawn, color, radius=0.45))

    if show_axes:
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0))

    return geometries

def render_scene(map_objects: List[OutlineMapObject],
                 poses: Dict[str, CameraPose], output_path: str,
                 show_trajectory: bool=True, show_axes: bool=False):

    geometries = build_scene(
        map_objects, poses, show_trajectory=show_trajectory, show_axes=show_axes
    )
    if not geometries:
        print("Warning: scene has no geometry to render.")
        return

    if output_path:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Outline Map", width=1280, height=720, visible=False)
        for g in geometries:
            vis.add_geometry(g)

        _fit_view(
            vis.get_view_control(), geometries,
            zoom=0.5,
            front=[0.35, -0.45, -0.82],
            up=[0.0, -1.0, 0.0],
        )
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