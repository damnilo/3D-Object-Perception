from typing import List, Dict, Optional

import numpy as np
import open3d as o3d
import cv2

from src.mapping.projector import MapObject
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

def env_point_cloud(sparse_points: Optional[np.ndarray]) -> o3d.geometry.PointCloud:

    pcd = o3d.geometry.PointCloud()

    if sparse_points is not None and len(sparse_points) > 0:
        pcd.points = o3d.utility.Vector3dVector(sparse_points)
        pcd.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([[0.5, 0.5, 0.5]]), (len(sparse_points), 1))
        )

    return pcd

def obj_bbox(map_objects: List[MapObject], min_size: float=0.5) -> List[o3d.geometry.LineSet]:

    edges = [
        [0,1], [1,2], [2,3], [3,0],
        [4,5], [5,6], [6,7], [7,4],
        [0,4], [1,5], [2,6], [3,7]
    ]

    boxes = []

    for obj in map_objects:
        pts = np.atleast_2d(obj.points_3d)
        min, max = pts.min(axis=0), pts.max(axis=0)
        pad = np.maximum(min_size - (max - min), 0) / 2

        min, max = min - pad, max + pad

        corners = np.array([
            [min[0], min[1], min[2]], [max[0], min[1], min[2]],
            [max[0], max[1], min[2]], [min[0], max[1], min[2]],
            [min[0], min[1], max[2]], [max[0], min[1], max[2]],
            [max[0], max[1], max[2]], [min[0], max[1], max[2]]
        ])

        color = CLASS_COLORS.get(obj.class_name, DEFAULT_COLOR)

        box = o3d.geometry.LineSet()
        box.points = o3d.utility.Vector3dVector(corners)
        box.lines = o3d.utility.Vector2iVector(edges)
        box.colors = o3d.utility.Vector3dVector([color for _ in edges])
        boxes.append(box)

    return boxes

def ground_grid(center: np.ndarray, extent: float=40.0, spacing: float=4.0) -> o3d.geometry.LineSet:

    points, lines = [], []
    n = int(extent / spacing)

    idx = 0
    for i in range(-n, n + 1):
        x = center[0] + i * spacing
        points += [x, center[1], center[2] - extent], [x, center[1], center[2] + extent]
        lines.append([idx, idx + 1]); idx += 2

    for i in range(-n, n + 1):
        z = center[2] + i * spacing
        points += [center[0] - extent, center[1], z], [center[0] + extent, center[1], z]
        lines.append([idx, idx + 1]); idx += 2

    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(np.array(points))
    grid.lines = o3d.utility.Vector2iVector(np.array(lines))
    grid.colors = o3d.utility.Vector3dVector([[0.7, 0.7, 0.7] for _ in lines])
    return grid

def camera_trajectory(
    poses: Dict[str, CameraPose]
) -> o3d.geometry.LineSet:

    sorted_poses = sorted(poses.values(), key=lambda p: p.frame_name)
    points = np.array([p.translation for p in sorted_poses])

    lines = [[i, i+1] for i in range(len(points)-1)]
    colors = [[0.9, 0.9, 0.2] for _ in lines]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    return line_set

def _fit_view(view_control, geometries):

    bbox = geometries[0].get_axis_aligned_bounding_box()
    for g in geometries[1:]:
        bbox += g.get_axis_aligned_bounding_box()

    view_control.set_lookat(bbox.get_center())
    view_control.set_front([0.3, -0.5, -0.8])
    view_control.set_up([0, -1, 0])
    view_control.set_zoom(0.7)

def _build_scene(map_objects, poses, environment_points, show_trajectory, show_grid, show_axes):

    geometries = [env_point_cloud(environment_points)]
    geometries.extend(obj_bbox(map_objects))

    if show_trajectory:
        geometries.append(camera_trajectory(poses))

    if show_grid and poses:
        center = np.mean([p.translation for p in poses.values()], axis=0)
        geometries.append(ground_grid(center))

    if show_axes:
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
        geometries.append(axes)

    return geometries

def render_map(map_objects: List[MapObject], poses: Dict[str, CameraPose],
               environment_points: Optional[np.ndarray] = None,
               show_trajectory: bool=True, show_axes: bool=True, show_grid: bool=True,
               point_size: float=8.0, output_path: str = None):

    geometries = _build_scene(map_objects, poses, environment_points, show_trajectory, show_grid, show_axes=show_axes)

    if output_path:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="3D Map", width=1280, height=720, visible=False)
        for g in geometries:
            vis.add_geometry(g)

        render = vis.get_render_option()
        render.point_size = point_size
        render.background_color = np.array([0, 0, 0])

        _fit_view(vis.get_view_control(), geometries)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(output_path)
        vis.destroy_window()
    else:
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="3D Map", width=1280, height=720)
        for g in geometries:
            vis.add_geometry(g)

        vis.get_render_option().point_size = point_size
        vis.get_render_option().background_color = np.array([0, 0, 0])
        _fit_view(vis.get_view_control(), geometries)
        vis.run()
        vis.destroy_window()

def _slerp(r0, r1, t):

    from scipy.spatial.transform import Rotation, Slerp

    key_rots = Rotation.from_matrix([r0, r1])
    slerp = Slerp([0, 1], key_rots)

    return slerp([t])[0].as_matrix()

def render_flythrough(map_objects: List[MapObject], poses: Dict[str, CameraPose],
                      env_points: Optional[np.ndarray], output_path: str,
                       fps: int=5, width: int=1280, height: int=720, point_size: float=8.0,
                       steps_per_pose: int=8):

    sorted_poses = sorted(poses.values(), key=lambda p: p.frame_name)
    if len(sorted_poses) < 2:
        raise ValueError("At least two poses are required for flythrough rendering.")

    frame_order = {p.frame_name: idx for idx, p in enumerate(sorted_poses)}

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Map Flythrough", width=width, height=height, visible=False)

    grid = ground_grid(np.mean([p.translation for p in sorted_poses], axis=0))
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    vis.add_geometry(grid)
    vis.add_geometry(axes)

    env_cloud = o3d.geometry.PointCloud()
    vis.add_geometry(env_cloud)

    box_geoms = obj_bbox(map_objects)
    box_reveal_frame = []

    for obj in map_objects:
        earliest = min(frame_order.get(f, float('inf')) for f in obj.frame_names)
        box_reveal_frame.append(earliest)

    for box in box_geoms:
        vis.add_geometry(box)

    render = vis.get_render_option()
    render.point_size = point_size
    render.background_color = np.array([0, 0, 0])
    ctr = vis.get_view_control()

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    revealed_points = []

    for i in range(len(sorted_poses) - 1):
        p0, p1 = sorted_poses[i], sorted_poses[i + 1]

        if env_points is not None and len(env_points) > 0:
            chunk = len(env_points) // len(sorted_poses)
            revealed_points.extend(env_points[i * chunk:(i + 1) * chunk])
            env_cloud.points = o3d.utility.Vector3dVector(np.array(revealed_points))
            env_cloud.colors = o3d.utility.Vector3dVector(
                np.tile(np.array([[0.5, 0.5, 0.5]]), (len(revealed_points), 1))
            )
            vis.update_geometry(env_cloud)

        for box, reveal_at in zip(box_geoms, box_reveal_frame):
            box.paint_uniform_color(box.colors[0] if i >= reveal_at else [0, 0, 0])
            vis.update_geometry(box)

        for step in range(steps_per_pose):
            t = step / steps_per_pose
            pos = (1 - t) * p0.translation + t * p1.translation
            rot = _slerp(p0.rotation, p1.rotation, t)

            ctr.set_lookat(pos + rot[:, 2] * 5.0)
            ctr.set_front(-rot[:, 2])
            ctr.set_up(rot[:, 1])
            ctr.set_zoom(0.5)

            vis.poll_events()
            vis.update_renderer()
            img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            frame_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

    writer.release()
    vis.destroy_window()

def save_map(env_points: Optional[np.ndarray], output_path: str):

    object_cloud = env_point_cloud(env_points)
    o3d.io.write_point_cloud(output_path, object_cloud)