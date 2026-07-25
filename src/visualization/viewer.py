from typing import List, Dict

import numpy as np
import open3d as o3d

from src.mapping.projector import MapObject
from src.slam.colmap_runner import CameraPose

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
DEFAULT_COLOR = [0.5, 0.5, 0.5]

def object_point(
    map_objects: List[MapObject], 
    point_size: float=0.3
) -> o3d.geometry.PointCloud:

    points = []
    colors = []

    for obj in map_objects:
        points.append(obj.positions_3d)
        colors.append(CLASS_COLORS.get(obj.class_name, DEFAULT_COLOR))

    pcd = o3d.geometry.PointCloud()

    if points:
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    else:
        pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        pcd.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))

    return pcd

def camera_trajectory(
    poses: Dict[str, CameraPose]
) -> o3d.geometry.LineSet:

    sorted_poses = sorted(poses.values(), key=lambda p: p.frame_name)
    points = np.array([p.translation for p in sorted_poses])

    lines = [[i, i+1] for i in range(len(points)-1)]
    colors = [[0.8, 0.8, 0.8] for _ in lines]

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

def render_map(map_objects: List[MapObject], poses: Dict[str, CameraPose],
               show_trajectory: bool=True, show_axes: bool=True, point_size: float=8.0, output_path: str = None):

    geometries = []

    object_cloud = object_point(map_objects)
    geometries.append(object_cloud)

    if show_trajectory and poses:
        geometries.append(camera_trajectory(poses))

    if show_axes:
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
        geometries.append(axes)

    if output_path:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="3D Map", width=1280, height=720, visible=False)
        for g in geometries:
            vis.add_geometry(g)

        render = vis.get_render_option()
        render.point_size = point_size
        render.background_color = np.array([1, 1, 1])

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
        _fit_view(vis.get_view_control(), geometries)

        vis.run()
        vis.destroy_window()

def save_map(map_objects: List[MapObject], poses: Dict[str, CameraPose], output_path: str):

    object_cloud = object_point(map_objects)
    o3d.io.write_point_cloud(output_path, object_cloud)