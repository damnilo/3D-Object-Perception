import numpy as np

from src.mapping.projector import backproject_build, collect_sightings, ObjectSighting
from src.slam.colmap_runner import CameraPose

def test_backproject():
    intrinsics = {'fx': 500, 'fy': 500, 'cx': 320, 'cy': 240}
    pose = CameraPose(
        frame_name="frame_001",
        rotation=np.eye(3),
        translation=np.array([0, 0, 0])
    )

    point = backproject_build((320, 240), 10, intrinsics, pose)

    assert np.allclose(point, [0, 0, 10], atol=1e-5), f"Expected [0, 0, 10], got {point}"

def test_cluster_sightings():

    sightings = [
        ObjectSighting(class_name="stop_sign", confidence=0.9, position_3d=np.array([0, 0, 10]), frame_name="frame_001"),
        ObjectSighting(class_name="stop_sign", confidence=0.85, position_3d=np.array([0.2, 0.1, 10.1]), frame_name="frame_002"),
        ObjectSighting(class_name="car", confidence=0.95, position_3d=np.array([5, 0, 20]), frame_name="frame_003"),
    ]

    map_objects = collect_sightings(sightings, eps_meters=1.5, min_samples=2)

    stop_signs = [obj for obj in map_objects if obj.class_name == "stop_sign"]
    car = [obj for obj in map_objects if obj.class_name == "car"]

    assert len(stop_signs) == 1
    assert stop_signs[0].num_sightings == 2
    assert len(car) == 0