import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Dict

import numpy as np

@dataclass
class CameraPose:
    frame_name: str
    rotation: np.ndarray
    translation: np.ndarray

class ColmapRunner:

    def __init__(self, workspace: str, frames_dir: str, matcher: str="sequential"):
        self.workspace = Path(workspace)
        self.frames_dir = Path(frames_dir)
        self.matcher = matcher
        self.database_path = self.workspace / "database.db"
        self.sparse_dir = self.workspace / "sparse"

        self.workspace.mkdir(parents=True, exist_ok=True)
        self.sparse_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        self._feature_extraction()
        self._feature_matching()
        self._sparse_reconstruction()

    def _feature_extraction(self):
        subprocess.run(["colmap", "feature_extractor",
                        "--database_path", str(self.database_path),
                        "--image_path", str(self.frames_dir),
                        "--ImageReader.single_camera", "1"], check=True)

    def _feature_matching(self):
        matcher_cmd = "sequential_matcher" if self.matcher == "sequential" else "exhaustive_matcher"
        subprocess.run(["colmap", matcher_cmd, 
                        "--database_path", str(self.database_path),
                        "--FeatureMatching.max_num_matches", "8192",
                        "--FeatureMatching.use_gpu", "0"], check=True)

    def _sparse_reconstruction(self):
        subprocess.run([
            "colmap", "mapper",
            "--database_path", str(self.database_path),
            "--image_path", str(self.frames_dir),
            "--output_path", str(self.sparse_dir)
        ], check=True)

    def read_poses(self) -> Dict[str, CameraPose]:
        import pycolmap

        model_path = self.sparse_dir / "0"
        reconstruction = pycolmap.Reconstruction(str(model_path))

        poses = {}
        for image_id, image in reconstruction.images.items():
            cam_from_world = image.cam_from_world()
            rotation = cam_from_world.rotation.matrix()
            translation = cam_from_world.translation

            R_world_from_cam = rotation.T
            t_world_from_cam = -R_world_from_cam @ translation

            poses[image.name] = CameraPose(
                frame_name=image.name,
                rotation=R_world_from_cam,
                translation=t_world_from_cam
            )

        return poses

    def read_intrinsics(self) -> Dict:
        import pycolmap

        model_path = self.sparse_dir / "0"
        reconstruction = pycolmap.Reconstruction(str(model_path))

        camera = next(iter(reconstruction.cameras.values()))
        parameters = camera.params

        return {
            "model": camera.model.name,
            "width": camera.width,
            "height": camera.height,
            "params": list(parameters)
        }

    def read_sparse_correspondences(self) -> Dict[str, np.ndarray]:
        import pycolmap

        model_path = self.sparse_dir / "0"
        reconstruction = pycolmap.Reconstruction(str(model_path))

        correspondences = {}
        for image_id, image in reconstruction.images.items():
            pts = []
            for p2d in image.points2D:
                if p2d.has_point3D():
                    point3D = reconstruction.points3D[p2d.point3D_id].xyz
                    pts.append((p2d.xy[0], p2d.xy[1], np.array(point3D)))

            correspondences[image.name] = pts

        return correspondences