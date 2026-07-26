from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

import cv2
import numpy as np

@dataclass
class CameraPose:
    frame_name: str
    rotation: np.ndarray
    translation: np.ndarray
    num_points3D: int
    mean_reprojection_error: float

class VisualOdometry:

    def __init__(self, frames_dir: str, intrinsics: Optional[Dict]=None,
                 min_matches: int=30, max_features: int=3000):

        self.frames_dir = Path(frames_dir)
        self.min_matches = min_matches
        self.max_features = max_features

        self._intrinsics = intrinsics
        self._poses: Dict[str, CameraPose] = {}
        self._sparse_points: List[list] = []
        self._correspondences: Dict[str, np.ndarray] = {}

        self._orb = cv2.ORB_create(nfeatures=self.max_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._prev_triangulated: Dict[tuple, np.ndarray] = {}

    def _default_intrinsics(self, width: int, height: int) -> Dict:

        f = 1.2 * max(width, height)
        return {"fx": f, "fy": f, "cx": width / 2, "cy": height / 2}

    def run(self):

        frame_paths = sorted(self.frames_dir.glob("*.jpg"))
        if len(frame_paths) < 2:
            raise ValueError("At least two frames are required for visual odometry.")

        first_frame = cv2.imread(str(frame_paths[0]))
        if first_frame is None:
            raise ValueError(f"Failed to read the first frame: {frame_paths[0]}")

        h, w = first_frame.shape[:2]

        if self._intrinsics is None:
            self._intrinsics = self._default_intrinsics(w, h)
            print(f"Using default intrinsics: {self._intrinsics}")

        K = np.array([
            [self._intrinsics["fx"], 0, self._intrinsics["cx"]],
            [0, self._intrinsics["fy"], self._intrinsics["cy"]],
            [0, 0, 1]
        ])

        prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        prev_kp, prev_des = self._orb.detectAndCompute(prev_gray, None)

        R_world = np.eye(3)
        t_world = np.zeros(3)

        self._poses[frame_paths[0].name] = CameraPose(
            frame_name=frame_paths[0].name,
            rotation=R_world.copy(),
            translation=t_world.copy(),
            num_points3D=0,
            mean_reprojection_error=0.0
        )

        num_registered = 1
        num_failed = 0

        for i in range(1, len(frame_paths)):

            frame_name = frame_paths[i].name
            frame = cv2.imread(str(frame_paths[i]))

            if frame is None:
                print(f"Warning: Failed to read frame {frame_name}. Skipping.")
                num_failed += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp, des = self._orb.detectAndCompute(gray, None)

            if des is None or prev_des is None or len(des) < 8 or len(prev_des) < 8:
                print(f"Warning: Not enough features detected in frame {frame_name}. Skipping.")
                num_failed += 1
                continue

            matches = sorted(self._matcher.match(prev_des, des), key=lambda x: x.distance)

            if len(matches) < self.min_matches:
                print(f"Warning: Not enough matches ({len(matches)}) between frames "
                      f"{frame_paths[i-1].name} and {frame_name}. Skipping.")
                num_failed += 1
                continue

            pts_prev = np.array([prev_kp[m.queryIdx].pt for m in matches])
            pts_curr = np.array([kp[m.trainIdx].pt for m in matches])

            E, mask = cv2.findEssentialMat(pts_curr, pts_prev, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

            if E is None:
                print(f"Warning: Essential matrix could not be computed for frame {frame_name}. Skipping.")
                num_failed += 1
                continue

            inlier_count, R_rel, t_rel, mask_pose = cv2.recoverPose(E, pts_curr, pts_prev, K, mask=mask)

            if inlier_count < self.min_matches // 2:
                print(f"Warning: Not enough inliers ({inlier_count}) for frame {frame_name}. Skipping.")
                num_failed += 1
                continue

            inlier_mask = mask_pose.ravel().astype(bool)

            scale = self._estime_relative_scale(
                pts_prev[inlier_mask], pts_curr[inlier_mask], K, R_world, t_world, R_rel, t_rel
            )

            t_world = t_world + R_world @ (t_rel.flatten() * scale)
            R_world = R_world @ R_rel

            self._poses[frame_name] = CameraPose(
                frame_name=frame_name,
                rotation=R_world.copy(),
                translation=t_world.copy(),
                num_points3D=int(inlier_mask.sum()),
                mean_reprojection_error=0.0
            )

            self._triangulate_store(pts_prev[inlier_mask], pts_curr[inlier_mask], 
                                    K, R_world, t_world, R_rel, t_rel, frame_name=frame_name, scale=scale)

            prev_gray, prev_kp, prev_des = gray, kp, des
            num_registered += 1

        print(f"Visual odometry completed. Registered {num_registered} / {len(frame_paths)} frames, Failed: {num_failed}")

    def _estime_relative_scale(self, pts_prev, pts_curr, K, R_world, t_world, R_rel, t_rel, default_scale: float=1.0) -> float:

        P_prev = K @ np.hstack([R_world.T, (-R_world.T @ t_world).reshape(3, 1)])

        R_curr_unit = R_world @ R_rel
        t_curr_unit = t_world + R_world @ t_rel.flatten()
        P_curr_unit = K @ np.hstack([R_curr_unit.T, (-R_curr_unit.T @ t_curr_unit).reshape(3, 1)])

        pts4d = cv2.triangulatePoints(P_prev, P_curr_unit, pts_prev.T, pts_curr.T)
        pts3d_prev = (pts4d[:3] / pts4d[3]).T

        prev_by_key = {
            (round(x, 1), round(y, 1)): pt for (x, y), pt in zip(pts_prev, pts3d_prev)
            if np.isfinite(pt).all()
        }

        shared_keys = set(prev_by_key.keys()) & set(self._prev_triangulated.keys())

        next_by_key = {
            (round(x, 1), round(y, 1)): pt for (x, y), pt in zip(pts_curr, pts3d_prev)
            if np.isfinite(pt).all()
        }

        if len(shared_keys) < 5:

            self._prev_triangulated = next_by_key
            return getattr(self, "_last_scale", default_scale)

        keys = list(shared_keys)
        new_pts = np.array([prev_by_key[k] for k in keys])
        old_pts = np.array([self._prev_triangulated[k] for k in keys])

        new_dists = np.linalg.norm(new_pts[:, None, :] - new_pts[None, :, :], axis=-1)
        old_dists = np.linalg.norm(old_pts[:, None, :] - old_pts[None, :, :], axis=-1)

        iu = np.triu_indices(len(keys), k=1)
        valid = old_dists[iu] > 1e-6
        if valid.sum() < 3:

            self._prev_triangulated = next_by_key
            return getattr(self, "_last_scale", default_scale)

        ratios = old_dists[iu][valid] / new_dists[iu][valid]
        scale = float(np.median(ratios))

        self._prev_triangulated = next_by_key
        self._last_scale = scale
        return scale

    def _triangulate_store(self, pts_prev, pts_curr, K, R_world, t_world, R_rel, t_rel, frame_name, scale):

        R_prev_world = R_world @ R_rel.T
        t_prev_world = t_world - R_prev_world @ (t_rel.flatten() * scale)

        P_prev = K @ np.hstack([R_prev_world.T, (-R_prev_world.T @ t_prev_world).reshape(3, 1)])
        P_curr = K @ np.hstack([R_world.T, (-R_world.T @ t_world).reshape(3, 1)])

        pts4d = cv2.triangulatePoints(P_prev, P_curr, pts_prev.T, pts_curr.T)
        pts3d = (pts4d[:3] / pts4d[3]).T

        valid = np.isfinite(pts3d).all(axis=1) & (np.linalg.norm(pts3d, axis=1) < 500)
        pts3d = pts3d[valid]
        pts_curr_valid = pts_curr[valid]

        self._sparse_points.extend(pts3d.tolist())
        self._correspondences[frame_name] = [
            (x, y, pt) for (x, y), pt in zip(pts_curr_valid, pts3d)
        ]

    def read_poses(self) -> Dict[str, CameraPose]:
        return self._poses

    def read_intrinsics(self) -> Dict:
        return {
            "model": "PINHOLE",
            "params": [
                self._intrinsics["fx"],
                self._intrinsics["fy"],
                self._intrinsics["cx"],
                self._intrinsics["cy"]
            ]
        }

    def read_sparse_points(self) -> np.ndarray:
        return np.array(self._sparse_points) if self._sparse_points else np.empty((0, 3))

    def read_sparse_correspondences(self) -> Dict[str, np.ndarray]:
        return self._correspondences