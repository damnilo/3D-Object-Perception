from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.sparse import eye as speye, vstack
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

        self._tracks: Dict[int, Dict] = {}
        self._next_track_id = 0
        self._active_tracks: Dict[int, int] = {}
        self._ba_window_size = 8
        self._ba_every_n_frames = 3
        self._frames_since_ba = 0
        self._ba_optimizable_tail = 4
        self._ba_warm_start_poses: Dict[str, np.ndarray] = {}
        self._ba_warm_start_points: Dict[int, np.ndarray] = {}

    def _pose_to_vec(self, R: np.ndarray, t: np.ndarray) -> np.ndarray:

        rvec, _ = cv2.Rodrigues(R)
        return np.hstack([rvec.flatten(), t.flatten()])

    def _vec_to_pose(self, vec: np.ndarray):

        rvec = vec[:3].reshape(3, 1)
        t = vec[3:6]
        R, _ = cv2.Rodrigues(rvec)
        return R, t

    def _build_sparsity(self, n_obs, is_opt, obs_track_i, n_pose_params, n_point_params):

        n_params = n_pose_params + n_point_params
        sparsity = lil_matrix((n_obs * 2, n_params), dtype=int)
        opt_positions = np.where(is_opt)[0]

        opt_frame_slot = 0
        running_opt_idx = []
        opt_frame_idx = np.cumsum(is_opt) - 1

        for obs_i in range(n_obs):

            tid_i = obs_track_i[obs_i]
            pt_off = n_pose_params + tid_i * 3
            sparsity[obs_i * 2: obs_i * 2 + 2, pt_off: pt_off + 3] = 1

            if is_opt[obs_i]:

                sparsity[obs_i * 2: obs_i * 2 + 2, 0:n_pose_params] = 1

        return sparsity

    def _update_tracks(self, prev_kp, kp, matches, inlier_mask,
                       prev_frame_name, curr_frame_name, inlier_world_pts, valid_mask):

        inlier_matches_all = [m for m, keep in zip(matches, inlier_mask) if keep]
        inlier_matches = [m for m, keep in zip(inlier_matches_all, valid_mask) if keep]

        new_active_tracks = {}

        for m, world_pt in zip(inlier_matches, inlier_world_pts):

            prev_idx, curr_idx = m.queryIdx, m.trainIdx
            x_curr, y_curr = kp[curr_idx].pt

            if prev_idx in self._active_tracks:

                track_id = self._active_tracks[prev_idx]
                self._tracks[track_id]['obs'][curr_frame_name] = (x_curr, y_curr)
            else:

                track_id = self._next_track_id
                self._next_track_id += 1
                x_prev, y_prev = prev_kp[prev_idx].pt
                self._tracks[track_id] = {
                    'point_3d': world_pt,
                    'obs': {
                        prev_frame_name: (x_prev, y_prev),
                        curr_frame_name: (x_curr, y_curr)
                    }
                }

            new_active_tracks[curr_idx] = track_id

        self._active_tracks = new_active_tracks

    def _run_windowed_ba(self, K):

        frame_order = list(self._poses.keys())
        window = frame_order[-self._ba_window_size:]
        if len(window) < 3:
            return

        window_set = set(window)
        anchor_frame = window[0]

        tail_start = max(1, len(window) - self._ba_optimizable_tail)
        optimizable_frames = window[tail_start:]
        fixed_frames = window[1:tail_start]
        opt_frame_idx = {name: i for i, name in enumerate(optimizable_frames)}

        ba_tracks = {
            tid: tr for tid, tr in self._tracks.items()
            if tr['point_3d'] is not None
            and sum(1 for f in tr['obs'] if f in window_set) >= 2
        }

        if len(ba_tracks) < 10:
            return

        max_tracks = 60
        if len(ba_tracks) > max_tracks:

            ranked = sorted(ba_tracks.items(), 
                            key=lambda kv: sum(1 for f in kv[1]['obs'] if f in window_set),
                            reverse=True)
            ba_tracks = dict(ranked[:max_tracks])

        track_ids = list(ba_tracks.keys())
        track_idx = {tid: i for i, tid in enumerate(track_ids)}
        n_pose_params = len(optimizable_frames) * 6
        n_track_params = len(track_ids) * 3

        x0 = np.empty(n_pose_params + n_track_params)
        for name, i in opt_frame_idx.items():

            if name in self._ba_warm_start_poses:

                x0[i * 6:(i + 1) * 6] = self._ba_warm_start_poses[name]
            else:

                pose = self._poses[name]
                x0[i * 6:(i + 1) * 6] = self._pose_to_vec(pose.rotation, pose.translation)

        for tid, i in track_idx.items():

            off = n_pose_params + i * 3
            if tid in self._ba_warm_start_points:

                x0[off:off + 3] = self._ba_warm_start_points[tid]
            else:

                x0[off:off + 3] = ba_tracks[tid]['point_3d']

        pose_prior_weight = 50.0
        x0_pose_only = x0[:n_pose_params].copy()

        fixed_R, fixed_t = {}, {}
        for name in [anchor_frame] + fixed_frames:

            p = self._poses[name]
            fixed_R[name], fixed_t[name] = p.rotation, p.translation

        obs_frame_name, obs_track_id, obs_y = [], [], []
        for td in track_ids:

            for f, (x, y) in ba_tracks[td]['obs'].items():

                if f in window_set:

                    obs_frame_name.append(f)
                    obs_track_id.append(track_idx[td])
                    obs_y.append((x, y))

        obs_track_i = np.array(obs_track_id, dtype=int)
        obs_xy = np.array(obs_y, dtype=float)
        n_obs = len(obs_xy)

        is_opt = np.array([f in opt_frame_idx for f in obs_frame_name], dtype=bool)
        opt_obs_frame_i = np.array([opt_frame_idx[f] for f, o in zip(obs_frame_name, is_opt) if o], dtype=int)
        fixed_obs_R = np.array([fixed_R[f] for f, o in zip(obs_frame_name, is_opt) if not o], dtype=float)
        fixed_obs_t = np.array([fixed_t[f] for f, o in zip(obs_frame_name, is_opt) if not o], dtype=float)

        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        last_valid_mask = np.ones(n_obs * 2, dtype=bool)

        def _project(R_batch, t_batch, pts_batch):

            cam = np.einsum('ikj,ik->ij', R_batch, pts_batch - t_batch)
            z = cam[:, 2]
            safe = z > 1e-3
            u = np.where(safe, fx * cam[:, 0] / np.where(safe, z, 1) + cx, 0)
            v = np.where(safe, fy * cam[:, 1] / np.where(safe, z, 1) + cy, 0)

            return u, v, safe

        def residuals(x):

            pose_vecs = x[:n_pose_params].reshape(-1, 6)
            pts = x[n_pose_params:].reshape(-1, 3)

            opt_R, opt_t = np.empty((len(optimizable_frames), 3, 3)), np.empty((len(optimizable_frames), 3))
            for i in range(len(optimizable_frames)):

                R_i, t_i = self._vec_to_pose(pose_vecs[i])
                opt_R[i], opt_t[i] = R_i, t_i

            res = np.zeros(n_obs * 2)
            valid_mask = np.zeros(n_obs * 2, dtype=bool)

            if opt_obs_frame_i.size > 0:

                opt_mask = is_opt
                pts_for_opt = pts[obs_track_i[opt_mask]]
                u, v, safe = _project(opt_R[opt_obs_frame_i], opt_t[opt_obs_frame_i], pts_for_opt)
                xy_obs = obs_xy[opt_mask]
                ru = np.where(safe, u - xy_obs[:, 0], 0)
                rv = np.where(safe, v - xy_obs[:, 1], 0)
                idx = np.where(opt_mask)[0]
                res[idx * 2] = ru
                res[idx * 2 + 1] = rv
                valid_mask[idx * 2] = safe
                valid_mask[idx * 2 + 1] = safe

            if fixed_obs_R.shape[0] > 0:

                fixed_mask = ~is_opt
                pts_for_fixed = pts[obs_track_i[fixed_mask]]
                u, v, safe = _project(fixed_obs_R, fixed_obs_t, pts_for_fixed)
                xy_obs = obs_xy[fixed_mask]
                ru = np.where(safe, u - xy_obs[:, 0], 0)
                rv = np.where(safe, v - xy_obs[:, 1], 0)
                idx = np.where(fixed_mask)[0]
                res[idx * 2] = ru
                res[idx * 2 + 1] = rv
                valid_mask[idx * 2] = safe
                valid_mask[idx * 2 + 1] = safe

            nonlocal last_valid_mask
            last_valid_mask = valid_mask

            pose_params = x[:n_pose_params]
            prior_res = pose_prior_weight * (pose_params - x0_pose_only)

            return np.concatenate([res, prior_res])

        sparsity = self._build_sparsity(n_obs, is_opt, obs_track_i, n_pose_params, n_track_params)

        prior_sparsity = speye(n_pose_params, n_pose_params + n_track_params, format='lil')
        sparsity = vstack([sparsity, prior_sparsity]).tolil()

        result = least_squares(residuals, x0, method='trf', loss='huber', f_scale=2.0, max_nfev=50, jac_sparsity=sparsity)
        refined = result.x
        final_res = result.fun

        reprojection_res = final_res[:n_obs * 2]
        valid_mask = reprojection_res[last_valid_mask]
        mean_err = float(np.sqrt(np.mean(valid_mask ** 2))) if len(valid_mask) > 0 else 0.0

        for name, i in opt_frame_idx.items():

            R_new, t_new = self._vec_to_pose(refined[i * 6:(i + 1) * 6])
            old = self._poses[name]

            self._poses[name] = CameraPose(
                frame_name=name,
                rotation=R_new,
                translation=t_new,
                num_points3D=old.num_points3D,
                mean_reprojection_error=mean_err
            )

        for tid, i in track_idx.items():

            off = n_pose_params + i * 3
            refined_pt = refined[off:off + 3]
            ba_tracks[tid]['point_3d'] = refined_pt
            self._ba_warm_start_points[tid] = refined_pt

        current_window_set = set(window)
        self._ba_warm_start_poses = {
            k: v for k, v in self._ba_warm_start_poses.items() if k in current_window_set
        }
        self._ba_warm_start_points = {
            tid: pt for tid, pt in self._ba_warm_start_points.items() if tid in track_idx
        }

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

            rel_angle_deg = np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1.0, 1.0)))
            max_rel_angle = 25.0

            if rel_angle_deg > max_rel_angle:
                print(f"Warning: Relative rotation ({rel_angle_deg:.2f} deg) exceeds "
                      f"threshold ({max_rel_angle} deg) for frame {frame_name}. Skipping.")
                num_failed += 1
                continue

            inlier_mask = mask_pose.ravel().astype(bool)

            scale = self._estimate_relative_scale(
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

            world_points, valid_mask = self._triangulate_store(pts_prev[inlier_mask], pts_curr[inlier_mask], 
                                    K, R_world, t_world, R_rel, t_rel, frame_name=frame_name, scale=scale)

            self._update_tracks(
                prev_kp, kp, matches, inlier_mask,
                prev_frame_name=frame_paths[i-1].name,
                curr_frame_name=frame_name,
                inlier_world_pts=world_points,
                valid_mask=valid_mask
            )

            self._frames_since_ba += 1
            if self._frames_since_ba >= self._ba_every_n_frames:

                self._run_windowed_ba(K)
                self._frames_since_ba = 0

                latest_pose = self._poses[frame_name]
                R_world, t_world = latest_pose.rotation.copy(), latest_pose.translation.copy()

            prev_gray, prev_kp, prev_des = gray, kp, des
            num_registered += 1

        print(f"Visual odometry completed. Registered {num_registered} / {len(frame_paths)} frames, Failed: {num_failed}")

    def _estimate_relative_scale(self, pts_prev, pts_curr, K, R_world, t_world, R_rel, t_rel, default_scale: float=1.0) -> float:

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

        return pts3d, valid

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