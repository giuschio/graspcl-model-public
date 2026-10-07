import time

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from grasp_learning_cl.grasp_heuristics.utils import thin_pointcloud
from grasp_learning_cl.grasp_classification.collision_checker import CollisionChecker


class LowQualitySampler:
    """Sampler that synthesizes low-quality grasp poses from nearby point pairs."""
    max_grasp_width = 0.08
    min_pair_distance = 0.003
    translation_noise = 0.008
    rotation_noise = 1.0
    grasp_noise = {
        "t_x": translation_noise,
        "t_y": translation_noise,
        "t_z": translation_noise,
        "r_x": rotation_noise,
        "r_y": rotation_noise,
        "r_z": rotation_noise,
    }

    def __init__(
        self,
    ) -> None:
        self.rng = np.random.default_rng(None)
        self.collision_checker = CollisionChecker()

    @torch.no_grad()
    def __call__(self, cloud_objects, cloud_collisions: None, n_grasps: int, debug: bool = False) -> list[dict]:
        """Sample grasps from object geometry and optionally reject colliding poses.

        Args:
            cloud_objects: Object-focused point cloud used to generate grasp proposals (e.g. without the table)
            cloud_collisions: Full scene point cloud used for collision checking (full scene cloud, including objects and obstacles). If ``None``, no collision checking is performed inside the sampler.
            n_grasps: Target number of grasps to return.formed inside the sampler.
        """
        start_time = time.perf_counter()
        if cloud_objects is None:
            return []

        cloud_collisions_ds = None if cloud_collisions is None else self.collision_checker.downsample_pointcloud(cloud_collisions)

        points = np.asarray(cloud_objects.points, dtype=np.float32)
        num_points = points.shape[0]
        if num_points < 2:
            return []

        tree = cKDTree(points)
        target_grasps = int(n_grasps)
        grasps: list[dict] = []
        max_attempts = max(20 * target_grasps, 100)

        for attempt_n in range(max_attempts):
            if len(grasps) >= target_grasps:
                break

            p1_idx = int(self.rng.integers(num_points))
            p1 = points[p1_idx]

            neighbor_idx = tree.query_ball_point(p1, r=self.max_grasp_width)
            neighbor_idx = [idx for idx in neighbor_idx if idx != p1_idx]
            if not neighbor_idx:
                continue

            p2 = points[int(self.rng.choice(neighbor_idx))]
            segment = p2 - p1
            width = float(np.linalg.norm(segment))
            if width < self.min_pair_distance:
                continue

            y_axis = segment / width
            rotation = self._sample_frame(y_axis)
            center = 0.5 * (p1 + p2)

            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = rotation
            pose[:3, 3] = center
            pose = self._apply_local_noise(pose)

            center_noisy = pose[:3, 3]
            y_axis_noisy = pose[:3, 1]
            point1 = center_noisy + 0.04 * y_axis_noisy
            point2 = center_noisy - 0.04 * y_axis_noisy

            grasp = {
                "pose": pose.astype(np.float32),
                "width": 0.08,
                "point1": point1.astype(np.float32),
                "point2": point2.astype(np.float32),
                "sampler": "lowquality"
            }
            if cloud_collisions_ds is None or self.collision_checker.is_collision_free(cloud_collisions_ds, grasp):
                grasps.append(grasp)

        if debug:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            hit_rate = len(grasps) / attempt_n
            print(f"{self.__class__.__name__}: hit rate {len(grasps)}/{attempt_n} ({hit_rate:.1%}), time {elapsed_ms:.2f} ms")

        return grasps

    def _apply_local_noise(self, pose: np.ndarray) -> np.ndarray:
        trans_noise_local = np.array(
            [
                self.rng.normal(0.0, self.grasp_noise["t_x"]),
                self.rng.normal(0.0, self.grasp_noise["t_y"]),
                self.rng.normal(0.0, self.grasp_noise["t_z"]),
            ],
            dtype=np.float32,
        )
        rot_noise = Rotation.from_euler(
            "xyz",
            [
                self.rng.normal(0.0, self.grasp_noise["r_x"]),
                self.rng.normal(0.0, self.grasp_noise["r_y"]),
                self.rng.normal(0.0, self.grasp_noise["r_z"]),
            ],
            degrees=True,
        ).as_matrix().astype(np.float32)

        noisy_pose = pose.copy()
        noisy_pose[:3, :3] = noisy_pose[:3, :3] @ rot_noise
        noisy_pose[:3, 3] = noisy_pose[:3, 3] + noisy_pose[:3, :3] @ trans_noise_local
        return noisy_pose

    def _sample_frame(self, y_axis: np.ndarray) -> np.ndarray:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(np.dot(y_axis, up)) > 0.95:
            up = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        z_axis = up - np.dot(up, y_axis) * y_axis
        z_axis /= np.linalg.norm(z_axis)
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)

        roll_deg = float(self.rng.uniform(0.0, 360.0))
        roll = Rotation.from_rotvec(np.deg2rad(roll_deg) * y_axis).as_matrix().astype(np.float32)
        base_rotation = np.column_stack((x_axis, y_axis, z_axis)).astype(np.float32)
        return (base_rotation @ roll).astype(np.float32)
