import time

import numpy as np

from grasp_learning_cl.grasp_classification.collision_checker import CollisionChecker
from grasp_learning_cl.grasp_heuristics.top_down_grasp_sampler import filter_top_layer_vectorized


class TopDownEdgeGraspSampler:
    """Top-down edge sampler that uses one visible contact point and its normal."""

    width = 0.08
    half_width = width / 2.0
    safety_factor = 0.01
    table_height = 0.054
    table_mask = "autodetect"
    z_axis_orthogonality_tolerance_deg = 45.0

    def __init__(self) -> None:
        self.rng = np.random.default_rng(None)
        self.collision_checker = CollisionChecker()

    def __call__(self, cloud_objects, cloud_collisions: None, n_grasps: int, debug: bool = False) -> list[dict]:
        start_time = time.perf_counter()
        if cloud_objects is None:
            return []

        cloud_collisions_ds = None if cloud_collisions is None else self.collision_checker.downsample_pointcloud(cloud_collisions)

        points = np.asarray(cloud_objects.points, dtype=np.float32)
        if points.shape[0] < 1:
            return []

        if not cloud_objects.has_normals():
            raise ValueError("Please estimate cloud normals before putting it in the sampler")
        normals = np.asarray(cloud_objects.normals, dtype=np.float32)

        table_height = float(points[:, 2].min() + 1e-3)
        mask = points[:, 2] > table_height
        points = points[mask]
        normals = normals[mask]
        if points.shape[0] < 1:
            return []

        top_mask = filter_top_layer_vectorized(points, 0.05, 0.025)
        points = points[top_mask]
        normals = normals[top_mask]
        if points.shape[0] < 1:
            return []

        orthogonal_mask = self._filter_normals_orthogonal_to_z(normals)
        points = points[orthogonal_mask]
        normals = normals[orthogonal_mask]
        if points.shape[0] < 1:
            return []

        grasps: list[dict] = []
        active_table_height = self._resolve_table_height(points)

        max_attempts = max(20 * int(n_grasps), 100)

        for attempt_n in range(max_attempts):
            if len(grasps) >= n_grasps:
                break

            grasp = self._sample_single_grasp(points, normals, active_table_height)
            if grasp is not None and (cloud_collisions_ds is None or self.collision_checker.is_collision_free(cloud_collisions_ds, grasp)):
                grasps.append(grasp)

        if debug:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            hit_rate = len(grasps) / attempt_n
            print(f"{self.__class__.__name__}: hit rate {len(grasps)}/{attempt_n} ({hit_rate:.1%}), time {elapsed_ms:.2f} ms")

        return grasps

    def _sample_single_grasp(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        table_height: float,
    ) -> dict | None:
        contact_idx = int(self.rng.integers(points.shape[0]))
        contact_point = points[contact_idx]
        y_axis = -1.0 * self._normalize(normals[contact_idx])
        if y_axis is None:
            return None

        z_axis = np.array((0.0, 0.0, -1.0), dtype=np.float32)
        x_axis = self._normalize(np.cross(y_axis, z_axis))
        if x_axis is None:
            return None
        y_axis = self._normalize(np.cross(z_axis, x_axis))
        if y_axis is None:
            return None

        contact_point = contact_point - y_axis * self.safety_factor
        grasp_center = contact_point + y_axis * self.half_width

        pose = np.eye(4, dtype=np.float32)
        pose[:3, 0] = x_axis
        pose[:3, 1] = y_axis
        pose[:3, 2] = z_axis
        pose[:3, 3] = grasp_center.astype(np.float32)

        pose = pose.copy()
        return self._format_grasp(pose, contact_point, y_axis)

    def _format_grasp(self, pose: np.ndarray, contact_point: np.ndarray, y_axis: np.ndarray) -> dict:
        point1 = np.asarray(contact_point, dtype=np.float32)
        point2 = point1 + self.width * y_axis.astype(np.float32)
        return {
            "pose": pose.astype(np.float32),
            "width": self.width,
            "point1": point1,
            "point2": point2.astype(np.float32),
            "sampler": "topdownedge",
        }

    def _resolve_table_height(self, points: np.ndarray) -> float:
        if self.table_mask == "autodetect":
            return float(points[:, 2].min())
        return self.table_height

    @classmethod
    def _filter_normals_orthogonal_to_z(cls, normals: np.ndarray) -> np.ndarray:
        z_axis = np.array((0.0, 0.0, 1.0), dtype=np.float32)
        normal_norms = np.linalg.norm(normals, axis=1)
        valid_normals = normal_norms > 1e-8
        normalized_normals = np.zeros_like(normals, dtype=np.float32)
        normalized_normals[valid_normals] = normals[valid_normals] / normal_norms[valid_normals, None]

        max_abs_z_alignment = float(np.cos(np.deg2rad(90.0 - cls.z_axis_orthogonality_tolerance_deg)))
        return valid_normals & (np.abs(normalized_normals @ z_axis) <= max_abs_z_alignment)

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray | None:
        norm = float(np.linalg.norm(vector))
        if norm < 1e-8:
            return None
        return vector / norm


TopDownGraspSampler = TopDownEdgeGraspSampler
