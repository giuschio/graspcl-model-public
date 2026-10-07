import time

import numpy as np

from grasp_learning_cl.grasp_heuristics.utils import thin_pointcloud
from grasp_learning_cl.grasp_classification.collision_checker import CollisionChecker


class VGNGraspSampler:
    """Sampler that returns random normal-aligned grasps as a bounded list."""

    def __init__(self) -> None:
        self.rng = np.random.default_rng(None)
        self.collision_checker = CollisionChecker()

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
        if points.shape[0] == 0:
            return []

        if not cloud_objects.has_normals():
            raise ValueError("Please estimate cloud normals before putting it in the sampler")
        normals = np.asarray(cloud_objects.normals, dtype=np.float32)

        grasps: list[dict] = []
        max_attempts = max(20 * int(n_grasps), 100)

        for attempt_n in range(max_attempts):
            if len(grasps) >= n_grasps:
                break

            idx = int(self.rng.choice(len(points)))
            point = points[idx]
            normal = self._normalize(normals[idx])
            if normal is None:
                continue

            z_axis = -normal
            x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            if np.isclose(abs(float(np.dot(x_axis, z_axis))), 1.0, atol=1e-4):
                x_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)

            y_axis = self._normalize(np.cross(z_axis, x_axis))
            if y_axis is None:
                continue
            x_axis = self._normalize(np.cross(y_axis, z_axis))
            if x_axis is None:
                continue

            base_rotation = np.column_stack((x_axis, y_axis, z_axis)).astype(np.float32)

            yaw = float(self.rng.uniform(0.0, np.pi))
            rot_z = np.array(
                [[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            rotation = (base_rotation @ rot_z).astype(np.float32)

            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = rotation
            pose[:3, 3] = point.astype(np.float32)
            grasp = self._format_grasp(pose)
            if cloud_collisions_ds is None or self.collision_checker.is_collision_free(cloud_collisions_ds, grasp):
                grasps.append(grasp)

        if debug:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            hit_rate = len(grasps) / attempt_n
            print(f"{self.__class__.__name__}: hit rate {len(grasps)}/{attempt_n} ({hit_rate:.1%}), time {elapsed_ms:.2f} ms")

        return grasps

    @staticmethod
    def _format_grasp(pose: np.ndarray) -> dict:
        center = pose[:3, 3]
        y_axis = pose[:3, 1]
        point1 = center + 0.04 * y_axis
        point2 = center - 0.04 * y_axis
        return {
            "pose": pose.astype(np.float32),
            "width": 0.08,
            "point1": point1.astype(np.float32),
            "point2": point2.astype(np.float32),
            "sampler": "vgn"
        }

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray | None:
        norm = float(np.linalg.norm(vector))
        if norm < 1e-8:
            return None
        return (vector / norm).astype(np.float32)
