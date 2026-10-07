import time

import numpy as np
from scipy.spatial import cKDTree

from grasp_learning_cl.grasp_heuristics.utils import thin_pointcloud
from grasp_learning_cl.grasp_classification.collision_checker import CollisionChecker


class SideGraspSampler:
    """Sampler that returns side grasps as a bounded list of grasp dicts."""
    max_finger_opening_width = 0.08
    grasp_depth_bounds = (0.0, 0.02)
    reject_failed_contacts = False
    rejection_radius = 0.01
    x_axis = np.array((0.0, 0.0, -1.0), dtype=np.float32)

    def __init__(
        self,
    ) -> None:
        self.rng = np.random.default_rng(None)
        self.collision_checker = CollisionChecker()

    def __call__(
        self,
        cloud_objects,
        cloud_collisions: None,
        n_grasps: int,
        debug: bool = False,
    ) -> list[dict]:
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
        if points.shape[0] < 2:
            return []

        table_height = float(points[:, 2].min() + 1e-3)
        mask = points[:, 2] > table_height
        points = points[mask]
        if points.shape[0] < 2:
            return []

        tree = cKDTree(points)
        rejection_neighbors = None
        if self.reject_failed_contacts:
            rejection_neighbors = tree.query_ball_point(points, r=self.rejection_radius)
        grasps: list[dict] = []
        max_attempts = max(20 * int(n_grasps), 100)
        active_mask = np.ones(points.shape[0], dtype=bool)
        active_indices = np.arange(points.shape[0], dtype=np.int32)

        for attempt_n in range(max_attempts):
            if len(grasps) >= n_grasps:
                break
            if active_indices.size == 0:
                # reset active mask and start again
                # active_mask = np.ones(points.shape[0], dtype=bool)
                # active_indices = np.arange(points.shape[0], dtype=np.int32)
                break

            grasp, source_index = self._sample_single_grasp(points, tree, active_mask, active_indices)
            if source_index is None:
                break

            if grasp is not None and (cloud_collisions_ds is None or self.collision_checker.is_collision_free(cloud_collisions_ds, grasp)):
                grasps.append(grasp)
                continue

            if self.reject_failed_contacts:
                rejected_indices = np.asarray(rejection_neighbors[source_index], dtype=np.int32)
                active_mask[rejected_indices] = False
                active_indices = active_indices[active_mask[active_indices]]

        if debug:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            hit_rate = len(grasps) / attempt_n
            print(f"{self.__class__.__name__}: hit rate {len(grasps)}/{attempt_n} ({hit_rate:.1%}), time {elapsed_ms:.2f} ms")

        return grasps

    def _sample_single_grasp(
        self,
        points: np.ndarray,
        tree: cKDTree,
        active_mask: np.ndarray,
        active_indices: np.ndarray,
    ) -> tuple[dict | None, int | None]:
        i = int(self.rng.choice(active_indices))
        point1 = points[i]

        neighbor_indices = np.asarray(tree.query_ball_point(point1, r=self.max_finger_opening_width), dtype=np.int32)
        if neighbor_indices.size == 0:
            return None, i

        z_mask = active_mask[neighbor_indices] & (neighbor_indices != i) & (np.abs(points[neighbor_indices, 2] - point1[2]) <= 0.015)
        valid_indices = neighbor_indices[z_mask]
        if valid_indices.size == 0:
            return None, i

        j = int(self.rng.choice(valid_indices))
        dx = float(points[j, 0] - point1[0])
        dy = float(points[j, 1] - point1[1])
        dist = float(np.sqrt(dx * dx + dy * dy))
        if dist > self.max_finger_opening_width or dist < 1e-4:
            return None, i

        center = np.array(
            (
                0.5 * (point1[0] + points[j, 0]),
                0.5 * (point1[1] + points[j, 1]),
                point1[2],
            ),
            dtype=np.float32,
        )
        y_axis = np.array((dx / dist, dy / dist, 0.0), dtype=np.float32)
        z_axis = np.array((dy / dist, -dx / dist, 0.0), dtype=np.float32)
        z_norm = float(np.sqrt(z_axis[0] * z_axis[0] + z_axis[1] * z_axis[1]))
        if z_norm < 1e-6:
            return None, i
        z_axis /= z_norm

        rotation = np.column_stack((self.x_axis, y_axis, z_axis)).astype(np.float32)

        grasp_depth = float(self.rng.uniform(*self.grasp_depth_bounds))
        shift = grasp_depth * rotation[:, 2]
        center = center + shift

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rotation
        pose[:3, 3] = center.astype(np.float32)

        return self._format_grasp(pose), i

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
            "sampler": "side"
        }


class SideGraspSamplerFast(SideGraspSampler):
    reject_failed_contacts = True
