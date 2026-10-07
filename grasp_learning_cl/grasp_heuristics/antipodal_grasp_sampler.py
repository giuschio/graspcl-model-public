import time

import numpy as np
from scipy.spatial import cKDTree

from grasp_learning_cl.grasp_heuristics.utils import thin_pointcloud
from grasp_learning_cl.grasp_classification.collision_checker import CollisionChecker


class AntipodalGraspSampler:
    """Sampler that returns antipodal pinch grasps as a bounded list of grasp dicts."""

    max_finger_opening_width = 0.08
    normal_alignment_thresh = 0.8
    hand_size_y = 0.2
    hand_size_z = 0.15
    grasp_depth_bounds = (0.0, 0.02)
    reject_failed_contacts = False
    rejection_radius = 0.01
    z_reference_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    x_reference_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)

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

        if not cloud_objects.has_normals():
            raise ValueError("Please estimate cloud normals before putting it in the sampler")
        normals = np.asarray(cloud_objects.normals, dtype=np.float32)

        table_height = float(points[:, 2].min() + 1e-3)
        mask = points[:, 2] > table_height
        points = points[mask]
        normals = normals[mask]
        normal_norms = np.linalg.norm(normals, axis=1)
        valid_normals = normal_norms > 1e-8
        points = points[valid_normals]
        normals = normals[valid_normals] / normal_norms[valid_normals, None]
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

            grasp, source_index = self._sample_single_grasp(points, normals, tree, table_height, active_mask, active_indices)
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
        normals: np.ndarray,
        tree: cKDTree,
        table_height: float,
        active_mask: np.ndarray,
        active_indices: np.ndarray,
    ) -> tuple[dict | None, int | None]:
        i = int(self.rng.choice(active_indices))
        point1 = points[i]
        normal1 = normals[i]

        neighbor_indices = np.asarray(tree.query_ball_point(point1, r=self.max_finger_opening_width), dtype=np.int64)
        valid_indices = neighbor_indices[active_mask[neighbor_indices] & (neighbor_indices != i)]
        if valid_indices.size == 0:
            return None, i

        candidate_points = points[valid_indices]
        candidate_normals = normals[valid_indices]
        offsets = candidate_points - point1[None, :]
        dists = np.linalg.norm(offsets, axis=1)
        nonzero_mask = dists >= 1e-4
        if not np.any(nonzero_mask):
            return None, i

        valid_indices = valid_indices[nonzero_mask]
        candidate_points = candidate_points[nonzero_mask]
        candidate_normals = candidate_normals[nonzero_mask]
        offsets = offsets[nonzero_mask]
        dists = dists[nonzero_mask]
        grasp_dirs = offsets / dists[:, None]

        antipodal_mask = np.einsum("ij,j->i", -candidate_normals, normal1) >= self.normal_alignment_thresh
        alignment1_mask = np.abs(np.einsum("ij,j->i", grasp_dirs, normal1)) >= self.normal_alignment_thresh
        alignment2_mask = np.abs(np.einsum("ij,ij->i", -grasp_dirs, candidate_normals)) >= self.normal_alignment_thresh
        candidate_mask = antipodal_mask & alignment1_mask & alignment2_mask
        if not np.any(candidate_mask):
            return None, i

        chosen = int(self.rng.choice(np.flatnonzero(candidate_mask)))
        j = int(valid_indices[chosen])
        point2 = candidate_points[chosen]
        dist = float(dists[chosen])
        grasp_dir = grasp_dirs[chosen].astype(np.float32)

        center = 0.5 * (point1 + point2)
        y_axis = grasp_dir.astype(np.float32)
        reference_axis = self.z_reference_axis
        if abs(float(np.dot(y_axis, reference_axis))) > 0.95:
            reference_axis = self.x_reference_axis
        x_axis = self._normalize(np.cross(y_axis, reference_axis))
        if x_axis is None:
            return None, i
        z_axis = self._normalize(np.cross(x_axis, y_axis))
        if z_axis is None:
            return None, i

        base_rotation = np.column_stack((x_axis, y_axis, z_axis)).astype(np.float32)

        angle = float(self.rng.uniform(0.0, 2.0 * np.pi))
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        roll = np.array(
            [[cos_a, 0.0, sin_a], [0.0, 1.0, 0.0], [-sin_a, 0.0, cos_a]],
            dtype=np.float32,
        )
        rotation = (base_rotation @ roll).astype(np.float32)

        grasp_depth = float(self.rng.uniform(*self.grasp_depth_bounds))
        center = center + grasp_depth * rotation[:, 2]

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rotation
        pose[:3, 3] = center.astype(np.float32)

        if not self._pose_clears_table(pose, table_height):
            return None, i

        return self._format_grasp(pose), i

    def _pose_clears_table(self, pose: np.ndarray, table_height: float) -> bool:
        dy = 0.5 * self.hand_size_y
        dz_front = 0.0
        dz_back = -self.hand_size_z
        hand_points_local = np.array(
            [[0.0, dy, dz_front], [0.0, -dy, dz_front], [0.0, dy, dz_back], [0.0, -dy, dz_back]],
            dtype=np.float32,
        )
        hand_points_world = hand_points_local @ pose[:3, :3].T + pose[:3, 3]
        return bool(np.all(hand_points_world[:, 2] >= table_height))

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
            "sampler": "antipodal"
            
        }

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray | None:
        norm = float(np.linalg.norm(vector))
        if norm < 1e-8:
            return None
        return (vector / norm).astype(np.float32)


class AntipodalGraspSamplerFast(AntipodalGraspSampler):
    reject_failed_contacts = True
