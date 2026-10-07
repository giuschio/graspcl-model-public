import time

import numpy as np
from scipy.spatial import cKDTree

from grasp_learning_cl.grasp_heuristics.utils import thin_pointcloud
from grasp_learning_cl.grasp_classification.collision_checker import CollisionChecker


class EdgeGraspSampler:
    """NumPy-only grasp sampler that follows the EdgeGrasp geometric construction.

    Each grasp is sampled independently from a point pair in the cloud:
    an approach point is selected, a nearby contact point is chosen, and the
    local grasp frame is derived from the contact normal and the point-pair
    geometry. 
    """

    neighbor_radius = 0.055
    min_pair_distance = 0.003
    max_pair_distance = 0.038
    min_depth_proj = 0.0
    max_depth_proj = 0.038
    retract_distance = 0.06
    table_height = 0.054
    table_mask = "autodetect"
    reject_failed_contacts = False
    rejection_radius = 0.01

    def __init__(
        self,
    ) -> None:
        """Initialize the sampler.

        Args:
            table_height: Table height in the point-cloud frame.
            table_mask: Whether to reject grasps that collide with the table.
                When set to ``"autodetect"``, the table height is inferred from
                the minimum scene ``z`` value.
        """
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

        Returns:
            A list of grasp dictionaries with ``pose``, ``width``, ``point1``,
            and ``point2`` entries. Fewer grasps may be returned if valid
            samples cannot be found within the attempt budget.
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

        tree = cKDTree(points)
        rejection_neighbors = None
        if self.reject_failed_contacts:
            rejection_neighbors = tree.query_ball_point(points, r=self.rejection_radius)
        grasps: list[dict] = []
        table_height = self._resolve_table_height(points)

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
        """Sample one grasp candidate and return it if it passes all filters."""
        approach_idx = int(self.rng.choice(active_indices))
        approach_point = points[approach_idx]

        neighbor_indices = np.asarray(tree.query_ball_point(approach_point, r=self.neighbor_radius), dtype=np.int32)
        valid_neighbors = neighbor_indices[active_mask[neighbor_indices]]
        neighbor_count = valid_neighbors.size
        if neighbor_count <= 1:
            return None, approach_idx
        
        contact_idx = int(valid_neighbors[int(self.rng.integers(neighbor_count))])
        if contact_idx == approach_idx:
            return None, approach_idx
        contact_point = points[contact_idx]
        contact_normal = self._normalize(normals[contact_idx])
        if contact_normal is None:
            return None, approach_idx

        relative_pos = contact_point - approach_point
        relative_norm = float(np.linalg.norm(relative_pos))
        if relative_norm < 1e-8:
            return None, approach_idx
        relative_pos_normalized = relative_pos / relative_norm

        x_axis = self._normalize(np.cross(contact_normal, relative_pos_normalized))
        if x_axis is None:
            return None, approach_idx

        valid_edge_approach = -self._normalize(np.cross(x_axis, contact_normal))
        if valid_edge_approach is None:
            return None, approach_idx

        up_dot = float(valid_edge_approach[2])
        if up_dot <= -0.1:
            valid_edge_approach = valid_edge_approach.copy()
            valid_edge_approach[2] = 0.0
            valid_edge_approach = self._normalize(valid_edge_approach)
            if valid_edge_approach is None: 
                return None, approach_idx

        grasp_depth = -np.dot(contact_point - approach_point, valid_edge_approach)
        grasp_depth = float(np.clip(grasp_depth, self.min_depth_proj, self.max_depth_proj,))

        z_axis = -valid_edge_approach
        gripper_dis_from_source = 0.072 - 0.007 - grasp_depth
        grasp_center = approach_point + gripper_dis_from_source * valid_edge_approach

        signed_halfwidth = float(np.dot(contact_point - grasp_center, contact_normal))
        grasp_halfwidth = abs(signed_halfwidth)
        if not (self.min_pair_distance < grasp_halfwidth < self.max_pair_distance):
            target_halfwidth = float(np.clip(grasp_halfwidth, self.min_pair_distance, self.max_pair_distance,))
            target_signed_halfwidth = np.copysign(target_halfwidth, signed_halfwidth if signed_halfwidth != 0.0 else 1.0)
            grasp_center = grasp_center + (signed_halfwidth - target_signed_halfwidth) * contact_normal

        y_axis = contact_normal
        x_axis = self._normalize(np.cross(y_axis, z_axis))
        if x_axis is None:
            return None, approach_idx
        y_axis = self._normalize(np.cross(z_axis, x_axis))
        if y_axis is None:
            return None, approach_idx

        pose = np.eye(4, dtype=np.float32)
        pose[:3, 0] = x_axis
        pose[:3, 1] = y_axis
        pose[:3, 2] = z_axis
        pose[:3, 3] = grasp_center.astype(np.float32)
        if self.table_mask and not self._pose_clears_table(pose, table_height):
            return None, approach_idx

        pose = pose.copy()
        pose[:3, 3] += self.retract_distance * pose[:3, 2]

        center = pose[:3, 3]
        y_axis = pose[:3, 1]
        point1 = center + 0.04 * y_axis
        point2 = center - 0.04 * y_axis
        return {
            "pose": pose.astype(np.float32),
            "width": 0.08,
            "point1": point1.astype(np.float32),
            "point2": point2.astype(np.float32),
            "sampler": "edge"
        }, approach_idx

    def _pose_clears_table(self, pose: np.ndarray, table_height: float) -> bool:
        """Check whether all gripper keypoints stay above the table plane."""
        gripper_points = self._get_gripper_points(pose)
        return bool(np.all(gripper_points[:, 2] > table_height))

    @staticmethod
    def _get_gripper_points(pose: np.ndarray) -> np.ndarray:
        """Transform the gripper collision keypoints into world coordinates."""
        gripper_points_local = np.array(
            [
                [0.0, 0.0, -0.02],
                [0.012, -0.09, 0.015],
                [-0.012, -0.09, 0.015],
                [0.012, 0.09, 0.015],
                [-0.012, 0.09, 0.015],
                [0.005, 0.09, 0.078],
                [0.005, -0.09, 0.078],
            ],
            dtype=np.float32,
        )
        return gripper_points_local @ pose[:3, :3].T + pose[:3, 3]

    def _resolve_table_height(self, points: np.ndarray) -> float:
        """Resolve the active table height, optionally from the scene itself."""
        if self.table_mask == "autodetect":
            return float(points[:, 2].min())
        return self.table_height

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray | None:
        """Return a unit vector, or ``None`` for degenerate inputs."""
        norm = float(np.linalg.norm(vector))
        if norm < 1e-8:
            return None
        return vector / norm


class EdgeGraspSamplerFast(EdgeGraspSampler):
    reject_failed_contacts = True
