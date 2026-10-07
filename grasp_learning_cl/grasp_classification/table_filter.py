import numpy as np
import open3d as o3d


class TableFilter:
    def __init__(self):
        self.ransac_inlier_distance=0.002       # RANSAC inlier threshold
        self.point_height_threshold=0.01        # keep points higher than this above plane
        self.point_angle_threshold=30.0         # degrees

    def __call__(
        self,
        pcd: o3d.geometry.PointCloud,
        debug: bool = False
    ):
        """
        Remove table using RANSAC plane + global height & normal thresholds.

        Rules:
        - dist < 0            -> remove
        - 0 <= dist <= h      -> remove if normal ~ plane normal
        - dist > h            -> keep
        """
        if len(pcd.points) < 3:
            return pcd

        # --- RANSAC plane ---
        plane_model, _ = pcd.segment_plane(
            distance_threshold=self.ransac_inlier_distance,
            ransac_n=3,
            num_iterations=2000
        )

        a, b, c, d = plane_model
        n = np.array([a, b, c])
        n /= np.linalg.norm(n)

        pts = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        normals_n = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        # --- signed distance to plane ---
        dist = pts @ n + d

        # --- normal similarity ---
        cos_thresh = np.cos(np.deg2rad(self.point_angle_threshold))
        normal_match = np.abs(normals_n @ n) > cos_thresh

        # --- table mask (your exact rule) ---
        table_mask = (
            (dist < 0) |
            ((dist >= 0) & (dist <= self.point_height_threshold) & normal_match)
        )

        keep_idx = np.where(~table_mask)[0]
        before_count = pts.shape[0]
        after_count = keep_idx.shape[0]
        pcd = pcd.select_by_index(keep_idx)

        if len(pcd.points) < 2:
            return pcd

        nearest_neighbor_distances = pcd.compute_nearest_neighbor_distance()
        if len(nearest_neighbor_distances) == 0:
            return pcd

        implied_voxel_size = np.median(nearest_neighbor_distances)
        if not np.isfinite(implied_voxel_size) or implied_voxel_size <= 0:
            return pcd

        pcd, _ = pcd.remove_statistical_outlier(20, 2.0)
        if len(pcd.points) < 12:
            return pcd

        pcd, _ = pcd.remove_radius_outlier(12, 5 * implied_voxel_size)

        if debug:
            table_height = float("nan")
            if abs(c) > 1e-12:
                table_height = -d / c
            print(
                "Table height (world): {:.6f}, point_height_threshold: {:.6f}, points before/after: {}/{}. Implied voxelization size {}".format(
                    table_height,
                    self.point_height_threshold,
                    before_count,
                    after_count,
                    implied_voxel_size,
                )
            )

        return pcd
