import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def thin_pointcloud(
    cloud: o3d.geometry.PointCloud,
    radius: float = 0.03,
) -> o3d.geometry.PointCloud:
    points = np.asarray(cloud.points, dtype=np.float32)
    if len(points) < 2:
        return cloud

    rng = np.random.default_rng(None)
    tree = cKDTree(points)
    neighbor_lists = tree.query_ball_point(points, r=radius)
    neighbor_counts = np.array([max(len(neighbors), 1) for neighbors in neighbor_lists], dtype=np.float32)
    min_neighbor_count = float(np.min(neighbor_counts))
    if not np.isfinite(min_neighbor_count) or min_neighbor_count <= 0.0:
        return cloud

    keep_probabilities = min_neighbor_count / neighbor_counts
    keep_mask = rng.random(len(points)) < keep_probabilities
    selected_indices = np.flatnonzero(keep_mask)

    if len(selected_indices) < 2:
        selected_indices = np.argsort(neighbor_counts)[: min(2, len(points))]
    return cloud.select_by_index(selected_indices.tolist())
