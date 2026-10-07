import numpy as np
import open3d as o3d


class CollisionChecker:
    """
    Collision checker for one or more axis-aligned bounding boxes defined in
    *local* frame, evaluated for multiple poses (4x4 transforms) against an
    Open3D point cloud.

    The default collision boxes approximate the Panda hand geometry used by our
    grasp samplers and experiments. Users with a different gripper should pass
    their own `boxes` argument. The expected grasp dict follows the project
    convention: `pose` is a 4x4 world-from-grasp transform, and `point1` /
    `point2` are the two fingertip/contact points used for the table check.

    Assumes each pose is a 4x4 matrix T_world_from_local:
        p_world = R @ p_local + t
    So to bring points from world into local frame we use:
        p_local = R.T @ (p_world - t)
    """
    DEFAULT_COLLISION = [
        ((-0.02, -0.10, -0.12), (0.02, 0.10, -0.04))
    ]

    def __init__(self, boxes = None, max_points: int = 1000):
        """
        Parameters
        ----------
        boxes : list of tuple pairs
            List like:
            [((xmin, ymin, zmin), (xmax, ymax, zmax)), ...].
        max_points : int, optional
            Maximum number of points to use (uniformly downsampled).
        """
        if boxes is None: boxes = self.DEFAULT_COLLISION
        try:
            box_pairs = list(boxes)
        except TypeError as exc:
            raise ValueError("boxes must be an iterable of (bbmin, bbmax) pairs.") from exc

        if len(box_pairs) == 0:
            raise ValueError("At least one bounding box pair is required.")

        mins = []
        maxs = []
        for idx, pair in enumerate(box_pairs):
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError(f"Box #{idx} must be a pair (bbmin, bbmax).")
            mins.append(pair[0])
            maxs.append(pair[1])

        bbmin_arr = np.asarray(mins, dtype=float)
        bbmax_arr = np.asarray(maxs, dtype=float)

        self.bbmin = bbmin_arr
        self.bbmax = bbmax_arr
        self.max_points = int(max_points)

    def downsample_pointcloud(self, cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Return a uniformly downsampled point cloud copy."""
        points = np.asarray(cloud.points)
        n = points.shape[0]
        if n <= self.max_points:
            return cloud

        stride = int(np.ceil(n / self.max_points))
        indices = np.arange(0, n, stride)[:self.max_points]

        pcd_down = o3d.geometry.PointCloud(cloud)
        pcd_down.points = o3d.utility.Vector3dVector(points[indices])
        if cloud.has_normals():
            normals = np.asarray(cloud.normals)
            pcd_down.normals = o3d.utility.Vector3dVector(normals[indices])
        return pcd_down

    def _collision_info(self, cloud_w_table, grasp, table_height=None, pose_key="pose"):
        """Return collision status and per-point collision mask for one grasp."""
        points = np.asarray(cloud_w_table.points, dtype=float)
        if points.size == 0:
            return {
                "collision_free": True,
                "collision": False,
                "box_collision": False,
                "table_collision": False,
                "inside_any_box": np.zeros((0,), dtype=bool),
                "table_height": table_height,
            }

        pose = np.asarray(grasp[pose_key], dtype=float)
        if pose.shape != (4, 4):
            raise ValueError(f"Grasp '{pose_key}' must be a (4,4) array.")

        R = pose[:3, :3]
        t = pose[:3, 3]

        rel = points - t
        local = rel @ R

        inside_per_box = np.all(
            (local[:, None, :] >= self.bbmin[None, :, :])
            & (local[:, None, :] <= self.bbmax[None, :, :]),
            axis=2,
        )
        inside_any_box = np.any(inside_per_box, axis=1)
        box_collision = bool(np.any(inside_any_box))

        if table_height is None:
            table_height = float(points[:, 2].min())
        table_collision = grasp["point1"][2] < table_height or grasp["point2"][2] < table_height

        collision = box_collision or table_collision
        return {
            "collision_free": not collision,
            "collision": collision,
            "box_collision": box_collision,
            "table_collision": table_collision,
            "inside_any_box": inside_any_box,
            "table_height": table_height,
        }

    def is_collision_free(self, cloud_w_table, grasp, table_height=None, pose_key="pose"):
        """Check whether a single grasp is collision free against the full point cloud."""
        return self._collision_info(
            cloud_w_table,
            grasp,
            table_height=table_height,
            pose_key=pose_key,
        )["collision_free"]

    def filter_grasps(
        self,
        pcd: o3d.geometry.PointCloud,
        grasp_candidates,
        debug: bool = False,
        pose_keys=("pose",),
    ):
        """
        Parameters
        ----------
        pcd : open3d.geometry.PointCloud
            Input point cloud in world frame.
        grasp_candidates : list of dict
            Each dict must contain key "pose": a (4,4) numpy array transform
            defining the local frame of the box in world coordinates
            (T_world_from_local).
        debug : bool, optional
            If True, show one Open3D visualization for every grasp being checked.
            Colliding point-cloud points are colored red.
        pose_keys : tuple[str], optional
            Grasp transform keys that must all be collision-free.

        Returns
        -------
        grasp_candidates : list of dict (filtered)
        """
        points = np.asarray(pcd.points)
        if points.size == 0:
            # No points = no collision for any pose
            for elem in grasp_candidates:
                elem["score_collision"] = 1.0
            return grasp_candidates

        table_height = float(points[:, 2].min())
        pcd_down = self.downsample_pointcloud(pcd)

        filtered = list()
        for idx, elem in enumerate(grasp_candidates):
            collision_infos = [
                self._collision_info(
                    pcd_down,
                    elem,
                    table_height=table_height,
                    pose_key=pose_key,
                )
                for pose_key in pose_keys
            ]
            if debug:
                for pose_key in pose_keys:
                    debug_collision_info = self._collision_info(
                        pcd,
                        elem,
                        table_height=table_height,
                        pose_key=pose_key,
                    )
                    self.visualize_collision_check(
                        pcd,
                        elem,
                        debug_collision_info,
                        grasp_index=idx,
                        grasp_count=len(grasp_candidates),
                        pose_key=pose_key,
                    )
            if all(info["collision_free"] for info in collision_infos):
                filtered.append(elem)
        return filtered

    def __call__(self, pcd: o3d.geometry.PointCloud, grasp_candidates, debug: bool = False):
        print("Please switch to calling self.filter_grasps")
        return self.filter_grasps(pcd, grasp_candidates, debug=debug)

    def get_collision_geometries(
        self,
        color=(1.0, 0.1, 0.1),
        show_grasp_frame=True,
        grasp_frame_size=0.04,
        pose=None,
    ):
        """
        Visualize the local collision boxes in an empty Open3D scene.

        Parameters
        ----------
        color : tuple
            RGB color used for all collision-box edges.
        show_grasp_frame : bool
            Whether to show the local grasp frame at the origin.
        grasp_frame_size : float
            Size of the displayed grasp frame.
        pose : numpy.ndarray, optional
            4x4 transform used to place the local collision geometry in world frame.
        """
        transform = None if pose is None else np.asarray(pose, dtype=float)
        geometries = []
        for bbmin, bbmax in zip(self.bbmin, self.bbmax):
            box = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=bbmin,
                max_bound=bbmax,
            )
            lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(box)
            lines.paint_uniform_color(color)
            if transform is not None:
                lines.transform(transform)
            geometries.append(lines)

        if show_grasp_frame:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=grasp_frame_size)
            if transform is not None:
                frame.transform(transform)
            geometries.append(frame)

        return geometries

    def visualize_collision_geometry(self):
        geometries = self.get_collision_geometries()
        o3d.visualization.draw_geometries(geometries)
        return geometries

    def visualize_collision_check(
        self,
        pcd: o3d.geometry.PointCloud,
        grasp,
        collision_info=None,
        grasp_index=None,
        grasp_count=None,
        pose_key="pose",
    ):
        """
        Visualize one grasp collision check in world frame.

        The point cloud is copied before coloring, so caller-owned colors are not
        modified. Points inside any collision box are shown in red.
        """
        if collision_info is None:
            collision_info = self._collision_info(pcd, grasp, pose_key=pose_key)

        pcd_debug = o3d.geometry.PointCloud(pcd)
        points = np.asarray(pcd_debug.points)
        if points.size > 0:
            colors = np.tile(np.array([[0.65, 0.65, 0.65]]), (points.shape[0], 1))
            colliding_points = collision_info["inside_any_box"]
            colors[colliding_points] = np.array([1.0, 0.0, 0.0])
            pcd_debug.colors = o3d.utility.Vector3dVector(colors)

        pose = np.asarray(grasp[pose_key], dtype=float)
        geometries = [pcd_debug]
        geometries.extend(
            self.get_collision_geometries(
                pose=pose,
                color=(1.0, 0.35, 0.0),
                show_grasp_frame=True,
            )
        )

        title_parts = [f"Collision checker {pose_key}"]
        if grasp_index is not None:
            if grasp_count is None:
                title_parts.append(f"grasp {grasp_index}")
            else:
                title_parts.append(f"grasp {grasp_index + 1}/{grasp_count}")
        title_parts.append(
            "collision"
            if collision_info["collision"]
            else "collision free"
        )
        title_parts.append(f"{int(np.sum(collision_info['inside_any_box']))} pcd hits")
        if collision_info["table_collision"]:
            title_parts.append("table collision")

        o3d.visualization.draw_geometries(
            geometries,
            window_name=" - ".join(title_parts),
        )
        return geometries


if __name__ == "__main__":
    collision_checker = CollisionChecker()
    collision_checker.visualize_collision_geometry()
