from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import open3d as o3d

from .grasp_recall_module_registration_base import GraspRecallModuleRegistrationBase


class GraspRecallModuleObject(GraspRecallModuleRegistrationBase):
    """Template-based matcher with disk-backed storage."""

    def add_demonstration(
        self,
        pointcloud_no_table: o3d.geometry.PointCloud,
        pointcloud_w_table: o3d.geometry.PointCloud,
        grasps: Optional[List[dict]] = None,
        object_type: Optional[str] = None,
        grasp_type: Optional[str] = None,
    ) -> None:
        # Object recall is defined on the no-table cloud only.
        _ = pointcloud_w_table
        resolved_grasps, metadata, pcd_down, features = self._prepare_demonstration(
            pointcloud=pointcloud_no_table,
            grasps=grasps,
            object_type=object_type,
            grasp_type=grasp_type,
        )

        self._append_entry(
            pcd_full=pointcloud_no_table,
            pcd_down=pcd_down,
            features=features,
            grasps=resolved_grasps,
            metadata=metadata,
        )

    def _find_matches_for_entry(
        self,
        entry: Dict[str, Any],
        scene_feature_cloud_down: o3d.geometry.PointCloud,
        scene_icp_cloud: o3d.geometry.PointCloud,
        scene_feature_cloud_fpfh: o3d.pipelines.registration.Feature,
        max_matches_per_template: int,
        debug: bool = False,
        max_time_ms: Optional[float] = None,
    ) -> List[o3d.pipelines.registration.RegistrationResult]:
        _ = scene_icp_cloud
        _ = debug
        _ = max_time_ms
        return self._find_multiple_matches(
            template_feature_cloud_down=entry["pcd_down"],
            template_feature_cloud_fpfh=entry["features"],
            scene_feature_cloud_down=scene_feature_cloud_down,
            scene_feature_cloud_fpfh=scene_feature_cloud_fpfh,
            max_matches=max_matches_per_template,
        )

    def _find_multiple_matches(
        self,
        template_feature_cloud_down: o3d.geometry.PointCloud,
        template_feature_cloud_fpfh: o3d.pipelines.registration.Feature,
        scene_feature_cloud_down: o3d.geometry.PointCloud,
        scene_feature_cloud_fpfh: o3d.pipelines.registration.Feature,
        max_matches: int = 10,
    ) -> List[o3d.pipelines.registration.RegistrationResult]:
        inlier_distance = self.voxel_size * 2.0
        template_feature_points = np.asarray(template_feature_cloud_down.points)
        if len(template_feature_points) == 0 or len(scene_feature_cloud_down.points) == 0:
            return []

        accepted: List[o3d.pipelines.registration.RegistrationResult] = []
        active_idx = np.arange(len(scene_feature_cloud_down.points), dtype=np.int64)

        for _ in range(max_matches):
            if len(active_idx) < 3:
                break

            active_scene_feature_cloud = scene_feature_cloud_down.select_by_index(active_idx.tolist())
            active_scene_feature_cloud_fpfh = o3d.pipelines.registration.Feature()
            active_scene_feature_cloud_fpfh.data = np.asarray(scene_feature_cloud_fpfh.data)[:, active_idx]

            result = self._fast_global_registration(
                source=template_feature_cloud_down,
                source_fpfh=template_feature_cloud_fpfh,
                target=active_scene_feature_cloud,
                target_fpfh=active_scene_feature_cloud_fpfh,
                use_icp=self.use_icp_refinement,
            )
            if result is None:
                break

            accepted.append(result)

            transformed_source = (result.transformation[:3, :3] @ template_feature_points.T).T + result.transformation[:3, 3]
            active_tree = o3d.geometry.KDTreeFlann(active_scene_feature_cloud)
            explained_local: set[int] = set()
            for point in transformed_source:
                _, nn_indices, _ = active_tree.search_radius_vector_3d(
                    point.astype(np.float64), float(inlier_distance)
                )
                explained_local.update(int(i) for i in nn_indices)

            if not explained_local:
                break

            keep_mask = np.ones(len(active_idx), dtype=bool)
            keep_mask[np.fromiter(explained_local, dtype=np.int64)] = False
            if keep_mask.all():
                break
            active_idx = active_idx[keep_mask]

        return accepted

    def _fast_global_registration(
        self,
        source: o3d.geometry.PointCloud,
        source_fpfh: o3d.pipelines.registration.Feature,
        target: o3d.geometry.PointCloud,
        target_fpfh: o3d.pipelines.registration.Feature,
        use_icp: bool = True,
    ) -> Optional[o3d.pipelines.registration.RegistrationResult]:
        fgr_max_corr = float(self.max_correspondence_distance_global_registration)

        option = o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=fgr_max_corr
        )

        fg = o3d.pipelines.registration.registration_fgr_based_on_feature_matching

        try:
            fgr_result = fg(source, target, source_fpfh, target_fpfh, option)
        except RuntimeError:
            return None

        if fgr_result.fitness < self.global_fitness_threshold:
            return None
        if not use_icp:
            return fgr_result

        init_T = fgr_result.transformation
        icp_max_corr = fgr_max_corr * 0.75

        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20)

        if source.has_normals() and target.has_normals():
            estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        else:
            estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

        try:
            icp = o3d.pipelines.registration.registration_icp(
                source,
                target,
                max_correspondence_distance=icp_max_corr,
                init=init_T,
                estimation_method=estimation,
                criteria=criteria,
            )
        except RuntimeError:
            return fgr_result
        if icp.fitness > fgr_result.fitness:
            return icp
        if abs(icp.fitness - fgr_result.fitness) < 1e-9 and icp.inlier_rmse < fgr_result.inlier_rmse:
            return icp
        return fgr_result
