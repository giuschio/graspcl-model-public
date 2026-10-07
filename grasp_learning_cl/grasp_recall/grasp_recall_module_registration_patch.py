from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from .grasp_recall_module_registration_base import GraspRecallModuleRegistrationBase


class GraspRecallModulePatch(GraspRecallModuleRegistrationBase):
    """Template-based matcher with disk-backed storage."""

    def __init__(
        self,
        voxel_size: float = 0.005,
        correspondence_multiplier: float = 1.5,
        fitness_threshold: float = 0.4,
        partition_radius: float = 0.04,
        min_partition_points: int = 40,
        sampled_centroids: int = 100,
        use_icp_refinement: bool = True,
        run_icp_on_full_cloud: bool = False,
        data_folder: str | Path | None = None,
    ) -> None:
        self.partition_radius = float(partition_radius)
        self.min_partition_points = int(min_partition_points)
        self.sampled_centroids = int(sampled_centroids)
        self.run_icp_on_full_cloud = bool(run_icp_on_full_cloud)
        super().__init__(
            voxel_size=voxel_size,
            correspondence_multiplier=correspondence_multiplier,
            fitness_threshold=fitness_threshold,
            use_icp_refinement=use_icp_refinement,
            data_folder=data_folder,
        )

    def add_demonstration(
        self,
        pointcloud_no_table: o3d.geometry.PointCloud,
        pointcloud_w_table: o3d.geometry.PointCloud,
        grasps: Optional[List[dict]] = None,
        object_type: Optional[str] = None,
        grasp_type: Optional[str] = None,
        debug: bool = False,
    ) -> None:
        # Templates are indexed and matched using the no-table cloud.
        template_feature_cloud = pointcloud_no_table
        # ICP refinement should see the table, so we store local geometry from this cloud.
        template_icp_cloud = pointcloud_w_table

        resolved_grasps, metadata, template_feature_cloud_down, template_feature_cloud_fpfh = self._prepare_demonstration(
            pointcloud=template_feature_cloud,
            grasps=grasps,
            object_type=object_type,
            grasp_type=grasp_type,
        )

        template_fpfh_data = np.asarray(template_feature_cloud_fpfh.data)
        sphere_center_offset_in_grasp = np.array([0.0, 0.0, 0.0], dtype=float)
        template_feature_points = np.asarray(template_feature_cloud_down.points)
        template_icp_points = np.asarray(template_icp_cloud.points)
        if len(template_feature_points) == 0:
            return
        debug_selected_icp_mask = np.zeros(len(template_icp_points), dtype=bool) if debug else None
        debug_grasp_centers: List[np.ndarray] = [] if debug else []

        for grasp in resolved_grasps:
            pose = np.asarray(grasp["pose"], dtype=float)
            sphere_center_world = pose[:3, :3] @ sphere_center_offset_in_grasp + pose[:3, 3]
            if debug:
                debug_grasp_centers.append(sphere_center_world)
            feature_patch_sq_dist = np.sum((template_feature_points - sphere_center_world) ** 2, axis=1)
            feature_patch_idx = np.flatnonzero(feature_patch_sq_dist <= (self.partition_radius ** 2))
            # if len(feature_patch_idx) < 3:
            #     continue

            feature_patch_down = template_feature_cloud_down.select_by_index(feature_patch_idx.tolist())
            feature_patch_fpfh = o3d.pipelines.registration.Feature()
            feature_patch_fpfh.data = np.asarray(template_fpfh_data[:, feature_patch_idx], dtype=np.float64)
            local_grasp = deepcopy(grasp)
            icp_patch_idx = np.flatnonzero(
                np.sum((template_icp_points - sphere_center_world) ** 2, axis=1) <= (self.partition_radius ** 2)
            )
            if debug_selected_icp_mask is not None and len(icp_patch_idx) > 0:
                debug_selected_icp_mask[icp_patch_idx] = True
            if len(icp_patch_idx) >= 3:
                icp_patch_full = template_icp_cloud.select_by_index(icp_patch_idx.tolist())
            else:
                icp_patch_full = o3d.geometry.PointCloud(feature_patch_down)

            self._append_entry(
                pcd_full=icp_patch_full,
                pcd_down=feature_patch_down,
                features=feature_patch_fpfh,
                grasps=[local_grasp],
                metadata=metadata,
            )

        if debug and debug_selected_icp_mask is not None:
            icp_cloud_vis = o3d.geometry.PointCloud(template_icp_cloud)
            icp_cloud_vis.paint_uniform_color([0.5, 0.5, 0.5])
            highlighted_idx = np.flatnonzero(debug_selected_icp_mask)
            geometries: List[o3d.geometry.Geometry] = [icp_cloud_vis]
            if len(highlighted_idx) > 0:
                highlighted = template_icp_cloud.select_by_index(highlighted_idx.tolist())
                highlighted.paint_uniform_color([1.0, 1.0, 0.0])
                geometries.append(highlighted)
            for grasp_center in debug_grasp_centers:
                grasp_center_vis = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
                grasp_center_vis.paint_uniform_color([0.0, 0.0, 1.0])
                grasp_center_vis.translate(grasp_center)
                geometries.append(grasp_center_vis)
            o3d.visualization.draw_geometries(geometries)

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
        return self._find_multiple_matches(
            template_feature_patch_down=entry["pcd_down"],
            template_icp_patch=entry["pcd_full"],
            template_feature_patch_fpfh=entry["features"],
            scene_feature_cloud_down=scene_feature_cloud_down,
            scene_icp_cloud=scene_icp_cloud,
            scene_feature_cloud_fpfh=scene_feature_cloud_fpfh,
            max_matches=max_matches_per_template,
            log_context=f"id={entry.get('id', 'unknown')}, metadata={entry.get('metadata', {})}",
            debug=debug,
            max_time_ms=max_time_ms,
        )

    def _normalized_mean_fpfh(self, feature_data: np.ndarray) -> Optional[np.ndarray]:
        if feature_data.size == 0 or feature_data.ndim != 2:
            return None
        mean_feature = feature_data.mean(axis=1)
        norm = float(np.linalg.norm(mean_feature))
        if norm <= 1e-12:
            return None
        return mean_feature / norm

    def _normalized_avg_normal(self, normals: np.ndarray) -> Optional[np.ndarray]:
        if normals.size == 0 or normals.ndim != 2:
            return None
        mean_normal = normals.mean(axis=0)
        norm = float(np.linalg.norm(mean_normal))
        if norm <= 1e-12:
            return None
        return mean_normal / norm

    def _rotation_matrix_from_vectors(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        c = float(np.dot(a, b))
        if c >= 1.0 - 1e-8:
            return np.eye(3, dtype=np.float64)
        if c <= -1.0 + 1e-8:
            axis = np.cross(a, np.array([1.0, 0.0, 0.0], dtype=np.float64))
            if np.linalg.norm(axis) <= 1e-8:
                axis = np.cross(a, np.array([0.0, 1.0, 0.0], dtype=np.float64))
            axis = axis / np.linalg.norm(axis)
            return Rotation.from_rotvec(np.pi * axis).as_matrix()

        rot, _ = Rotation.align_vectors(np.asarray([b]), np.asarray([a]))
        return rot.as_matrix()

    def _rotation_matrix_from_axis_angle(self, axis: np.ndarray, angle_rad: float) -> np.ndarray:
        axis = np.asarray(axis, dtype=np.float64)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-12:
            return np.eye(3, dtype=np.float64)
        return Rotation.from_rotvec((axis / axis_norm) * float(angle_rad)).as_matrix()

    def _icp_with_centroid_init(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        max_correspondence_distance: float,
        source_center: np.ndarray,
        source_avg_normal: Optional[np.ndarray],
        init_rotation: Optional[np.ndarray] = None,
        use_icp: bool = True,
    ) -> Optional[o3d.pipelines.registration.RegistrationResult]:
        if len(target.points) == 0:
            return None

        if init_rotation is not None:
            init_R = np.asarray(init_rotation, dtype=np.float64)
        else:
            init_R = np.eye(3, dtype=np.float64)
            if source_avg_normal is not None and target.has_normals():
                target_avg_normal = self._normalized_avg_normal(np.asarray(target.normals))
                if target_avg_normal is not None:
                    init_R = self._rotation_matrix_from_vectors(source_avg_normal, target_avg_normal)

        target_center = np.asarray(target.get_center(), dtype=np.float64)
        init_T = np.eye(4, dtype=np.float64)
        init_T[:3, :3] = init_R
        init_T[:3, 3] = target_center - (init_R @ source_center)

        if not use_icp:
            return o3d.pipelines.registration.evaluate_registration(
                source,
                target,
                max_correspondence_distance=max_correspondence_distance,
                transformation=init_T,
            )

        if source.has_normals() and target.has_normals():
            estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        else:
            estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=3)

        try:
            return o3d.pipelines.registration.registration_icp(
                source,
                target,
                max_correspondence_distance=max_correspondence_distance,
                init=init_T,
                estimation_method=estimation,
                criteria=criteria,
            )
        except RuntimeError:
            return None

    def _find_multiple_matches(
        self,
        template_feature_patch_down: o3d.geometry.PointCloud,
        template_icp_patch: o3d.geometry.PointCloud,
        template_feature_patch_fpfh: o3d.pipelines.registration.Feature,
        scene_feature_cloud_down: o3d.geometry.PointCloud,
        scene_icp_cloud: o3d.geometry.PointCloud,
        scene_feature_cloud_fpfh: o3d.pipelines.registration.Feature,
        max_matches: int = 10,
        log_context: str = "",
        debug: bool = False,
        max_time_ms: Optional[float] = None,
    ) -> List[o3d.pipelines.registration.RegistrationResult]:
        total_t0 = time.perf_counter()
        t0 = total_t0
        scene_feature_points = np.asarray(scene_feature_cloud_down.points)
        if len(scene_feature_points) == 0:
            return []
        scene_feature_tree = o3d.geometry.KDTreeFlann(scene_feature_cloud_down)
        scene_fpfh_data = np.asarray(scene_feature_cloud_fpfh.data)
        template_fpfh_data = np.asarray(template_feature_patch_fpfh.data)
        template_feature_signature = self._normalized_mean_fpfh(template_fpfh_data)
        template_icp_center = np.asarray(template_icp_patch.get_center(), dtype=np.float64)
        template_icp_avg_normal = (
            self._normalized_avg_normal(np.asarray(template_icp_patch.normals)) if template_icp_patch.has_normals() else None
        )
        scene_icp_points = np.asarray(scene_icp_cloud.points)
        # if len(template_icp_patch.points) < 3:
        #     return []
        # if len(scene_icp_points) < 3:
        #     return []
        scene_icp_tree = o3d.geometry.KDTreeFlann(scene_icp_cloud)
        setup_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        sampled_centroids = min(len(scene_feature_points), self.sampled_centroids)
        if sampled_centroids <= 0:
            return []

        rng = np.random.default_rng(0)
        candidate_centroids = rng.choice(
            len(scene_feature_points), size=sampled_centroids, replace=False
        ).astype(np.int64)

        candidate_feature_partitions = []
        for center_idx in candidate_centroids:
            _, idx, _ = scene_feature_tree.search_radius_vector_3d(
                scene_feature_points[int(center_idx)], self.partition_radius
            )
            partition_idx = np.asarray(idx, dtype=np.int64)
            if len(partition_idx) >= self.min_partition_points:
                candidate_feature_partitions.append((int(center_idx), partition_idx))
        partition_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        ranked_centroids: List[Tuple[int, np.ndarray, float]] = []
        if template_feature_signature is None:
            ranked_centroids = [(center_idx, partition_idx, 0.0) for center_idx, partition_idx in candidate_feature_partitions]
        else:
            for center_idx, partition_idx in candidate_feature_partitions:
                partition_signature = self._normalized_mean_fpfh(scene_fpfh_data[:, partition_idx])
                if partition_signature is None:
                    cosine_similarity = -1.0
                else:
                    cosine_similarity = float(np.dot(template_feature_signature, partition_signature))
                ranked_centroids.append((center_idx, partition_idx, cosine_similarity))
        ranked_centroids.sort(key=lambda x: x[2], reverse=True)
        ranking_ms = (time.perf_counter() - t0) * 1000

        accepted: List[o3d.pipelines.registration.RegistrationResult] = []
        in_plane_rotation_degrees = (0.0, 90.0, 180.0, 270.0)
        center_cursor = 0
        target_ms = 0.0
        rotation_ms = 0.0
        icp_ms = 0.0
        icp_trials = 0
        while len(accepted) < max_matches and center_cursor < len(ranked_centroids):
            if max_time_ms is not None and (time.perf_counter() - total_t0) * 1000 >= max_time_ms:
                break
            _, partition_idx, cosine_similarity = ranked_centroids[center_cursor]
            center_cursor += 1

            if template_feature_signature is not None and cosine_similarity < 0.7:
                continue

            feature_partition = scene_feature_cloud_down.select_by_index(partition_idx.tolist())
            icp_source = template_icp_patch
            icp_source_center = template_icp_center
            icp_source_avg_normal = template_icp_avg_normal
            t0 = time.perf_counter()
            if self.run_icp_on_full_cloud:
                icp_target = scene_icp_cloud
            else:
                partition_center = np.asarray(feature_partition.get_center(), dtype=np.float64)
                _, full_partition_idx, _ = scene_icp_tree.search_radius_vector_3d(
                    partition_center, self.partition_radius
                )
                if len(full_partition_idx) < 3:
                    target_ms += (time.perf_counter() - t0) * 1000
                    continue
                icp_target = scene_icp_cloud.select_by_index(list(full_partition_idx))
            target_ms += (time.perf_counter() - t0) * 1000

            result: Optional[o3d.pipelines.registration.RegistrationResult] = None
            best_fitness = -np.inf
            best_rmse = np.inf

            t0 = time.perf_counter()
            hypothesis_rotations: List[Optional[np.ndarray]] = [None]
            if icp_source_avg_normal is not None and icp_target.has_normals():
                target_avg_normal = self._normalized_avg_normal(np.asarray(icp_target.normals))
                if target_avg_normal is not None:
                    base_rotation = self._rotation_matrix_from_vectors(icp_source_avg_normal, target_avg_normal)
                    hypothesis_rotations = [
                        self._rotation_matrix_from_axis_angle(target_avg_normal, np.deg2rad(deg)) @ base_rotation
                        for deg in in_plane_rotation_degrees
                    ]
            rotation_ms += (time.perf_counter() - t0) * 1000

            for init_rotation in hypothesis_rotations:
                if max_time_ms is not None and (time.perf_counter() - total_t0) * 1000 >= max_time_ms:
                    break
                t0 = time.perf_counter()
                trial_result = self._icp_with_centroid_init(
                    source=icp_source,
                    target=icp_target,
                    max_correspondence_distance=self.max_correspondence_distance_global_registration,
                    source_center=icp_source_center,
                    source_avg_normal=icp_source_avg_normal,
                    init_rotation=init_rotation,
                    use_icp=self.use_icp_refinement,
                )
                trial_ms = (time.perf_counter() - t0) * 1000
                icp_ms += trial_ms
                icp_trials += 1
                if trial_result is None:
                    continue
                trial_fitness = float(trial_result.fitness)
                trial_rmse = float(trial_result.inlier_rmse)
                if trial_fitness > best_fitness or (
                    trial_fitness == best_fitness and trial_rmse < best_rmse
                ):
                    result = trial_result
                    best_fitness = trial_fitness
                    best_rmse = trial_rmse

            if result is None or result.fitness < self.global_fitness_threshold:
                continue
            accepted.append(result)
        t0 = time.perf_counter()
        accepted.sort(key=lambda r: (r.fitness, -r.inlier_rmse), reverse=True)
        sort_ms = (time.perf_counter() - t0) * 1000
        total_ms = (time.perf_counter() - total_t0) * 1000
        if debug:
            print(
                "Patch recall timing [ms]: setup={:.2f}, partitions={:.2f}, ranking={:.2f}, target={:.2f}, rotations={:.2f}, icp={:.2f} ({} trials), sort={:.2f}, total={:.2f}; candidates={}, ranked={}, attempted={}, accepted={}".format(
                    setup_ms,
                    partition_ms,
                    ranking_ms,
                    target_ms,
                    rotation_ms,
                    icp_ms,
                    icp_trials,
                    sort_ms,
                    total_ms,
                    len(candidate_feature_partitions),
                    len(ranked_centroids),
                    center_cursor,
                    len(accepted),
                )
                + (f"; {log_context}" if log_context else "")
            )
        return accepted[:max_matches]

    def _fast_global_registration(
        self,
        source: o3d.geometry.PointCloud,
        source_fpfh: o3d.pipelines.registration.Feature,
        target: o3d.geometry.PointCloud,
        target_fpfh: o3d.pipelines.registration.Feature,
    ) -> Optional[o3d.pipelines.registration.RegistrationResult]:

        # ---------- FGR ----------
        fgr_max_corr = float(self.max_correspondence_distance_global_registration)

        option = o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=fgr_max_corr
        )

        fg = o3d.pipelines.registration.registration_fgr_based_on_feature_matching

        best: Optional[o3d.pipelines.registration.RegistrationResult] = None
        best_fitness = float("-inf")

        for _ in range(7):
            try:
                result = fg(source, target, source_fpfh, target_fpfh, option)
            except RuntimeError:
                continue

            if result.fitness > best_fitness:
                best_fitness = result.fitness
                best = result

        if best is None or best.fitness < self.global_fitness_threshold:
            return None

        init_T = best.transformation

        # ---------- ICP (local refinement) ----------
        icp_max_corr = fgr_max_corr * 0.5

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
            return best

        return icp
