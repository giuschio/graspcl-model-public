from __future__ import annotations

import json
import shutil
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import open3d as o3d

from grasp_learning_cl.grasp_datagen.datasets import SceneGraspDataHandler


class GraspRecallModuleRegistrationBase(ABC):
    """Shared base for registration-based grasp recall modules with disk-backed storage."""

    _RAW_CLOUD_FILE = "pointcloud.pcd"
    _DOWN_CLOUD_FILE = "pointcloud_downsampled.pcd"
    _FEATURE_FILE = "features.fpfh"
    _GRASP_FILE = "grasps.csv"
    _META_FILE = "metadata.json"

    def __init__(
        self,
        voxel_size: float = 0.005,
        correspondence_multiplier: float = 1.5,
        fitness_threshold: float = 0.4,
        use_icp_refinement: bool = True,
        data_folder: str | Path | None = None,
    ) -> None:
        self.voxel_size = voxel_size
        self.max_correspondence_distance_global_registration = voxel_size * correspondence_multiplier
        self.global_fitness_threshold = fitness_threshold
        self.use_icp_refinement = bool(use_icp_refinement)
        self.data_folder = Path(data_folder) if data_folder is not None else None
        if self.data_folder is not None:
            self.data_folder.mkdir(parents=True, exist_ok=True)

        self._db: List[Dict[str, Any]] = []
        self.load_from_disk()

    def _prepare_demonstration(
        self,
        pointcloud: o3d.geometry.PointCloud,
        grasps: Optional[List[dict]],
        object_type: Optional[str],
        grasp_type: Optional[str],
    ) -> Tuple[List[dict], Dict[str, Any], o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
        if not isinstance(pointcloud, o3d.geometry.PointCloud):
            raise TypeError("pointcloud must be an Open3D PointCloud")
        if self.data_folder is None:
            raise RuntimeError("storage_dir must be specified to persist demonstrations")

        resolved_object_type = object_type or "unknown"
        resolved_grasp_type = grasp_type or "unknown"
        resolved_grasps = grasps or []

        metadata = {
            "object_type": resolved_object_type,
            "grasp_type": resolved_grasp_type,
        }

        for grasp in resolved_grasps:
            grasp_metadata = grasp.setdefault("metadata", {})
            grasp_metadata["object_type"] = resolved_object_type
            grasp_metadata["grasp_type"] = resolved_grasp_type
            if "feature_encoder" in grasp:
                grasp_metadata["feature_encoder"] = grasp.pop("feature_encoder")

        pcd_down, features = self._compute_features(pointcloud)
        if features is None:
            raise ValueError("Unable to compute features for the provided pointcloud")

        return resolved_grasps, metadata, pcd_down, features

    def _append_entry(
        self,
        pcd_full: o3d.geometry.PointCloud,
        pcd_down: o3d.geometry.PointCloud,
        features: o3d.pipelines.registration.Feature,
        grasps: List[dict],
        metadata: Dict[str, Any],
    ) -> None:
        entry_id = uuid4().hex
        entry = {
            "id": entry_id,
            "metadata": metadata,
            "pcd_full": pcd_full,
            "pcd_down": pcd_down,
            "features": features,
            "grasps": grasps,
        }
        self._save_entry_to_disk(entry)
        self._db.append(entry)

    def predict(
        self,
        pointcloud_no_table: o3d.geometry.PointCloud,
        pointcloud_w_table: o3d.geometry.PointCloud,
        object_type: Optional[str] = None,
        grasp_type: Optional[str] = None,
        debug: bool = False,
        max_matches_per_template: Optional[int] = None,
        max_total_time_ms: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        if not isinstance(pointcloud_no_table, o3d.geometry.PointCloud):
            print("pointcloud_no_table must be an Open3D PointCloud")
            return []
        if not isinstance(pointcloud_w_table, o3d.geometry.PointCloud):
            print("pointcloud_w_table must be an Open3D PointCloud")
            return []

        filtered_entries = self._filter_entries(object_type, grasp_type)
        if not filtered_entries:
            return []
        if max_matches_per_template is None:
            max_matches_per_template = 10

        # Feature matching always happens on the no-table cloud.
        scene_feature_cloud = pointcloud_no_table
        # ICP refinement always happens on the with-table cloud.
        scene_icp_cloud = pointcloud_w_table

        scene_feature_cloud_down, scene_feature_cloud_fpfh = self._compute_features(scene_feature_cloud)
        if scene_feature_cloud_fpfh is None:
            return []

        max_matches_for_entry = max(1, int(max_matches_per_template))
        max_time_per_entry_ms = None if max_total_time_ms is None else float(max_total_time_ms) / len(filtered_entries)
        results: List[Dict[str, Any]] = []
        matches_for_debug: List[o3d.geometry.PointCloud] = []
        for entry in filtered_entries:
            matches = self._find_matches_for_entry(
                entry=entry,
                scene_feature_cloud_down=scene_feature_cloud_down,
                scene_icp_cloud=scene_icp_cloud,
                scene_feature_cloud_fpfh=scene_feature_cloud_fpfh,
                max_matches_per_template=max_matches_for_entry,
                debug=debug,
                max_time_ms=max_time_per_entry_ms,
            )

            for match in matches:
                if debug:
                    transformed = o3d.geometry.PointCloud(entry["pcd_full"])
                    transformed.paint_uniform_color([1.0, 1.0, 0.0])
                    transformed.transform(match.transformation)
                    matches_for_debug.append(transformed)

                for grasp_data in entry["grasps"]:
                    transform = match.transformation
                    pose = np.matmul(transform, grasp_data["pose"])
                    approach_z = -float(pose[2, 2])
                    if approach_z <= -0.1:
                        continue
                    recalled_grasp = {
                        "pose": pose,
                        "point1": (
                            transform[:3, :3] @ np.array(grasp_data["point1"]).T + transform[:3, 3]
                        ).tolist(),
                        "point2": (
                            transform[:3, :3] @ np.array(grasp_data["point2"]).T + transform[:3, 3]
                        ).tolist(),
                        "width": grasp_data["width"],
                        "metadata": deepcopy(grasp_data.get("metadata", {})),
                    }
                    results.append(recalled_grasp)

        if debug and results:
            scene_vis = o3d.geometry.PointCloud(scene_icp_cloud)
            scene_vis.paint_uniform_color([0.5, 0.5, 0.5])
            o3d.visualization.draw_geometries([scene_vis, *matches_for_debug])
        return results

    @abstractmethod
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
        raise NotImplementedError

    def load_from_disk(self) -> None:
        """Populate the in-memory database from the storage directory."""
        self._db = []
        if self.data_folder is None:
            return

        for entry_dir in sorted([d for d in self.data_folder.iterdir() if d.is_dir()]):
            entry = self._load_entry(entry_dir)
            if entry is not None:
                self._db.append(entry)

    def clear(self) -> None:
        """Remove all persisted recall entries and clear the in-memory database."""
        if self.data_folder is not None:
            for path in self.data_folder.iterdir():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        self._db = []

    def _entry_paths(self, entry_id: str) -> Dict[str, Path]:
        if self.data_folder is None:
            raise RuntimeError("storage_dir must be specified to persist demonstrations")
        entry_dir = self.data_folder / entry_id
        return {
            "dir": entry_dir,
            "raw": entry_dir / self._RAW_CLOUD_FILE,
            "down": entry_dir / self._DOWN_CLOUD_FILE,
            "feature": entry_dir / self._FEATURE_FILE,
            "grasps": entry_dir / self._GRASP_FILE,
            "meta": entry_dir / self._META_FILE,
        }

    def _save_entry_to_disk(self, entry: Dict[str, Any]) -> None:
        paths = self._entry_paths(entry["id"])
        paths["dir"].mkdir(parents=True, exist_ok=True)

        success = o3d.io.write_point_cloud(str(paths["raw"]), entry["pcd_full"])
        if not success:
            raise IOError(f"Failed to write pointcloud to {paths['raw']}")

        success = o3d.io.write_point_cloud(str(paths["down"]), entry["pcd_down"])
        if not success:
            raise IOError(f"Failed to write downsampled pointcloud to {paths['down']}")

        success = o3d.io.write_feature(str(paths["feature"]), entry["features"])
        if not success:
            raise IOError(f"Failed to write features to {paths['feature']}")

        SceneGraspDataHandler.data_to_file(self._json_safe(entry["grasps"]), str(paths["grasps"]))
        paths["meta"].write_text(json.dumps(self._json_safe(entry["metadata"]), indent=2))

    def _load_entry(self, entry_dir: Path) -> Optional[Dict[str, Any]]:
        meta_path = entry_dir / self._META_FILE
        down_path = entry_dir / self._DOWN_CLOUD_FILE
        feature_path = entry_dir / self._FEATURE_FILE
        grasps_path = entry_dir / self._GRASP_FILE
        raw_path = entry_dir / self._RAW_CLOUD_FILE

        for path in (meta_path, down_path, feature_path, grasps_path, raw_path):
            if not path.exists():
                return None

        try:
            metadata = json.loads(meta_path.read_text())
            down_cloud = o3d.io.read_point_cloud(str(down_path))
            features = o3d.io.read_feature(str(feature_path))
            grasp_handler = SceneGraspDataHandler.from_file(str(grasps_path))
            grasps = grasp_handler.data
            full_cloud = o3d.io.read_point_cloud(str(raw_path))
        except Exception:
            return None

        return {
            "id": entry_dir.name,
            "metadata": metadata,
            "pcd_full": full_cloud,
            "pcd_down": down_cloud,
            "features": features,
            "grasps": grasps,
        }

    def _filter_entries(
        self,
        object_type: Optional[str],
        grasp_type: Optional[str],
    ) -> List[Dict[str, Any]]:
        if object_type is None and grasp_type is None:
            return list(self._db)

        results: List[Dict[str, Any]] = []
        for entry in self._db:
            metadata = entry["metadata"]
            if object_type is not None and metadata.get("object_type") != object_type:
                continue
            if grasp_type is not None and metadata.get("grasp_type") != grasp_type:
                continue
            results.append(entry)
        return results

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {key: self._json_safe(val) for key, val in value.items()}
        if isinstance(value, list):
            return [self._json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [self._json_safe(item) for item in value]
        return deepcopy(value)

    def _compute_features(
        self, pcd: o3d.geometry.PointCloud
    ) -> Tuple[o3d.geometry.PointCloud, Optional[o3d.pipelines.registration.Feature]]:
        if self.voxel_size is not None and self.voxel_size > 0:
            pcd_down = pcd.voxel_down_sample(self.voxel_size)
        else:
            pcd_down = o3d.geometry.PointCloud(pcd)

        if self.voxel_size < 1e-4:
            raise ValueError("Specified voxel size is too small")

        if len(pcd_down.points) == 0:
            return pcd_down, None

        if not pcd_down.has_normals():
            pcd_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.voxel_size * 2.0, max_nn=30
                )
            )

        try:
            radius_feature = self.voxel_size * 5.0
            fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                pcd_down,
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=radius_feature, max_nn=100
                ),
            )
        except RuntimeError:
            fpfh = None

        return pcd_down, fpfh
