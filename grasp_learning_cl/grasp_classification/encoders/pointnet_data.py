"""Data loading helpers for PointNet grasp scoring."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import open3d as o3d
import torch
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl

from grasp_learning_cl.grasp_datagen.datasets import SceneGraspDataHandler


ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_numpy(value: ArrayLike, *, dtype: np.dtype = np.float32) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray) and value.dtype == dtype:
        return value
    return np.asarray(value, dtype=dtype)


def _natural_key(path: Union[str, Path]) -> List[Union[int, str]]:
    text = Path(path).name if not isinstance(path, Path) else path.name
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _identity_collate(batch: Sequence[Dict[str, Any]]) -> Sequence[Dict[str, Any]]:
    return batch


class GraspPointPreprocessorSphere:
    """Crop points around a grasp pose and align them to its local frame."""

    def __init__(self, radius: float = 0.04, max_points: int = 1024, seed: Optional[int] = None) -> None:
        self.radius = float(radius)
        self.max_points = int(max_points)
        self._seed = seed
        self._cpu_generator: Optional[torch.Generator] = None
        if seed is not None:
            self._cpu_generator = torch.Generator(device="cpu")
            self._cpu_generator.manual_seed(seed)

    def __call__(
        self,
        points: ArrayLike,
        normals: ArrayLike,
        pose: ArrayLike,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        pts = self._to_tensor(points, device)
        nrm = self._to_tensor(normals, device)
        transform = self._to_tensor(pose, device).reshape(4, 4)

        if pts.shape[0] != nrm.shape[0]:
            raise ValueError("Points and normals must have the same length.")

        radius_sq = self.radius * self.radius
        center = transform[:3, 3]
        rel = pts - center
        sq_dist = (rel * rel).sum(dim=1)
        mask = sq_dist <= radius_sq

        if not mask.any().item():
            nearest_k = max(1, min(self.max_points, pts.shape[0]))
            idx = sq_dist.topk(nearest_k, largest=False).indices
            mask = torch.zeros_like(mask, dtype=torch.bool)
            mask[idx] = True

        rel_points = rel[mask]
        rel_normals = nrm[mask]

        rotation = transform[:3, :3]
        aligned_points = rel_points @ rotation
        aligned_normals = rel_normals @ rotation

        features = torch.cat([aligned_points, aligned_normals], dim=1)
        features = self._resample(features)
        return features.to(device=pts.device)

    def _to_tensor(self, value: ArrayLike | torch.Tensor, device: Optional[torch.device]) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=torch.float32)
        return torch.as_tensor(value, dtype=torch.float32, device=device)

    def _randperm(self, count: int, device: torch.device) -> torch.Tensor:
        if self._cpu_generator is None:
            return torch.randperm(count, device=device)
        return torch.randperm(count, generator=self._cpu_generator).to(device)

    def _randint(self, high: int, size: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        if self._cpu_generator is None:
            return torch.randint(high, size, device=device)
        return torch.randint(high, size, generator=self._cpu_generator).to(device)

    def _resample(self, features: torch.Tensor) -> torch.Tensor:
        device = features.device
        count = features.shape[0]
        if count >= self.max_points:
            indices = self._randperm(count, device)[: self.max_points]
            return features.index_select(0, indices)

        if count == 0:
            return torch.zeros((self.max_points, 6), dtype=torch.float32, device=device)

        padded = torch.empty((self.max_points, features.shape[1]), dtype=features.dtype, device=device)
        padded[:count] = features
        extra_idx = self._randint(count, (self.max_points - count,), device=device)
        padded[count:] = features.index_select(0, extra_idx)
        return padded


class GraspPointPreprocessorBox(GraspPointPreprocessorSphere):
    """Crop points inside a grasp-aligned bounding box."""

    def __init__(
        self,
        max_points: int = 1024,
        bbox_min: Sequence[float] = (-0.025, -0.08, -0.06),
        bbox_max: Sequence[float] = (0.025, 0.08, 0.06),
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(radius=0.0, max_points=max_points, seed=seed)
        self._bbox_min = torch.as_tensor(bbox_min, dtype=torch.float32)
        self._bbox_max = torch.as_tensor(bbox_max, dtype=torch.float32)
        self._bbox_center = (self._bbox_min + self._bbox_max) * 0.5

    def __call__(
        self,
        points: ArrayLike,
        normals: ArrayLike,
        pose: ArrayLike,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        pts = self._to_tensor(points, device)
        nrm = self._to_tensor(normals, device)
        transform = self._to_tensor(pose, device).reshape(4, 4)

        if pts.shape[0] != nrm.shape[0]:
            raise ValueError("Points and normals must have the same length.")

        rotation = transform[:3, :3]
        center = transform[:3, 3]
        rel = pts - center
        rel_local = rel @ rotation

        bbox_min = self._bbox_min.to(device=pts.device)
        bbox_max = self._bbox_max.to(device=pts.device)
        mask = torch.logical_and(rel_local >= bbox_min, rel_local <= bbox_max).all(dim=1)

        if not mask.any().item():
            center_local = self._bbox_center.to(device=pts.device)
            sq_dist = ((rel_local - center_local) ** 2).sum(dim=1)
            nearest_k = max(1, min(self.max_points, pts.shape[0]))
            idx = sq_dist.topk(nearest_k, largest=False).indices
            mask = torch.zeros_like(mask, dtype=torch.bool)
            mask[idx] = True

        rel_normals = nrm[mask]
        aligned_points = rel_local[mask]
        aligned_normals = rel_normals @ rotation

        features = torch.cat([aligned_points, aligned_normals], dim=1)
        features = self._resample(features)
        return features.to(device=pts.device)


class GraspPointCloudDataset(Dataset):
    """Iterates grasp annotations for preprocessed point clouds."""

    def __init__(self, samples: Sequence[Dict[str, Any]]) -> None:
        if not samples:
            raise ValueError("Expected at least one scene sample.")
        self.samples = samples

        self._pairs: List[Tuple[int, int]] = []
        for scene_idx, sample in enumerate(samples):
            for grasp_idx, _ in enumerate(sample.get("grasps", [])):
                self._pairs.append((scene_idx, grasp_idx))
        if not self._pairs:
            raise ValueError("No grasps found in provided samples.")

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        scene_idx, grasp_idx = self._pairs[index]
        scene = self.samples[scene_idx]
        grasp = scene["grasps"][grasp_idx]

        return {
            "points": scene["points"],
            "normals": scene["normals"],
            "pose": _to_numpy(grasp["pose"]).reshape(4, 4),
            "label": float(grasp["label"]),
            "scene_index": scene_idx,
            "grasp_index": grasp_idx,
            "scene_path": scene.get("scene_path"),
        }


class GraspPointCloudDataModule(pl.LightningDataModule):
    """Construct train/val/test loaders for grasp scoring."""

    def __init__(
        self,
        dataset_root: Optional[Union[str, Path]] = None,
        train_split: float = 0.6,
        val_split: float = 0.2,
        test_split: float = 0.2,
        label_key: str = "label",
        cloud_filename: str = "cloud.ply",
        batch_size: int = 32,
        num_workers: int = 0,
        persistent_workers: bool = True,
    ) -> None:
        self.train_samples = None
        self.val_samples = None
        self.test_samples = None

        self.dataset_root = Path(dataset_root) if dataset_root is not None else None
        self.train_split = float(train_split)
        self.val_split = float(val_split)
        self.test_split = float(test_split)
        self.label_key = label_key
        self.cloud_filename = cloud_filename

        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.persistent_workers = bool(persistent_workers and num_workers > 0)
        self._log_hyperparams = False
        self.allow_zero_length_dataloader_with_multiple_devices = False

        self._train_ds: Optional[GraspPointCloudDataset] = None
        self._val_ds: Optional[GraspPointCloudDataset] = None
        self._test_ds: Optional[GraspPointCloudDataset] = None
        self._loaded = False

    # Lightning hooks -------------------------------------------------
    def setup(self, stage: Optional[str] = None) -> None:  # pragma: no cover - thin wrapper
        if not self._loaded and self.dataset_root is not None:
            self._load_from_disk()
            self._loaded = True

        if stage in (None, "fit"):
            self._train_ds = GraspPointCloudDataset(self.train_samples) if self.train_samples else None
            self._val_ds = GraspPointCloudDataset(self.val_samples) if self.val_samples else None
        if stage in (None, "test"):
            self._test_ds = GraspPointCloudDataset(self.test_samples) if self.test_samples else None

    def train_dataloader(self) -> DataLoader:
        if self._train_ds is None:
            raise RuntimeError("Training split not initialised; call setup() or provide train_samples.")
        return self._build_loader(self._train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if self._val_ds is None:
            raise RuntimeError("Validation split not initialised; call setup() or provide val_samples.")
        return self._build_loader(self._val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if self._test_ds is None:
            raise RuntimeError("Test split not initialised; call setup() or provide test_samples.")
        return self._build_loader(self._test_ds, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    # Internal helpers ------------------------------------------------
    def _build_loader(self, dataset: GraspPointCloudDataset, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers,
            collate_fn=_identity_collate,
        )

    def _load_from_disk(self) -> None:
        assert self.dataset_root is not None
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.dataset_root}")

        scenes = sorted((p for p in self.dataset_root.iterdir() if p.is_dir()), key=_natural_key)
        if not scenes:
            raise ValueError(f"No scene folders found in {self.dataset_root}")

        if self.train_split + self.val_split + self.test_split > 1.0 + 1e-6:
            raise ValueError("Split fractions must sum to <= 1.0")

        total = len(scenes)
        train_end = int(total * self.train_split)
        val_end = train_end + int(total * self.val_split)

        train_paths = scenes[:train_end]
        val_paths = scenes[train_end:val_end]
        test_paths = scenes[val_end:]

        self.train_samples = [sample for path in train_paths if (sample := self._load_scene(path)) is not None]
        self.val_samples = [sample for path in val_paths if (sample := self._load_scene(path)) is not None]
        self.test_samples = [sample for path in test_paths if (sample := self._load_scene(path)) is not None]

    def _load_scene(self, scene_dir: Path) -> Optional[Dict[str, Any]]:
        cloud_path = scene_dir / self.cloud_filename
        grasps_path = scene_dir / "grasps.csv"
        cache_path = scene_dir / "pointnet_cache.npz"

        if not cloud_path.exists() or not grasps_path.exists():
            return None

        sources = [cloud_path, grasps_path]
        if cache_path.exists():
            cache_mtime = cache_path.stat().st_mtime
            if all(cache_mtime >= src.stat().st_mtime for src in sources):
                with np.load(cache_path, allow_pickle=False) as cached:
                    points = cached["points"].astype(np.float32, copy=False)
                    normals = cached["normals"].astype(np.float32, copy=False)
                    poses = cached["poses"].astype(np.float32, copy=False)
                    labels = cached["labels"].astype(np.float32, copy=False)

                grasps = [
                    {"pose": pose.reshape(4, 4), "label": float(label)}
                    for pose, label in zip(poses, labels)
                ]
                if not grasps:
                    return None
                return {
                    "points": points,
                    "normals": normals,
                    "grasps": grasps,
                    "scene_path": str(scene_dir),
                }

        cloud = o3d.io.read_point_cloud(str(cloud_path))
        if cloud.is_empty():
            raise ValueError(f"Empty point cloud: {cloud_path}")
        if not cloud.has_normals():
            cloud.estimate_normals()

        points = np.asarray(cloud.points, dtype=np.float32)
        normals = np.asarray(cloud.normals, dtype=np.float32)
        if normals.shape[0] != points.shape[0]:
            raise ValueError(f"Normals and points mismatch in {cloud_path}")

        entries = SceneGraspDataHandler.from_file(str(grasps_path)).data
        grasps: List[Dict[str, Any]] = []
        poses: List[np.ndarray] = []
        labels: List[float] = []
        for entry in entries:
            if self.label_key not in entry:
                raise KeyError(f"Label '{self.label_key}' missing in {grasps_path}")
            pose = _to_numpy(entry["pose"]).reshape(4, 4).astype(np.float32)
            label = float(entry[self.label_key])
            grasps.append({"pose": pose, "label": label})
            poses.append(pose)
            labels.append(label)
        if not grasps:
            return None

        try:
            np.savez_compressed(
                cache_path,
                points=points.astype(np.float32, copy=False),
                normals=normals.astype(np.float32, copy=False),
                poses=np.asarray(poses, dtype=np.float32),
                labels=np.asarray(labels, dtype=np.float32),
            )
        except OSError:
            cache_path.unlink(missing_ok=True)

        return {
            "points": points,
            "normals": normals,
            "grasps": grasps,
            "scene_path": str(scene_dir),
        }


__all__ = [
    "GraspPointPreprocessorSphere",
    "GraspPointCloudDataset",
    "GraspPointCloudDataModule",
    "_identity_collate",
]
