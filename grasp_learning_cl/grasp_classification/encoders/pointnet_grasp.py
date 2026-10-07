"""PointNet-based grasp scoring model and Lightning module."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
import open3d as o3d 

import pytorch_lightning as pl

from grasp_learning_cl.grasp_classification.encoders.pointnet_data import GraspPointPreprocessorSphere


class PointNetGraspClassifier(nn.Module):
    """PointNet backbone with a detachable classification head."""

    def __init__(self, in_channels: int = 6, feature_dim: int = 256) -> None:
        super().__init__()
        self.feature_dim = feature_dim

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, feature_dim, kernel_size=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
        )
        self.output = nn.Linear(64, 1)

    def forward_features(self, point_features: torch.Tensor) -> torch.Tensor:
        """Return pooled backbone features (pre-classifier)."""
        if point_features.dim() != 3:
            raise ValueError("Expected features shaped (batch, points, channels).")
        x = self.encoder(point_features.transpose(1, 2))
        return torch.max(x, dim=2).values

    def forward_penultimate(self, pooled_features: torch.Tensor) -> torch.Tensor:
        return self.mlp(pooled_features)

    def forward_head(self, pooled_features: torch.Tensor) -> torch.Tensor:
        penultimate = self.forward_penultimate(pooled_features)
        return self.output(penultimate).squeeze(-1)

    def forward(self, point_features: torch.Tensor) -> torch.Tensor:
        pooled = self.forward_features(point_features)
        return self.forward_head(pooled)


class PointNetGraspLightningModule(pl.LightningModule):
    """PyTorch Lightning wrapper around ``PointNetGraspClassifier``."""

    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        class_weight: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = PointNetGraspClassifier()
        self.preprocessor = GraspPointPreprocessorSphere(radius=0.08, max_points=512, seed=0)
        self.lr = lr
        self.weight_decay = float(weight_decay)

        class_weight_normalised = (class_weight or None)
        if isinstance(class_weight_normalised, str):
            class_weight_normalised = class_weight_normalised.lower()
        if class_weight_normalised not in (None, "balanced"):
            raise ValueError("class_weight must be None or 'balanced'.")
        self.class_weight = class_weight_normalised

        self._pos_weight: Optional[float] = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features)

    # ------------------------------------------------------------------
    def training_step(self, batch: Sequence[Dict[str, Any]], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, log_prefix="train")

    def validation_step(self, batch: Sequence[Dict[str, Any]], batch_idx: int) -> None:
        self._shared_step(batch, log_prefix="val")

    def test_step(self, batch: Sequence[Dict[str, Any]], batch_idx: int) -> None:
        self._shared_step(batch, log_prefix="test")

    # ------------------------------------------------------------------
    def preprocess(self, points: Any, normals: Any, pose: Any) -> torch.Tensor:
        """Convenience helper for inference-time preprocessing."""
        features = self.preprocessor(points, normals, pose, device=self.device)
        return features.unsqueeze(0)

    def predict(
        self,
        cloud: o3d.geometry.PointCloud,
        grasps: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if cloud.is_empty():
            raise ValueError("Input point cloud is empty")
        if not cloud.has_normals():
            cloud.estimate_normals()

        points = np.asarray(cloud.points, dtype=np.float32)
        normals = np.asarray(cloud.normals, dtype=np.float32)

        pointsets: List[torch.Tensor] = []
        for grasp in grasps:
            pose = grasp.get("pose")
            if pose is None:
                raise KeyError("Each grasp dict must contain a 'pose' entry")
            pointsets.append(self.preprocessor(points, normals, pose, device=self.device))

        batch = torch.stack(pointsets, dim=0).to(self.device)
        self.eval()
        with torch.no_grad():
            feats_encoder = self.model.forward_features(batch)
            feats_head = self.model.forward_penultimate(feats_encoder)
            logits = self.model.output(feats_head).squeeze(-1)
            scores = torch.sigmoid(logits).cpu().numpy()
            feats_encoder_np = feats_encoder.cpu().numpy().astype(np.float32)
            feats_head_np = feats_head.cpu().numpy().astype(np.float32)

        annotated: List[Dict[str, Any]] = []
        for grasp, score, feat_encoder, feat_head in zip(
            grasps, scores, feats_encoder_np, feats_head_np
        ):
            enriched = dict(grasp)
            enriched["score"] = float(score)
            enriched["feature_encoder"] = feat_encoder
            enriched["feature_head"] = feat_head
            annotated.append(enriched)
        return annotated

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=getattr(self.hparams, "weight_decay", self.weight_decay),
        )

    # Internal helpers -------------------------------------------------
    def on_fit_start(self) -> None:
        super().on_fit_start()
        if self._pos_weight is not None:
            return
        if self.class_weight != "balanced":
            return
        trainer = getattr(self, "trainer", None)
        datamodule = getattr(trainer, "datamodule", None) if trainer is not None else None
        if datamodule is None:
            return
        train_samples = getattr(datamodule, "train_samples", None)
        if not train_samples:
            datamodule.setup("fit")
            train_samples = getattr(datamodule, "train_samples", None)
        if not train_samples:
            return

        positives = 0
        negatives = 0
        for sample in train_samples:
            for grasp in sample.get("grasps", []):
                label = float(grasp.get("label", 0.0))
                if label >= 0.5:
                    positives += 1
                else:
                    negatives += 1

        if positives == 0:
            return
        if negatives == 0:
            self._pos_weight = 1.0
            return
        self._pos_weight = negatives / positives

    def _shared_step(self, batch: Sequence[Dict[str, Any]], *, log_prefix: str) -> torch.Tensor:
        features, labels = self._prepare_batch(batch)
        logits = self(features)
        loss = self._compute_loss(logits, labels)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        acc = (preds == labels).float().mean()
        positives = preds.sum().float()
        true_positives = ((preds == 1.0) & (labels == 1.0)).sum().float()
        predicted_positive = positives
        actual_positive = (labels == 1.0).sum().float()
        precision = true_positives / predicted_positive.clamp(min=1.0)
        recall = true_positives / actual_positive.clamp(min=1.0)

        self.log(f"{log_prefix}/loss", loss, prog_bar=(log_prefix == "train"), batch_size=labels.size(0))
        self.log(f"{log_prefix}/acc", acc, prog_bar=True, batch_size=labels.size(0))
        self.log(f"{log_prefix}/precision", precision, prog_bar=False, batch_size=labels.size(0))
        self.log(f"{log_prefix}/recall", recall, prog_bar=False, batch_size=labels.size(0))
        return loss

    def _prepare_batch(self, batch: Sequence[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.device
        features = [
            self.preprocessor(sample["points"], sample["normals"], sample["pose"], device=device)
            for sample in batch
        ]
        labels = [float(sample["label"]) for sample in batch]
        feature_tensor = torch.stack(features, dim=0)
        label_tensor = torch.tensor(labels, dtype=torch.float32, device=device)
        return feature_tensor, label_tensor

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self._pos_weight is not None:
            pos_weight = torch.tensor(self._pos_weight, dtype=logits.dtype, device=logits.device)
            return nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
        return nn.functional.binary_cross_entropy_with_logits(logits, labels)

    def predict_step(
        self,
        batch: Sequence[Dict[str, Any]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Dict[str, torch.Tensor]:
        features, labels = self._prepare_batch(batch)
        logits = self(features)
        probs = torch.sigmoid(logits)
        return {
            "probs": probs.detach().cpu(),
            "labels": labels.detach().cpu(),
        }


__all__ = [
    "PointNetGraspClassifier",
    "PointNetGraspLightningModule",
]
