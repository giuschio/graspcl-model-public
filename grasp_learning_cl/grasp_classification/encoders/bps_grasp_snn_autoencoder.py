"""BPS grasp encoder with soft NN and reconstruction losses."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import open3d as o3d
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from grasp_learning_cl.grasp_classification.encoders.pointnet_data import GraspPointPreprocessorSphere, GraspPointPreprocessorBox

from scipy.stats import qmc

def sample_basis_lhc(
    num_basis: int,
    bbox_min: Tuple[float, float, float] = (-0.1, -0.1, -0.1),
    bbox_max: Tuple[float, float, float] = (0.1, 0.1, 0.1),
    seed: int = 0,
) -> torch.Tensor:
    """Generate basis points with Latin hypercube sampling inside the given bounds."""
    if num_basis <= 0:
        raise ValueError("num_basis must be positive")
    sampler = qmc.LatinHypercube(d=3, seed=seed)
    samples = sampler.random(n=num_basis)
    min_arr = np.array(bbox_min, dtype=np.float32)
    max_arr = np.array(bbox_max, dtype=np.float32)
    basis_np = qmc.scale(samples, min_arr, max_arr)
    return torch.from_numpy(basis_np.astype(np.float32))


def sample_basis_sphere(
    num_basis: int,
    radius: float = 0.1,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    seed: int = 0,
) -> torch.Tensor:
    """Sample basis points uniformly inside a sphere."""
    if num_basis <= 0:
        raise ValueError("num_basis must be positive")
    rng = np.random.default_rng(seed)

    # Sample directions uniformly on the sphere using normal distribution
    direction = rng.normal(size=(num_basis, 3))
    norms = np.linalg.norm(direction, axis=1, keepdims=True)
    direction /= np.clip(norms, a_min=1e-12, a_max=None)

    # Sample radii so that points are uniform inside the sphere
    radii = rng.random(num_basis) ** (1.0 / 3.0) * radius
    points = direction * radii[:, None]
    points += np.asarray(center, dtype=np.float32)[None, :]
    return torch.from_numpy(points.astype(np.float32))


def bps_encode(points: torch.Tensor, basis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode point clouds by nearest-point displacement and nearest-point normals."""
    coords = points[..., :3]
    dists = torch.cdist(coords, basis.unsqueeze(0))
    indices = torch.argmin(dists, dim=1)
    gather_idx = indices.unsqueeze(-1).expand(-1, -1, coords.size(-1))
    closest = torch.gather(coords, 1, gather_idx)
    displacements = closest - basis.unsqueeze(0)
    if points.size(-1) < 6:
        raise ValueError("Expected point features with normals.")
    normals = points[..., 3:6]
    normal_idx = indices.unsqueeze(-1).expand(-1, -1, 3)
    closest_normals = torch.gather(normals, 1, normal_idx)
    return displacements.flatten(start_dim=1), closest_normals.flatten(start_dim=1)


class SoftNearestNeighborLoss(nn.Module):
    """Supervised soft nearest-neighbour loss on normalized embeddings."""

    def __init__(self, tau: float = 0.1, ignore_flag: int = -1) -> None:
        super().__init__()
        self.tau = float(tau)
        self.ignore_flag = int(ignore_flag)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.view(-1)
        labeled = labels != self.ignore_flag
        if not labeled.any():
            return torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype, requires_grad=True)

        z = embeddings[labeled]
        dist = torch.cdist(z, z, p=2)
        sim = -dist
        eye = torch.eye(sim.size(0), device=z.device, dtype=torch.bool)
        labels = labels[labeled].view(-1, 1)
        pos_mask = (labels == labels.t()) & (~eye)

        exp_sim = torch.exp(sim / self.tau)
        denom = exp_sim.masked_fill(eye, 0.0).sum(dim=1)
        num = (exp_sim * pos_mask.float()).sum(dim=1)

        valid = pos_mask.sum(dim=1) > 0
        if not valid.any():
            # Return zero with gradient for batches lacking positive pairs.
            return torch.tensor(0.0, device=z.device, dtype=z.dtype, requires_grad=True)
        loss = -torch.log((num[valid] + 1e-6) / (denom[valid] + 1e-6))
        return loss.mean()


class MaskedBCELoss(nn.Module):
    """Binary cross-entropy loss that ignores unlabeled samples."""

    def __init__(self, ignore_flag: int = -1) -> None:
        super().__init__()
        self.ignore_flag = int(ignore_flag)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.view(-1)
        labeled = labels != self.ignore_flag
        if not labeled.any():
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype, requires_grad=True)
        return F.binary_cross_entropy_with_logits(logits[labeled], labels[labeled].float())


class BpsSnnAutoencoder(nn.Module):
    """Encode grasp crops with displacement BPS features, MLP, and mirror decoder."""

    def __init__(
        self,
        basis: torch.Tensor,
        num_basis: int = 512,
        embedding_dim: int = 64,
        hidden_encoder: Sequence[int] = (512, 256, 128),
        hidden_decoder: Sequence[int] = (128, 256, 512),
        dropout: float = 0.0,
        use_normals: bool = True,
    ) -> None:
        super().__init__()
        self.register_buffer("basis", basis)
        self.num_basis = num_basis
        self.use_normals = bool(use_normals)

        encoder_layers: List[nn.Module] = []
        last_dim = num_basis * (6 if self.use_normals else 3)
        for hidden in hidden_encoder:
            encoder_layers.append(nn.Linear(last_dim, hidden))
            encoder_layers.append(nn.BatchNorm1d(hidden))
            encoder_layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                encoder_layers.append(nn.Dropout(p=dropout))
            last_dim = hidden
        encoder_layers.append(
            nn.Linear(last_dim if encoder_layers else num_basis * (6 if self.use_normals else 3), embedding_dim)
        )
        self.encoder_mlp = nn.Sequential(*encoder_layers)

        decoder_layers: List[nn.Module] = []
        last_dim = embedding_dim
        for hidden in hidden_decoder:
            decoder_layers.append(nn.Linear(last_dim, hidden))
            decoder_layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                decoder_layers.append(nn.Dropout(p=dropout))
            last_dim = hidden
        decoder_layers.append(nn.Linear(last_dim if decoder_layers else embedding_dim, num_basis * 3))
        self.decoder_mlp = nn.Sequential(*decoder_layers)

    def forward(self, point_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        SCALE_FACTOR = 50.
        bps_encoding, bps_normals = bps_encode(point_features, self.basis)

        encoder_input = bps_encoding*SCALE_FACTOR if not self.use_normals else torch.cat([bps_encoding*SCALE_FACTOR, bps_normals], dim=1)
        # print(encoder_input[0, :6], encoder_input[0, -6:])
        latent_embedding = self.encoder_mlp(encoder_input * SCALE_FACTOR)

        latent_embedding = F.normalize(latent_embedding, dim=1)
        # norm = torch.norm(latent_embedding, dim=1, keepdim=True)
        # max_norm = 1.0
        # # Scale down only if norm exceeds max_norm
        # scale = torch.clamp(max_norm / (norm + 1e-8), max=1.0)
        # latent_embedding = latent_embedding * scale

        reconstruction = self.decoder_mlp(latent_embedding).reshape(point_features.size(0), self.num_basis, 3) / SCALE_FACTOR

        target_points = bps_encoding.reshape(point_features.size(0), self.num_basis, 3)
        basis = self.basis.unsqueeze(0)
        reconstruction = reconstruction + basis
        target_points = target_points + basis
        return latent_embedding, reconstruction, target_points


# class PointnetSnnAutoencoder(nn.Module):
#     """PointNet-style autoencoder over grasp-aligned point features."""

#     def __init__(
#         self,
#         in_channels: int = 6,
#         num_points: int = 512,
#         embedding_dim: int = 64,
#         pointnet_hidden: Sequence[int] = (64, 128, 256),
#         decoder_hidden: Sequence[int] = (256, 512, 1024),
#         dropout: float = 0.0,
#     ) -> None:
#         super().__init__()
#         self.in_channels = int(in_channels)
#         self.num_points = int(num_points)

#         encoder_layers: List[nn.Module] = []
#         last_channels = self.in_channels
#         for hidden in pointnet_hidden:
#             encoder_layers.append(nn.Conv1d(last_channels, hidden, kernel_size=1))
#             encoder_layers.append(nn.BatchNorm1d(hidden))
#             encoder_layers.append(nn.ReLU(inplace=True))
#             last_channels = hidden
#         self.point_encoder = nn.Sequential(*encoder_layers)
#         self.embedding_head = nn.Linear(last_channels, embedding_dim)

#         decoder_layers: List[nn.Module] = []
#         last_dim = embedding_dim
#         for hidden in decoder_hidden:
#             decoder_layers.append(nn.Linear(last_dim, hidden))
#             decoder_layers.append(nn.ReLU(inplace=True))
#             if dropout > 0:
#                 decoder_layers.append(nn.Dropout(p=dropout))
#             last_dim = hidden
#         decoder_layers.append(nn.Linear(last_dim, self.num_points * 3))
#         self.decoder_mlp = nn.Sequential(*decoder_layers)

#     def forward(self, point_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         x = point_features.transpose(1, 2)
#         encoded_points = self.point_encoder(x)
#         pooled = torch.max(encoded_points, dim=2).values
#         latent_embedding = self.embedding_head(pooled)

#         # latent_embedding = F.normalize(latent_embedding, dim=1)

#         # Compute L2 norm
#         norm = torch.norm(latent_embedding, dim=1, keepdim=True)
#         max_norm = 1.0
#         # Scale down only if norm exceeds max_norm
#         scale = torch.clamp(max_norm / (norm + 1e-8), max=1.0)
#         latent_embedding = latent_embedding * scale

        
#         reconstruction = self.decoder_mlp(latent_embedding).reshape(point_features.size(0), self.num_points, 3)
#         target_points = point_features[..., :3]
#         return latent_embedding, reconstruction, target_points


class BpsSnnAutoencoderLightning(pl.LightningModule):
    """Lightning module that trains the BPS encoder with soft-NN and reconstruction losses.
       The loss is calculated as (1-lambda_loss)*ssn_loss + lambda_loss*reconstruction_loss
       lambda_loss = 0.0 -> pure softneighborloss
       lambda_loss = 1.0 -> pure autoencoder

       ssn_loss and reconstruction_loss are aggregated with the mean
       a good value for lambda_loss for the snn + reconstruction is 0.2
    
    """

    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        temperature: float = 0.1,
        embedding_dim: int = 64,
        num_basis: int = 512,
        hidden_encoder: Sequence[int] = (512, 256, 128),
        hidden_decoder: Sequence[int] = (128, 256, 512),
        classification_loss: str = "snn",
        lambda_reconstrution_loss: Union[float, Dict[str, Any]] = 0.2,
        feature_key: str = "feature_encoder",
        basis_type: str = "sphere",
        use_normals: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        if basis_type == "sphere":
            sphere_rad = 0.055
            basis = sample_basis_sphere(num_basis, radius=sphere_rad)
            self.preprocessor = GraspPointPreprocessorSphere(radius=sphere_rad, max_points=512, seed=0)
        elif basis_type == "box":
            bbox_min = (-0.025, -0.06, -0.03)
            bbox_max = (0.025, 0.06, 0.03)
            basis = sample_basis_lhc(num_basis, bbox_min=bbox_min, bbox_max=bbox_max)
            self.preprocessor = GraspPointPreprocessorBox(
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                max_points=512,
                seed=0,
            )
        else:
            raise ValueError(f"Unsupported basis_type: {basis_type}")

        self.encoder = BpsSnnAutoencoder(
            basis=basis,
            num_basis=num_basis,
            embedding_dim=embedding_dim,
            hidden_encoder=hidden_encoder,
            hidden_decoder=hidden_decoder,
            use_normals=use_normals,
        )
        self.embedding_dim = embedding_dim

        self.snn_loss = SoftNearestNeighborLoss(tau=temperature)
        if classification_loss not in {"snn", "crossentropy"}:
            raise ValueError(
                f"Unsupported classification_loss: {classification_loss}. "
                "Expected one of {'snn', 'crossentropy'}."
            )
        self.classification_loss = classification_loss
        self.classifier_head = nn.Linear(embedding_dim, 1) if classification_loss == "crossentropy" else None
        self.bce_loss = MaskedBCELoss()
        self.lambda_reconstrution_loss = lambda_reconstrution_loss
        self.lambda_loss = self._get_lambda_loss()
        self.feature_key = feature_key
        self.lr = lr
        self.weight_decay = weight_decay
        self._last_bps_debug_epoch: Optional[int] = None

    def forward(self, point_features: torch.Tensor) -> torch.Tensor:
        latent_embedding, _, _ = self.encoder(point_features)
        return latent_embedding

    def training_step(self, batch: Sequence[Dict[str, Any]], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: Sequence[Dict[str, Any]], batch_idx: int) -> None:
        self._shared_step(batch, stage="val")

    def test_step(self, batch: Sequence[Dict[str, Any]], batch_idx: int) -> None:
        self._shared_step(batch, stage="test")

    def on_train_epoch_start(self) -> None:
        self.lambda_loss = self._get_lambda_loss()

    def preprocess(self, points: Any, normals: Any, pose: Any) -> torch.Tensor:
        return self.preprocessor(points, normals, pose, device=self.device).unsqueeze(0)

    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        embedding, _, _ = self.encoder(batch)
        return embedding
    
    def _shared_step(self, batch: Sequence[Dict[str, Any]], stage: str) -> torch.Tensor:
        features, labels = self._prepare_batch(batch)
        latent_embeddings, reconstruction_points, target_points = self.encoder(features)
        clf_loss = self._compute_classification_loss(latent_embeddings, labels)
        point_dists = torch.linalg.norm(reconstruction_points - target_points, dim=-1)
        rec = point_dists.mean() * 100.0  # loss is in centimeters
        clf_weighted = (1 - self.lambda_loss) * clf_loss
        rec_weighted = self.lambda_loss * rec
        loss = clf_weighted + rec_weighted
        show_on_prog_bar = stage == "val"

        self.log(f"{stage}/loss", loss, prog_bar=show_on_prog_bar, batch_size=labels.size(0))
        if self.classification_loss == "snn":
            self.log(f"{stage}/snn", clf_loss, prog_bar=show_on_prog_bar, batch_size=labels.size(0))
        else:
            self.log(f"{stage}/bce", clf_loss, prog_bar=show_on_prog_bar, batch_size=labels.size(0))
        self.log(f"{stage}/rec", rec, prog_bar=show_on_prog_bar, batch_size=labels.size(0))
        return loss

    def _get_lambda_loss(self) -> float:
        if isinstance(self.lambda_reconstrution_loss, dict):
            if self.lambda_reconstrution_loss["type"] == "constant":
                return self.lambda_reconstrution_loss["value"]
            if self.lambda_reconstrution_loss["type"] == "linear":
                start_epoch = self.lambda_reconstrution_loss["start_epoch"]
                end_epoch = self.lambda_reconstrution_loss["end_epoch"]
                start = self.lambda_reconstrution_loss["start"]
                end = self.lambda_reconstrution_loss["end"]
                if self.current_epoch < start_epoch:
                    return start
                if self.current_epoch > end_epoch:
                    return end
                return start + (end - start) * (self.current_epoch - start_epoch) / (end_epoch - start_epoch)
        return self.lambda_reconstrution_loss

    def _compute_classification_loss(
        self,
        latent_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if self.classification_loss == "snn":
            return self.snn_loss(latent_embeddings, labels.long())

        assert self.classifier_head is not None
        logits = self.classifier_head(latent_embeddings).squeeze(-1)
        return self.bce_loss(logits, labels)

    def predict(
        self,
        cloud: o3d.geometry.PointCloud,
        grasps: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if len(grasps) == 0:
            return []
        if cloud.is_empty():
            raise ValueError("Input point cloud is empty")
        if not cloud.has_normals():
            cloud.estimate_normals()

        points = np.asarray(cloud.points, dtype=np.float32)
        normals = np.asarray(cloud.normals, dtype=np.float32)

        processed: List[torch.Tensor] = []
        for grasp in grasps:
            pose = grasp.get("pose")
            if pose is None:
                raise KeyError("Each grasp dict must contain a 'pose' entry")
            processed.append(self.preprocessor(points, normals, pose, device=self.device))

        batch = torch.stack(processed, dim=0).to(self.device)
        self.eval()
        with torch.no_grad():
            latent_embeddings, _, _ = self.encoder(batch)
            latent_embeddings_np = latent_embeddings.cpu().numpy().astype(np.float32)

        annotated: List[Dict[str, Any]] = []
        for grasp, embed in zip(grasps, latent_embeddings_np):
            enriched = dict(grasp)
            assert len(embed) == self.embedding_dim
            enriched[self.feature_key] = embed
            annotated.append(enriched)
        return annotated

    def predict_proba(
        self,
        cloud: o3d.geometry.PointCloud,
        grasps: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if self.classification_loss != "crossentropy":
            raise ValueError(
                "predict_proba is only available when classification_loss is 'crossentropy'."
            )
        if len(grasps) == 0:
            return []
        if cloud.is_empty():
            raise ValueError("Input point cloud is empty")
        if not cloud.has_normals():
            cloud.estimate_normals()

        points = np.asarray(cloud.points, dtype=np.float32)
        normals = np.asarray(cloud.normals, dtype=np.float32)

        processed: List[torch.Tensor] = []
        for grasp in grasps:
            pose = grasp.get("pose")
            if pose is None:
                raise KeyError("Each grasp dict must contain a 'pose' entry")
            processed.append(self.preprocessor(points, normals, pose, device=self.device))

        batch = torch.stack(processed, dim=0).to(self.device)
        self.eval()
        with torch.no_grad():
            latent_embeddings, _, _ = self.encoder(batch)
            assert self.classifier_head is not None
            logits = self.classifier_head(latent_embeddings).squeeze(-1)
            scores = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
            latent_embeddings_np = latent_embeddings.cpu().numpy().astype(np.float32)

        annotated: List[Dict[str, Any]] = []
        for grasp, score, embed in zip(grasps, scores, latent_embeddings_np):
            enriched = dict(grasp)
            assert len(embed) == self.embedding_dim
            enriched[self.feature_key] = embed
            enriched["score"] = float(score)
            annotated.append(enriched)
        return annotated

    def _prepare_batch(self, batch: Sequence[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.device
        features = [
            self.preprocessor(sample["points"], sample["normals"], sample["pose"], device=device)
            for sample in batch
        ]
        labels = torch.tensor([float(sample.get("label", -1)) for sample in batch], device=device)
        return torch.stack(features, dim=0), labels

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
