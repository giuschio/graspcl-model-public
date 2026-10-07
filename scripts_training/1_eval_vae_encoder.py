#!/usr/bin/env python3
"""Visualise BPSEncoder reconstructions on random samples."""

from __future__ import annotations

"""
Brief description: Evaluate and visualize BPS autoencoder reconstruction quality on grasp crops.

Last modified on: 2026-05-06
Commit message: Minor changes on encoder trainin (choosing w or without table)
"""

import argparse
import random
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import open3d as o3d
import torch

from grasp_learning_cl.grasp_classification.encoders import GraspPointCloudDataModule, BpsSnnAutoencoderLightning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default="data_local/2026_03_18_encoder/snn_ae/lightning_logs/version_24/checkpoints/last.ckpt", help="Path to BPSEncoder checkpoint.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data_local/2026_03_18_small_collection_run"))
    parser.add_argument("--cloud-filename", choices=["cloud_no_table.ply", "cloud_w_table.ply"], default="cloud_no_table.ply")
    parser.add_argument("--label-key", type=str, default="success")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=5, help="How many grasp crops to visualise.")
    return parser.parse_args()


def collect_samples(loader: Iterable[Sequence[dict]], desired: int) -> List[dict]:
    samples: List[dict] = []
    for batch in loader:
        samples.extend(batch)
        if len(samples) >= desired:
            break
    if not samples:
        raise RuntimeError("Dataloader yielded no samples; check dataset paths.")
    random.shuffle(samples)
    return samples[: min(desired, len(samples))]


def as_point_cloud(points: np.ndarray, color: Sequence[float]) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64, copy=False))
    cloud.paint_uniform_color(color)
    return cloud


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.checkpoint}")
    model = BpsSnnAutoencoderLightning.load_from_checkpoint(str(args.checkpoint))
    model.to(device)
    model.eval()
    model.freeze()
    datamodule = GraspPointCloudDataModule(
        dataset_root=str(args.dataset_root),
        label_key=args.label_key,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cloud_filename=args.cloud_filename,
    )

    if args.split in ("train", "val"):
        datamodule.setup("fit")
    if args.split == "test":
        datamodule.setup("test")

    if args.split == "train":
        loader = datamodule.train_dataloader()
    elif args.split == "val":
        loader = datamodule.val_dataloader()
    else:
        loader = datamodule.test_dataloader()

    samples = collect_samples(loader, args.num_samples)

    for idx, sample in enumerate(samples, start=1):
        features = model.preprocessor(sample["points"], sample["normals"], sample["pose"], device=device)
        features = features.unsqueeze(0)
        with torch.no_grad():
            _, reconstruction, encoding = model.encoder(features)

        original_points = features.squeeze(0)[..., :3].cpu().numpy()
        recon_disp = reconstruction.squeeze(0).view(-1, 3).cpu().numpy()

        original_cloud = as_point_cloud(original_points, (0.1, 0.6, 1.0))
        recon_cloud = as_point_cloud(recon_disp, (1.0, 0.4, 0.1))

        print(f"[{idx}/{len(samples)}] Showing original crop.")
        o3d.visualization.draw_geometries([original_cloud], window_name=f"Sample {idx} - Original", width=800, height=600)

        print(f"[{idx}/{len(samples)}] Showing reconstruction.")
        o3d.visualization.draw_geometries([recon_cloud], window_name=f"Sample {idx} - Reconstruction", width=800, height=600)


if __name__ == "__main__":
    main()
