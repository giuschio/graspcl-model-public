"""
Brief description: Build a CL module by wiring encoder, recall, and score-module data for one category.

Last modified on: 2026-05-06
Commit message: Minor changes on encoder trainin (choosing w or without table)
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch

from typing import Sequence, Optional
from tqdm import tqdm
from sklearn.metrics import average_precision_score

from grasp_learning_cl.grasp_classification.encoders import GraspPointCloudDataModule
from grasp_learning_cl.grasp_classification.grasp_score_module_probabilistic import GraspScoreProbabilisticModule
from grasp_learning_cl.cl_module_probabilistic import ContinualLearningModule

def score_module_kwargs(args, temperature: float) -> dict:
    return {
        "temperature": temperature,
        "truncation_distance": args.truncation_distance,
        "offline_max_strength": args.offline_max_strength,
        "online_max_strength": args.online_max_strength,
        "online_scale_factor": args.online_scale_factor,
        "prior_strength": args.prior_strength,
        "prior_mean": args.prior_mean,
        "max_neighbor_candidates": args.max_neighbor_candidates,
        "demo_odds_margin": args.demo_odds_margin,
        "max_single_demo_weight": args.max_single_demo_weight,
        "data_folder": None,
    }


"""
Brief description: Provide utilities for preparing probabilistic score modules from encoded grasps.

Last modified on: 2026-04-20
Commit message: Some reorganization
"""



def prepare_batch(
    batch: Sequence[dict],
    preprocessor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = [
        preprocessor(sample['points'], sample['normals'], sample['pose'], device=device)
        for sample in batch
    ]
    labels = torch.tensor([float(sample['label']) for sample in batch], dtype=torch.float32, device=device)
    return torch.stack(features, dim=0), labels


def collect_embeddings(
    model,
    loader,
    limit: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    embeddings: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    total = 0
    for batch in tqdm(loader, total=len(loader), desc='Collecting embeddings'):
        feats, lbls = prepare_batch(batch, model.preprocessor, model.device)
        with torch.no_grad():
            emb = model.encode(feats)
        embeddings.append(emb.detach().cpu().numpy())
        labels.append(lbls.detach().cpu().numpy())
        total += emb.size(0)
        if limit is not None and total >= limit:
            break

    if not embeddings:
        raise RuntimeError('No samples collected; check dataset path and loaders.')

    feat_arr = np.concatenate(embeddings, axis=0)
    lbl_arr = np.concatenate(labels, axis=0)
    if limit is not None:
        feat_arr = feat_arr[:limit]
        lbl_arr = lbl_arr[:limit]

    return feat_arr.astype(np.float32, copy=False), lbl_arr.astype(np.float32, copy=False)


def get_embeddings(
    encoder,
    data_path: str,
    label_key: str,
    batch_size: int,
    num_workers: int,
    cloud_filename: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    datamodule = GraspPointCloudDataModule(
        dataset_root=data_path,
        label_key=label_key,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        cloud_filename=cloud_filename,
    )
    datamodule.setup("fit")

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    train_features, train_labels = collect_embeddings(encoder, train_loader)
    val_features, val_labels = collect_embeddings(encoder, val_loader)

    return train_features, train_labels, val_features, val_labels


def score_average_precision(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    temperature: float,
    args,
) -> float:
    score_module = GraspScoreProbabilisticModule(
        **score_module_kwargs(args, temperature),
    )
    score_module.fit_offline(
        train_features[:args.n_training_points],
        train_labels[:args.n_training_points],
        persist=False,
    )
    scores = score_module.predict(val_features, estimation_mode="offline")
    return float(average_precision_score(val_labels, scores))


def select_temperature(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    args,
) -> tuple[float, float]:
    temperatures = np.arange(
        args.temperature_min,
        args.temperature_max + 0.5 * args.temperature_step,
        args.temperature_step,
        dtype=np.float32,
    )

    best_temperature = None
    best_average_precision = None
    for temperature in temperatures:
        average_precision = score_average_precision(
            train_features=train_features,
            train_labels=train_labels,
            val_features=val_features,
            val_labels=val_labels,
            temperature=float(temperature),
            args=args,
        )
        print(f"temperature={float(temperature):.2f} val_ap={average_precision:.4f}")
        if best_average_precision is None or average_precision > best_average_precision:
            best_temperature = float(temperature)
            best_average_precision = average_precision

    if best_temperature is None or best_average_precision is None:
        raise RuntimeError("Failed to select a temperature.")

    return best_temperature, best_average_precision


def build_cl_module(
    encoder_path: str,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    temperature: float,
    cloud_filename: str,
    args,
) -> ContinualLearningModule:
    encoder_pointcloud = "with_table" if cloud_filename == "cloud_w_table.ply" else "no_table"
    cl_module = ContinualLearningModule(
        encoder_path=encoder_path,
        device="cuda",
        encoder_pointcloud=encoder_pointcloud,
    )
    score_module = GraspScoreProbabilisticModule(
        **score_module_kwargs(args, temperature),
    )
    score_module.fit_offline(
        train_features[:args.n_training_points],
        train_labels[:args.n_training_points],
        persist=False,
    )
    cl_module.grasp_scoring_module_cl = score_module
    return cl_module


def main(args) -> None:
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    encoder_pointcloud = "with_table" if args.cloud_filename == "cloud_w_table.ply" else "no_table"

    cl_module_for_embeddings = ContinualLearningModule(
        encoder_path=args.encoder_path,
        encoder_pointcloud=encoder_pointcloud,
    )
    train_features, train_labels, val_features, val_labels = get_embeddings(
        encoder=cl_module_for_embeddings.encoder,
        data_path=args.training_data_path,
        label_key="success_binary",
        batch_size=1024,
        num_workers=0,
        cloud_filename=args.cloud_filename,
    )

    print(
        f"Collected embeddings: train={train_features.shape[0]} "
        f"val={val_features.shape[0]} dim={train_features.shape[1]}"
    )

    best_temperature, best_average_precision = select_temperature(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        args=args,
    )
    print(
        f"Selected temperature={best_temperature:.2f} "
        f"with val_ap={best_average_precision:.4f}"
    )

    cl_module = build_cl_module(
        encoder_path=args.encoder_path,
        train_features=train_features,
        train_labels=train_labels,
        temperature=best_temperature,
        cloud_filename=args.cloud_filename,
        args=args,
    )
    cl_module.set_module_root(output_path)
    if args.internalize_encoder:
        internal_encoder_path = cl_module.internalize_encoder()
        print(f"Copied encoder to {internal_encoder_path}")
    cl_module.save(output_path)
    ContinualLearningModule.update_training_metadata(output_path, base_train_object_set="base")
    print(f"Saved CL module to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder_path",
        default="data_local_2/models/2026_04_07_different_embeddings/snn_autoencoder_bootstrapped/lightning_logs/version_0/checkpoints/best.ckpt",
    )
    parser.add_argument(
        "--training_data_path",
        default="data_local_2/training_data/2026_03_25_large_collection_run_single_view",
    )
    parser.add_argument(
        "--output_path",
        default="data_local_2/models/2026_04_23_starting_snn_model_modern",
    )
    parser.add_argument(
        "--internalize_encoder",
        action="store_true",
        help="Copy the encoder checkpoint into the output CL-module directory and store a relative path",
    )
    parser.add_argument("--cloud_filename", choices=["cloud_no_table.ply", "cloud_w_table.ply"], default="cloud_w_table.ply")
    parser.add_argument("--n_training_points", type=int, default=10_000, help="Maximum number of training embeddings used to fit the score module")
    parser.add_argument("--temperature_min", type=float, default=0.2, help="Minimum temperature evaluated during validation")
    parser.add_argument("--temperature_max", type=float, default=0.5, help="Maximum temperature evaluated during validation")
    parser.add_argument("--temperature_step", type=float, default=0.05, help="Step between temperatures evaluated during validation")
    parser.add_argument("--truncation_distance", type=float, default=1.0, help="Maximum embedding distance considered by the score module")
    parser.add_argument("--offline_max_strength", type=float, default=10.0, help="Maximum evidence strength contributed by offline neighbors")
    parser.add_argument("--online_max_strength", type=float, default=1000.0, help="Maximum evidence strength contributed by online neighbors")
    parser.add_argument("--online_scale_factor", type=float, default=3.0, help="Scale applied to online evidence")
    parser.add_argument("--prior_strength", type=float, default=1.0, help="Strength of the Beta prior")
    parser.add_argument("--prior_mean", type=float, default=0.1, help="Mean of the Beta prior")
    parser.add_argument("--max_neighbor_candidates", type=int, default=100, help="Maximum neighbors considered for each prediction")
    parser.add_argument("--demo_odds_margin", type=float, default=1.0, help="Target odds margin assigned to demonstrations")
    parser.add_argument("--max_single_demo_weight", type=float, default=500.0, help="Maximum evidence weight assigned to one demonstration")
    args = parser.parse_args()
    main(args)
