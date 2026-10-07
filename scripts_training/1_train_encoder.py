#!/usr/bin/env python3
"""
Brief description: Train a BPS SNN autoencoder on grasp crop datasets.

Last modified on: 2026-05-06
Commit message: Minor changes on encoder trainin (choosing w or without table)
"""

"""Train various BPS-based encoders (SNN, SNN+AE, or plain AE)."""

from pathlib import Path
import argparse

import pytorch_lightning as pl
import torch

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from grasp_learning_cl.grasp_classification.encoders import GraspPointCloudDataModule, BpsSnnAutoencoderLightning


"""
The average distance between random points increases as the dimensionality grows.

How to tune the dimensionality of the embedding?
  Train it. 
  Embed some of the training data
  Then, start sampling unit vectors at random
  For each vector and a radius, figure out if it has some neighbors or not
  Print the percentage of vectors without any neighbors
  Should give a measure of how much of the space is empty

"""

"""
To train the encoder:
- train a pure_autoencoder
- then, train a snn_autoencoder, where you bootstrap the weights from the autoencoder you trained beforehand

"""


def build_model(model_type: str) -> pl.LightningModule:
    enmbedding_size = 32
    models_dict = {
        "pure_autoencoder": {"cls_loss": "snn", "lambda": 1.0},
        "pure_snn": {"cls_loss": "snn", "lambda": 0.0},
        "pure_bce": {"cls_loss": "crossentropy", "lambda": 0.0},
        "snn_autoencoder": {"cls_loss": "snn", "lambda": 0.2},
    }
    params = models_dict[model_type]

    return BpsSnnAutoencoderLightning(
            lr=1e-3,
            weight_decay=1e-5,
            temperature=0.2,
            embedding_dim=enmbedding_size,
            num_basis=512,
            hidden_encoder=(512, 256, 128),
            hidden_decoder=(512, 256, 128),
            classification_loss=params["cls_loss"],
            lambda_reconstrution_loss=params["lambda"],
            basis_type="sphere",
        )
    

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_type",
        type=str,
        default="snn_autoencoder",
        help="Choose which encoder variant to train (see models_dict above)",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("data_local_2/models/2026_05_04_embeddings_w_table"))
    parser.add_argument("--dataset_root", type=Path, default=Path("data_local_2/training_data/2026_03_25_large_collection_run_single_view_2"))
    parser.add_argument("--label_key", type=str, default="success_binary")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--cloud_filename", choices=["cloud_no_table.ply", "cloud_w_table.ply"], default="cloud_w_table.ply")
    parser.add_argument(
        "--load_initial_weights", 
        type=str, 
        default=None,
        help="Optional checkpoint used to initialize the encoder weights",
    )
    args = parser.parse_args()

    output_dir = args.output_dir / args.model_type

    output_dir.mkdir(parents=True, exist_ok=True)

    datamodule = GraspPointCloudDataModule(
        dataset_root=str(args.dataset_root),
        label_key=args.label_key,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cloud_filename=args.cloud_filename,
    )

    model = build_model(args.model_type)
    if args.load_initial_weights is not None:
        checkpoint_path = Path(args.load_initial_weights)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False)


    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        filename=f"best",
        save_top_k=1,
        save_last=True,
    )
    early_stopping = EarlyStopping(monitor="val/loss", patience=15, mode="min", verbose=True)

    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        default_root_dir=str(output_dir),
    )

    trainer.fit(model, datamodule=datamodule)

    best_path = checkpoint_callback.best_model_path
    if best_path:
        print(f"Best checkpoint saved at: {best_path}")
    else:
        print("Training finished without saving a monitored checkpoint.")


if __name__ == "__main__":
    main()
