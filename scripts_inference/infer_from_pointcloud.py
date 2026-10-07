"""Run grasp inference on a single-view scene point cloud.

Whenever possible, pass a point cloud that includes the support table. The
continual-learning module uses the complete scene for its internal collision
checks, so retaining the table generally produces safer, better-filtered grasp
proposals.
"""

import argparse
import sys
from pathlib import Path

import open3d as o3d

from grasp_learning_cl.cl_module_probabilistic import ContinualLearningModule
from grasp_learning_cl.visualization import visualize_scene_pcd


def main():
    parser = argparse.ArgumentParser(
        description="Run CL grasp inference on a point cloud.",
        epilog=(
            "The model expects a single-view point cloud. For best collision "
            "filtering, include the support table whenever possible."
        ),
    )
    parser.add_argument("--cl_module_path", default="data/corl_adapted_model_rw/bootstrapped")
    parser.add_argument(
        "--pointcloud_path",
        default="example_assets/scene_1/pointcloud.ply",
        help="Single-view scene point cloud; include the support table when possible for better collision filtering.",
    )
    parser.add_argument("--grasp_score_threshold", type=float, default=0.0)
    parser.add_argument("--visualize", action="store_true", help="Visualize up to 100 predicted grasps.")
    args = parser.parse_args()

    pointcloud = o3d.io.read_point_cloud(args.pointcloud_path)
    cl_module = ContinualLearningModule.load_inplace(args.cl_module_path, device="cuda")
    grasps = cl_module.predict(
        pointcloud_w_table=pointcloud,
        object_type="unknown",
        grasp_score_threshold=args.grasp_score_threshold,
    )
    print(f"predicted_grasps={len(grasps)}")
    if grasps:
        print(f"best_score={grasps[0].get('score')}")
    if args.visualize:
        visualize_scene_pcd(pointcloud, grasps[:100], color_key="score")


if __name__ == "__main__":
    main()
