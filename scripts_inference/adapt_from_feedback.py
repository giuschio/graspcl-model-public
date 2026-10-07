import argparse
import sys
from pathlib import Path

import open3d as o3d

from grasp_learning_cl.cl_module_probabilistic import ContinualLearningModule
from grasp_learning_cl.visualization import visualize_scene_pcd


def ask_yes_no(prompt):
    response = ""
    while response not in {"y", "n"}:
        response = input(prompt).strip().lower()
    return response == "y"


def main():
    parser = argparse.ArgumentParser(description="Adapt a CL module from manual pointcloud feedback.")
    parser.add_argument("--cl_module_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--pointcloud_path", required=True)
    parser.add_argument("--object_type", required=True)
    args = parser.parse_args()

    pointcloud = o3d.io.read_point_cloud(args.pointcloud_path)
    cl_module = ContinualLearningModule.load_copy(args.cl_module_path, args.output_path, device="cuda")
    grasps = cl_module.predict(pointcloud_w_table=pointcloud, object_type=args.object_type)
    if not grasps:
        print("No grasps found.")
        return

    best_grasp = grasps[0]
    visualize_scene_pcd(pointcloud, [best_grasp], color_key="score")
    best_grasp["success"] = float(ask_yes_no("Was this grasp successful (y/n)? "))
    cl_module.add_datapoint(
        pointcloud_w_table=pointcloud,
        grasps=[best_grasp],
        object_type=args.object_type,
    )
    cl_module.save(args.output_path)
    ContinualLearningModule.update_training_metadata(args.output_path, train_object_set=args.object_type)
    print(f"saved_adapted_module={args.output_path}")


if __name__ == "__main__":
    main()
