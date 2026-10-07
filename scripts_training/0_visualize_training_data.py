"""
Brief description: Visualize saved grasp training scenes and their grasp annotations.

Last modified on: 2026-04-20
Commit message: Some reorganization
"""

import os
import open3d as o3d

from grasp_learning_cl.grasp_datagen.datasets import SceneGraspDataHandler
from grasp_learning_cl.visualization import visualize_scene_pcd

from random import choices

if __name__ == '__main__':
    # Example usage
    sampler = "edge"

    scene_name = f"data_blablabla/scene_data_eval/2026_05_05_starting_snn_model_w_table/scene_000000"
    annotations_file = "data_blablabla/scene_data_eval/bootstrapped_demos/scene_000000/grasps.csv"
    pcd = o3d.io.read_point_cloud(os.path.join('data_blablabla/scene_data_eval/2026_05_05_starting_snn_model_w_table/scene_000000', "cloud_w_table.ply"))

    grasps = SceneGraspDataHandler.from_file(annotations_file).data
    # if sampler is not None:
    #     grasps = [grasp for grasp in grasps if grasp.get("sampler") == sampler]
    # grasps = choices(grasps, k=100)
    visualize_scene_pcd(pcd, grasps, visualize_ref_system=True, color_key="score")
