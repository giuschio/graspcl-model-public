import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import radius

from grasp_learning_cl.grasp_recall.samplers.utils import FarthestSamplerTorch, get_gripper_points_mask, orthognal_grasps


class DefaultSampler:
    """Default grasp sampler used from EdgeGraspNet.

    https://haojhuang.github.io/edge_grasp_page/
    """

    def __init__(self, device, sample_number=32, table_mask=True, table_height=0.054) -> None:
        """
        device: device
        sample_number: how many approach points should be sampled
        table_height: height of the table in the pcd reference frame
        table_mask: if True, filters out the grasps that could be in collision with the table        
        """
        self.device = device
        self.sample_number = sample_number
        self.table_height = table_height
        self.table_masking = table_mask

    @torch.no_grad()
    def __call__(self, cloud) -> Data | None:
        """Build candidate edges and package them into a Data object for scoring."""
        """
        Build candidate edges and package them into a Data object for scoring
        From the paper:
            Grasps are represented as (pa, pc) where pa is the approach point and pc is the contact point (of one of the fingers)
            To constrain the grasp, they use the surface normal at pc

            To sample a grasp, they sample the approach point first. Then, for each approach point multiple contact points
            are sampled within a ball of diameter G (gripper width).
        """
        points = torch.from_numpy(np.asarray(cloud.points)).to(torch.float32).to(self.device)
        normals = torch.from_numpy(np.asarray(cloud.normals)).to(torch.float32).to(self.device)

        fps_sample = FarthestSamplerTorch()
        _, sample = fps_sample(points, self.sample_number)
        sample = torch.as_tensor(sample, device=self.device, dtype=torch.long).reshape(-1)
        sample = torch.unique(sample, sorted=True)

        sample_pos = points[sample, :]

        # Neighbor search (ball query)
        radius_p_batch_index = radius(points, sample_pos, r=0.05, max_num_neighbors=1024)
        radius_p_index = radius_p_batch_index[1, :]
        radius_p_batch = radius_p_batch_index[0, :]

        # Expand sample positions to match neighbor indices
        sample_pos_expanded = torch.cat(
            [sample_pos[i, :].repeat((radius_p_batch == i).sum(), 1) for i in range(len(sample))],
            dim=0,
        )
        sample_index = torch.cat(
            [sample[i].unsqueeze(0).repeat((radius_p_batch == i).sum(), 1) for i in range(len(sample))],
            dim=0,
        )
        edges = torch.cat((sample_index, radius_p_index.unsqueeze(dim=-1)), dim=1)
        all_edge_index = torch.arange(0, len(edges), device=self.device)

        # Local geometry
        des_pos = points[radius_p_index, :]
        des_normals = normals[radius_p_index, :]
        relative_pos = des_pos - sample_pos_expanded
        relative_pos_normalized = F.normalize(relative_pos, p=2, dim=1)

        x_axis = torch.cross(des_normals, relative_pos_normalized, dim=1)
        x_axis = F.normalize(x_axis, p=2, dim=1)

        valid_edge_approach = torch.cross(x_axis, des_normals, dim=1)
        valid_edge_approach = -F.normalize(valid_edge_approach, p=2, dim=1)

        up_dot_mask = torch.einsum(
            "ik,k->i",
            valid_edge_approach,
            torch.tensor([0.0, 0.0, 1.0], device=self.device),
        )

        relative_norm = torch.linalg.norm(relative_pos, dim=-1)
        depth_proj = -torch.sum(relative_pos * valid_edge_approach, dim=-1)

        geometry_mask = torch.logical_and(up_dot_mask > -0.1, relative_norm > 0.003)
        geometry_mask = torch.logical_and(relative_norm < 0.038, geometry_mask)
        depth_proj_mask = torch.logical_and(depth_proj > -0.000, depth_proj < 0.04)
        geometry_mask = torch.logical_and(geometry_mask, depth_proj_mask)

        # Early exit if too few geometric candidates
        if torch.sum(geometry_mask) < 10:
            return None

        # Table/gripper clearance filter via coarse pose synthesis
        pose_candidates = orthognal_grasps(
            geometry_mask,
            depth_proj,
            valid_edge_approach,
            des_normals,
            sample_pos_expanded,
        )

        if self.table_masking == "height" or self.table_masking == True:
            table_grasp_mask = get_gripper_points_mask(pose_candidates, threshold=self.table_height)
            # Apply table clearance mask back onto the candidate set
            geometry_mask[geometry_mask.clone()] = table_grasp_mask
        elif self.table_masking == "autodetect":
            self.table_height = points[:, 2].min()
            table_grasp_mask = get_gripper_points_mask(pose_candidates, threshold=self.table_height)
            # Apply table clearance mask back onto the candidate set
            geometry_mask[geometry_mask.clone()] = table_grasp_mask

        edge_sample_index = all_edge_index[geometry_mask]
        if edge_sample_index.numel() == 0:
            return None

        if edge_sample_index.numel() > 1500:
            perm = torch.randperm(edge_sample_index.numel(), device=self.device)[:1500]
            edge_sample_index = edge_sample_index[perm]
        edge_sample_index, _ = torch.sort(edge_sample_index)

        # Pack everything needed by the model into a Data object
        data = Data(
            pos=points,  # pointcloud (N, 3)
            normals=normals,  # normals (N, 3)
            sample=sample,    # indexes of the sampled grasp centers (self.sample_number, )
            radius_p_index=radius_p_index,   # 
            ball_batch=radius_p_batch,
            ball_edges=edges,
            approaches=valid_edge_approach[edge_sample_index, :],
            reindexes=edge_sample_index,
            relative_pos=relative_pos[edge_sample_index, :],
            depth_proj=depth_proj[edge_sample_index],
        )
        return data.to(self.device)