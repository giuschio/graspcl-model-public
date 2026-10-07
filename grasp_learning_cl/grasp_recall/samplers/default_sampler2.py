import torch

from grasp_learning_cl.grasp_recall.samplers.default_sampler import DefaultSampler
from grasp_learning_cl.grasp_recall.samplers.utils import orthognal_grasps


class DefaultSampler2:
    """Sampler that returns grasp proposals directly in list(dict) format."""

    def __init__(
        self,
        device: str = "cpu",
        sample_number: int = 64,
        table_height: float = 0.054,
        table_mask: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.base_sampler = DefaultSampler(
            device=self.device,
            sample_number=sample_number,
            table_height=table_height,
            table_mask=table_mask,
        )

    @torch.no_grad()
    def __call__(self, cloud) -> list[dict]:
        """
        Sample grasp candidates and return them as SE(3) poses without scoring.
        Useful when only geometric proposals are needed, not classifier scores.
        """
        if cloud is None:
            return []

        data = self.base_sampler(cloud)
        if data is None:
            return []

        num_candidates = len(data.reindexes)
        if num_candidates == 0:
            return []

        depth_proj = data.depth_proj
        approaches = data.approaches
        sample_pos = data.pos[data.ball_edges[:, 0][data.reindexes], :]
        des_normals = data.normals[data.ball_edges[:, 1][data.reindexes], :]

        # Treat every candidate as valid and synthesize gripper poses
        grasp_mask = torch.ones_like(depth_proj, dtype=torch.bool)
        trans_matrix = orthognal_grasps(
            grasp_mask.to(depth_proj.device),
            depth_proj,
            approaches,
            des_normals,
            sample_pos,
        )

        trans_matrix = trans_matrix.detach().cpu().numpy()

        # Retract slightly along gripper -Z to avoid collisions
        z_axis = trans_matrix[:, :3, 2]
        translation = 0.06 * z_axis
        trans_matrix[:, :3, 3] += translation

        centers = trans_matrix[:, :3, 3]
        y_axis = trans_matrix[:, :3, 1]
        point1 = centers + 0.04 * y_axis
        point2 = centers - 0.04 * y_axis

        grasps: list[dict] = []
        for i in range(len(trans_matrix)):
            grasps.append(
                {
                    "pose": trans_matrix[i],
                    "width": 0.08,
                    "point1": point1[i].flatten(),
                    "point2": point2[i].flatten(),
                }
            )
        return grasps
