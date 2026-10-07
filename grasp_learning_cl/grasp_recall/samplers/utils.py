import numpy as np
import torch
import torch.nn.functional as F


def orthognal_grasps(geometry_mask, depth_projection, sample_normal, des_normals, sample_pos):
    """
    :param geometry_mask: [bool,bool,,]
    :param depth_projection:
    :param sample_normal:
    :param des_normals:
    :param sample_pos:
    :return: mX4X4 matrices that used to execute grasp in simulation
    """
    # if these is no reasonable points do nothing
    assert sum(geometry_mask) > 0
    depth = depth_projection[geometry_mask]
    # finger depth
    gripper_dis_from_source = (0.072 - 0.007 - depth).unsqueeze(dim=-1)
    z_axis = -sample_normal[geometry_mask]  # todo careful
    y_axis = des_normals[geometry_mask]
    x_axis = torch.cross(y_axis, z_axis, dim=1)
    x_axis = F.normalize(x_axis, p=2, dim=1)
    y_axis = torch.cross(z_axis, x_axis, dim=1)
    y_axis = F.normalize(y_axis, p=2, dim=1)
    gripper_position = gripper_dis_from_source.repeat(1, 3) * (-z_axis) + sample_pos[geometry_mask]
    transform_matrix = torch.cat(
        (
            x_axis.unsqueeze(dim=-1),
            y_axis.unsqueeze(dim=-1),
            z_axis.unsqueeze(dim=-1),
            gripper_position.unsqueeze(dim=-1),
        ),
        dim=-1,
    )
    homo_agument = (
        torch.as_tensor([0.0, 0.0, 0.0, 1.0]).reshape(1, 1, 4).repeat(len(z_axis), 1, 1).to(des_normals.device)
    )
    transform_matrix = torch.cat((transform_matrix, homo_agument), dim=1)
    # transform_matrix = transform_matrix.numpy()
    # print(transform_matrix.shape)

    # flip_trans = torch.as_tensor([[1.0, 0., 0., 0.],
    #                               [0.0, -1., 0., 0.],
    #                               [0.0, 0., -1., 0.],
    #                               [0.0, 0., 0., 1.]])
    # transform_matrix = torch.einsum('nij,jk->nik', transform_matrix, flip_trans)
    return transform_matrix


def get_gripper_points(trans):
    gripper_points_sim = (
        torch.tensor(
            [
                [0, 0, -0.02],
                [0.012, -0.09, 0.015],
                [-0.012, -0.09, 0.015],
                [0.012, 0.09, 0.015],
                [-0.012, 0.09, 0.015],
                [0.005, 0.09, 0.078],
                [0.005, -0.09, 0.078],
            ]
        )
        .to(torch.float)
        .to(trans.device)
    )

    num_p = gripper_points_sim.size(0)
    gripper_points_sim = gripper_points_sim.unsqueeze(dim=0).repeat(len(trans), 1, 1)
    gripper_points_sim = torch.einsum("pij,pjk->pik", trans[:, :3, :3], gripper_points_sim.transpose(1, 2))
    gripper_points_sim = gripper_points_sim.transpose(1, 2)
    gripper_points_sim = gripper_points_sim + trans[:, :3, -1].unsqueeze(dim=1).repeat(1, num_p, 1)
    return gripper_points_sim


def get_gripper_points_mask(trans, threshold):
    gripper_points_sim = get_gripper_points(trans)
    z_value = gripper_points_sim[:, :, -1]
    z_mask = z_value > threshold
    z_mask = torch.all(z_mask, dim=1)
    return z_mask


class FarthestSamplerTorch:
    def __init__(self):
        pass

    def _calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)

    def __call__(self, pts, k):
        index_list = []
        farthest_pts = torch.zeros(k, 3).to(pts.device)
        index = np.random.randint(len(pts))
        farthest_pts[0] = pts[index]
        index_list.append(index)
        distances = self._calc_distances(farthest_pts[0], pts)
        for i in range(1, k):
            index = torch.argmax(distances)
            farthest_pts[i] = pts[index]
            index_list.append(index)
            distances = torch.minimum(distances, self._calc_distances(farthest_pts[i], pts))
        return farthest_pts, index_list

