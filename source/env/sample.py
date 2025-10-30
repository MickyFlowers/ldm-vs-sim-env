import torch
import kornia
import omni.isaac.lab.utils.math as math_utils


def sample_coordinate_in_sphere(batch_size, radius_range, u_range, v_range):
    radius_uniform_dist = torch.distributions.Uniform(radius_range[0], radius_range[1])
    u_uniform_dist = torch.distributions.Uniform(u_range[0], u_range[1])
    v_uniform_dist = torch.distributions.Uniform(v_range[0], v_range[1])
    radius = radius_uniform_dist.sample([batch_size])
    u = u_uniform_dist.sample([batch_size])
    v = v_uniform_dist.sample([batch_size])

    x = radius * torch.sin(u) * torch.cos(v)
    y = radius * torch.sin(u) * torch.sin(v)
    z = radius * torch.cos(u)

    point = torch.stack([x, y, z], dim=-1)
    z_axis = -point
    z_axis = z_axis / torch.norm(z_axis, dim=-1, keepdim=True)
    k = (
        torch.tensor([0.0, 0.0, 1.0], device=point.device)
        .unsqueeze(0)
        .repeat(batch_size, 1)
    )
    x_axis = torch.cross(z_axis, k, dim=-1)
    x_axis = x_axis / torch.norm(x_axis, dim=-1, keepdim=True)
    y_axis = torch.cross(z_axis, x_axis, dim=-1)
    rotation_matrix = torch.stack([x_axis, y_axis, z_axis], dim=-1)

    trans_matrix = torch.eye(4).repeat(batch_size, 1, 1)
    trans_matrix[:, :3, :3] = rotation_matrix
    trans_matrix[:, :3, 3] = point
    return trans_matrix


def sample_coordinate_in_cardisian(
    batch_size, pos_lower, pos_upper, rot_lower, rot_upper, convention
):
    pos = torch.tensor(pos_lower) + torch.rand(batch_size, 3) * (
        torch.tensor(pos_upper) - torch.tensor(pos_lower)
    )
    rot = torch.tensor(rot_lower) + torch.rand(batch_size, 3) * (
        torch.tensor(rot_upper) - torch.tensor(rot_lower)
    )
    trans_matrix = torch.eye(4).repeat(batch_size, 1, 1)
    trans_matrix[:, :3, 3] = pos
    trans_matrix[:, :3, :3] = math_utils.matrix_from_euler(rot, convention)
    return trans_matrix
