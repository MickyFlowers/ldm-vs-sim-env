import torch
import omni.isaac.lab.utils.math as math_utils


def str_to_color_tuple(color_str):
    color_str = color_str.strip("()")
    color_list = color_str.split(",")
    color_tuple = tuple(int(value) for value in color_list)
    return color_tuple


def trans_matrix_to_pose(trans_matrix):
    position = trans_matrix[:, :3, 3]
    rotation_matrix = trans_matrix[:, :3, :3]
    quaternion = math_utils.quat_from_matrix(rotation_matrix)
    return torch.cat([position, quaternion], dim=1)
