from scipy.spatial.transform import Rotation as R
import torch
import numpy as np

a = [-torch.pi / 12, -torch.pi / 12, -torch.pi / 12]
b = [0.0, torch.pi / 12, torch.pi / 12]
a_axis = R.from_euler("XYZ", a, degrees=False).as_rotvec()
b_axis = R.from_euler("XYZ", b, degrees=False).as_rotvec()
# print(np.linalg.norm(a_axis - b_axis) * 180 / np.pi)
print(np.linalg.norm(a_axis) * 180 / np.pi)

print(np.linalg.norm(b_axis) * 180 / np.pi)
