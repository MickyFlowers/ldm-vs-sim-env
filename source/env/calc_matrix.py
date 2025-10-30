from scipy.spatial.transform import Rotation as R


a = R.from_euler("XYZ", [90, 90, 0], True).as_matrix()
b = R.from_euler("XYZ", [0, -90, 0], True).as_matrix()
c = R.from_euler("XYZ", [45, 0, 0], True).as_matrix()
output = R.from_matrix(a @ b @ c).as_quat()
print(output)
