
import open3d as o3d
# 加载 mesh
mesh = o3d.io.read_triangle_mesh("power_bank_cabinet/rotated_mesh.ply")

# 可视化
o3d.visualization.draw_geometries([mesh])
