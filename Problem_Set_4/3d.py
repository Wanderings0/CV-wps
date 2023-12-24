import open3d as o3d
import numpy as np
# Read the PLY file
ply = o3d.io.read_point_cloud('result\gtmoebius_pointcloud.ply')

# Invert the z-axis
ply.points = o3d.utility.Vector3dVector(np.asarray(ply.points) * [1, -1, 1])

# Visualize the point cloud
o3d.visualization.draw_geometries([ply])

