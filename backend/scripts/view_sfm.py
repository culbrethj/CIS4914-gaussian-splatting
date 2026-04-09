import open3d as o3d
import pycolmap

# 1. Load the model
rec = pycolmap.Reconstruction("datasets/south-building/colmap/sparse/0")

# 2. Extract XYZ coordinates and RGB colors
points = [p.xyz for p in rec.points3D.values()]
colors = [p.color / 255.0 for p in rec.points3D.values()]

# 3. Create the point cloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# 4. Run the visualizer
o3d.visualization.draw_geometries([pcd], window_name="COLMAP Point Cloud")