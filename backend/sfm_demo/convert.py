import pycolmap

# Load your model from the directory containing the .bin files
reconstruction = pycolmap.Reconstruction("south-building/output")

# Export to a PLY file
reconstruction.export_PLY("output_cloud.ply")
print("Point cloud exported as output_cloud.ply")