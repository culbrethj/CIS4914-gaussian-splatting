import pycolmap
from pathlib import Path

# Setup paths
dataset_path = Path("south-building/images_small") # Folder containing .jpg or .png
output_path = Path("")
output_path.mkdir(exist_ok=True)

database_path = output_path / "database.db"

# 1. Feature Extraction
# Extracts SIFT features from the images
pycolmap.extract_features(database_path, dataset_path)

# 2. Feature Matching
# Exhaustive matching is robust for small/medium datasets
pycolmap.match_exhaustive(database_path)

# 3. Incremental Mapping
# Performs the actual 3D reconstruction
# This returns a dictionary of reconstruction objects (usually just one)
reconstructions = pycolmap.incremental_mapping(database_path, dataset_path, output_path)

# 4. Save and Export
if reconstructions:
    # Save the first (usually only) reconstruction found
    reconstructions[0].write(output_path)
    reconstructions[0].export_PLY("output_cloud.ply")
    print(f"Reconstruction successful! Saved to {output_path}")
else:
    print("No reconstruction could be created.")
