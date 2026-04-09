import os
from PIL import Image

# ===== SETTINGS =====
input_dir = "datasets/south-building/images_full"        # folder with original JPGs
output_dir = "datasets/south-building/images" # folder to save resized images
max_size = 1200             # longest side in pixels
quality = 95                # JPEG quality (85–95 recommended)
# ====================

os.makedirs(output_dir, exist_ok=True)

def resize_image(input_path, output_path):
    with Image.open(input_path) as img:
        width, height = img.size

        # Skip resize if already small enough
        if max(width, height) <= max_size:
            img.save(output_path, quality=quality)
            return

        scale = max_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        resized = img.resize((new_width, new_height), Image.LANCZOS)
        resized.save(output_path, quality=quality)

for filename in os.listdir(input_dir):
    if filename.lower().endswith((".jpg", ".jpeg")):
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)

        resize_image(in_path, out_path)
        print(f"Resized: {filename}")

print("Done!")
