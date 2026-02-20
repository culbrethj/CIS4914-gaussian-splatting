from pathlib import Path
import os
import cv2
import numpy as np

def preprocessor(raw_images, output_path, duplicate_threshold = 3.0, blur_threshold = 50):
    output_path.mkdir(parents=True, exist_ok=True)

    # gets raw images
    path = Path(raw_images)
    imgs = sorted(os.listdir(path))

    prev = None
    count = 0
    for filename in imgs:
        img_path = path / filename
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        # resize
        height = image.shape[0]
        width = image.shape[1]
        if width > 1280:
            scale = 1280 / width
            new_height = int(height * scale)

            dimension = (1280, new_height)
            resized_image = cv2.resize(image, dimension, interpolation=cv2.INTER_AREA)
        else:
            # want to avoid stretching small images
            resized_image = image

        gray_img = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        # blur detection
        laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
        lap_var = laplacian.var()
        if lap_var < blur_threshold:
            continue

        # drop approx duplicate frames
        downscaled_img = cv2.resize(gray_img, (64, 36), interpolation=cv2.INTER_AREA)
        keep = True
        if prev is not None:
            diff = np.abs(downscaled_img.astype(np.float32) - prev.astype(np.float32))
            mad = diff.mean()
            if mad < duplicate_threshold:
                keep = False

        if keep:
            cv2.imwrite(str(output_path / filename), resized_image)
            prev = downscaled_img
            count += 1
    print(f"{count} frames saved to {output_path}")
    return