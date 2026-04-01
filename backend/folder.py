import os
import shutil
import re

# ===== CONFIG =====
SOURCE_DIR = "datasets/speaker/videos/raw"          # folder containing frame_000000.jpg etc
OUTPUT_DIR = "datasets/speaker/videos/folded"  # output root folder
NUM_FOLDS = 3

# ==================

def main():
    # Create fold directories
    for i in range(NUM_FOLDS):
        os.makedirs(os.path.join(OUTPUT_DIR, f"fold_{i}"), exist_ok=True)

    # Regex to extract frame number
    pattern = re.compile(r"frame_(\d+)\.jpg")

    for filename in sorted(os.listdir(SOURCE_DIR)):
        match = pattern.match(filename)
        if not match:
            continue

        frame_index = int(match.group(1))
        fold_index = frame_index % NUM_FOLDS

        src = os.path.join(SOURCE_DIR, filename)
        dst = os.path.join(OUTPUT_DIR, f"fold_{fold_index}", filename)

        shutil.copy2(src, dst)

        print(f"{filename} -> fold_{fold_index}")

    print("\n✅ Done splitting dataset.")

if __name__ == "__main__":
    main()