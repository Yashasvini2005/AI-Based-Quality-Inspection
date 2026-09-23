from pathlib import Path
import random
import shutil

# Original Kaggle dataset
SOURCE_DIR = Path("dataset/datasets/obj_train_data")

# New YOLO dataset
OUTPUT_DIR = Path("dataset/yolo")

# Dataset split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

# Same split every time we run the script
random.seed(42)

# Find all JPG images
images = list(SOURCE_DIR.glob("*.jpg"))

print(f"Found {len(images)} images.")

# Shuffle the images
random.shuffle(images)

# Calculate split positions
total = len(images)

train_end = int(total * TRAIN_RATIO)
val_end = train_end + int(total * VAL_RATIO)

train_images = images[:train_end]
val_images = images[train_end:val_end]
test_images = images[val_end:]

print(f"Training images:   {len(train_images)}")
print(f"Validation images: {len(val_images)}")
print(f"Test images:       {len(test_images)}")


def copy_pairs(image_list, split):
    image_destination = OUTPUT_DIR / "images" / split
    label_destination = OUTPUT_DIR / "labels" / split

    for image_path in image_list:

        # Find matching YOLO annotation
        label_path = image_path.with_suffix(".txt")

        if not label_path.exists():
            print(f"WARNING: Missing label for {image_path.name}")
            continue

        # Copy image
        shutil.copy2(
            image_path,
            image_destination / image_path.name
        )

        # Copy annotation
        shutil.copy2(
            label_path,
            label_destination / label_path.name
        )


# Create the three splits
copy_pairs(train_images, "train")
copy_pairs(val_images, "val")
copy_pairs(test_images, "test")

print("\nDataset preparation complete!")