import os
import shutil
from pathlib import Path
import torch
from ultralytics import YOLO

# =========================================================
# 1. Configuration & Relative Paths
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

# Relative directory paths for standard GitHub repository structure
SOURCE_DATASET = BASE_DIR / "data_raw"
CLEANED_DATASET = BASE_DIR / "ppe_cleaned"
DATA_YAML_PATH = BASE_DIR / "data_cleaned.yaml"

# Class Mapping Rules (Filtering out unwanted classes, remapping remaining to 0-5)
CLASS_MAPPING = {
    0: 0,   # Helmet
    1: 1,   # Gloves
    2: 2,   # Vest
    3: 3,   # Boots
    4: 4,   # Goggles
    6: 5    # Person
}
REMOVE_CLASSES = {5, 7, 8, 9, 10}


# =========================================================
# 2. Dataset Cleaning & Class Remapping
# =========================================================
def clean_dataset(source_path: Path, dest_path: Path):
    """Filters negative classes and remaps valid class IDs into a destination folder."""
    print("🧹 Cleaning dataset and updating class labels...")
    splits = ["train", "val", "test"]

    for split in splits:
        src_images = source_path / "images" / split
        src_labels = source_path / "labels" / split

        dst_images = dest_path / "images" / split
        dst_labels = dest_path / "labels" / split

        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        if not src_labels.exists():
            print(f"⚠️ Directory missing: {src_labels}. Skipping split: {split}")
            continue

        for label_file in src_labels.glob("*.txt"):
            new_lines = []
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    
                    old_class = int(parts[0])

                    if old_class in REMOVE_CLASSES:
                        continue
                        
                    if old_class in CLASS_MAPPING:
                        parts[0] = str(CLASS_MAPPING[old_class])
                        new_lines.append(" ".join(parts))

            # Save updated labels and copy image if valid annotations remain
            if new_lines:
                with open(dst_labels / label_file.name, "w") as f:
                    f.write("\n".join(new_lines))

                for ext in [".jpg", ".jpeg", ".png"]:
                    img_file = src_images / f"{label_file.stem}{ext}"
                    if img_file.exists():
                        shutil.copy(img_file, dst_images / img_file.name)
                        break

    print(f"✅ Cleaned dataset successfully generated at: {dest_path}")


# =========================================================
# 3. YAML Configuration Generator
# =========================================================
def generate_data_yaml(dest_path: Path, yaml_file: Path):
    """Generates the required data.yaml file for YOLOv8."""
    yaml_content = f"""train: {dest_path / 'images' / 'train'}
val: {dest_path / 'images' / 'val'}
test: {dest_path / 'images' / 'test'}

nc: 6

names:
  0: Helmet
  1: Gloves
  2: Vest
  3: Boots
  4: Goggles
  5: Person
"""
    with open(yaml_file, "w") as f:
        f.write(yaml_content)
    print(f"✅ Generated config file at: {yaml_file}")


# =========================================================
# 4. Model Training & Evaluation
# =========================================================
def train_model():
    # 1. Clean dataset if target folder doesn't exist
    if not CLEANED_DATASET.exists():
        clean_dataset(SOURCE_DATASET, CLEANED_DATASET)

    # 2. Setup configuration file
    generate_data_yaml(CLEANED_DATASET, DATA_YAML_PATH)

    # 3. Initialize YOLOv8 and select hardware device
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"🚀 Starting YOLOv8 Training on Device: {device}")

    model = YOLO("yolov8n.pt")

    model.train(
        data=str(DATA_YAML_PATH),
        epochs=50,
        imgsz=640,
        batch=16,
        workers=2,
        device=device,
        project="runs",
        name="ppe_compliance",
        pretrained=True,
        save=True,
        patience=10
    )

    # 4. Run validation and export best weights to repository root
    best_weights = BASE_DIR / "runs" / "ppe_compliance" / "weights" / "best.pt"
    if best_weights.exists():
        best_model = YOLO(str(best_weights))
        metrics = best_model.val()
        print("📊 Validation Results:", metrics)

        # Copy best.pt to root so predict.py can load it seamlessly
        shutil.copy(best_weights, BASE_DIR / "best.pt")
        print("✅ Saved optimized weights to root directory: best.pt")


if __name__ == "__main__":
    train_model()