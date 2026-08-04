"""
Training Script Scaffolding for YOLOv8 Card & Region Detector
Fine-tunes a pretrained YOLOv8n object detection model on annotated Pokémon card images.

Dataset Structure Expected (YOLO Format):
dataset/
├── card_dataset.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/

Outputs:
- models/yolov8_card_detector.pt
"""

import os
import yaml
from typing import Dict, Any

DATASET_CONFIG = {
    "path": "./dataset",
    "train": "images/train",
    "val": "images/val",
    "names": {
        0: "card",
        1: "header_region",
        2: "id_region"
    }
}

def generate_dataset_yaml(config_path: str = "models/card_dataset.yaml") -> str:
    """Generates YOLOv8 dataset configuration file."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(DATASET_CONFIG, f, default_flow_style=False)
    print(f"[train_card_detector] Saved dataset config to '{config_path}'")
    return config_path

def train_yolo_model(
    epochs: int = 10,
    img_size: int = 640,
    batch_size: int = 8,
    output_model_path: str = "models/yolov8_card_detector.pt"
) -> bool:
    """
    Fine-tunes YOLOv8n model on annotated Pokémon card dataset.
    """
    config_file = generate_dataset_yaml()

    try:
        from ultralytics import YOLO

        print("[train_card_detector] Initializing YOLOv8n model...")
        model = YOLO("yolov8n.pt")

        dataset_images = os.path.join(DATASET_CONFIG["path"], DATASET_CONFIG["train"])
        if not os.path.exists(dataset_images) or len(os.listdir(dataset_images)) == 0:
            print(
                f"[train_card_detector] Warning: Labeled training dataset directory '{dataset_images}' is empty or not found.\n"
                "To train a production model:\n"
                "1. Annotate a few hundred card photos in Roboflow or Label Studio.\n"
                "2. Export in YOLOv8 format into ./dataset/ (images/train, labels/train, images/val, labels/val).\n"
                "3. Re-run train_card_detector.py."
            )
            return False

        print(f"[train_card_detector] Training for {epochs} epochs...")
        results = model.train(
            data=config_file,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            name="yolov8_pokemon_cards"
        )

        # Copy best trained weights to target path
        best_weights = os.path.join("runs", "detect", "yolov8_pokemon_cards", "weights", "best.pt")
        if os.path.exists(best_weights):
            import shutil
            os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
            shutil.copy(best_weights, output_model_path)
            print(f"[train_card_detector] Successfully saved best weights to '{output_model_path}'")
            return True

    except Exception as e:
        print(f"[train_card_detector] Training failed or interrupted: {e}")

    return False

if __name__ == "__main__":
    train_yolo_model()
