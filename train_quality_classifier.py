"""
Training Script for MobileNetV2 Frame Quality Classifier
Generates synthetic degradation data (Gaussian blur, bright glare spots, occlusion masks)
from card templates and fine-tunes MobileNetV2 multi-label quality classifier model.

Outputs:
- models/quality_mobilenetv2.pt (PyTorch weights)
- models/quality_mobilenetv2.keras (Optional Keras weights)
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from typing import Tuple, List

# Target output labels: ['blurry', 'glare', 'occluded', 'good']
LABELS = ["blurry", "glare", "occluded", "good"]
NUM_CLASSES = len(LABELS)

def apply_gaussian_blur(image: np.ndarray) -> np.ndarray:
    """Applies heavy Gaussian blur to simulate out-of-focus or moving camera image."""
    kernel_size = np.random.choice([15, 21, 27, 35])
    sigma = float(np.random.uniform(5.0, 10.0))
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def apply_glare_spot(image: np.ndarray) -> np.ndarray:
    """Overlays bright reflective glare spots/blobs simulating flash reflection."""
    result = image.copy().astype(np.float32)
    h, w = image.shape[:2]
    num_spots = np.random.randint(1, 3)

    for _ in range(num_spots):
        center_x = np.random.randint(int(w * 0.2), int(w * 0.8))
        center_y = np.random.randint(int(h * 0.2), int(h * 0.8))
        radius = np.random.randint(int(min(h, w) * 0.15), int(min(h, w) * 0.35))

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        glare_mask = np.clip(1.0 - (dist_from_center / float(radius)), 0, 1)
        glare_mask = np.power(glare_mask, 1.5)[:, :, np.newaxis]

        glare_intensity = np.random.uniform(180, 255)
        result += glare_mask * glare_intensity

    return np.clip(result, 0, 255).astype(np.uint8)

def apply_occlusion_mask(image: np.ndarray) -> np.ndarray:
    """Overlays dark/colored polygonal masks simulating fingers or objects over card regions."""
    result = image.copy()
    h, w = image.shape[:2]

    corner = np.random.choice(["top_left", "top_right", "bottom_left", "bottom_right"])
    color = (int(np.random.randint(30, 100)), int(np.random.randint(30, 100)), int(np.random.randint(30, 100)))

    if corner == "top_left":
        pts = np.array([[0, 0], [int(w * 0.4), 0], [0, int(h * 0.4)]], np.int32)
    elif corner == "top_right":
        pts = np.array([[w, 0], [int(w * 0.6), 0], [w, int(h * 0.4)]], np.int32)
    elif corner == "bottom_left":
        pts = np.array([[0, h], [int(w * 0.4), h], [0, int(h * 0.6)]], np.int32)
    else:
        pts = np.array([[w, h], [int(w * 0.6), h], [w, int(h * 0.6)]], np.int32)

    cv2.fillPoly(result, [pts], color)
    return result

def create_synthetic_base_card() -> np.ndarray:
    """Creates a synthetic high-contrast card canvas if sample image is unavailable."""
    canvas = np.ones((880, 630, 3), dtype=np.uint8) * 240
    cv2.rectangle(canvas, (20, 20), (610, 860), (30, 30, 180), 8)
    cv2.rectangle(canvas, (40, 40), (590, 160), (220, 180, 50), -1)
    cv2.putText(canvas, "Charizard ex", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    cv2.putText(canvas, "HP 330", (450, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (180, 0, 0), 3)
    cv2.rectangle(canvas, (40, 180), (590, 550), (100, 150, 220), -1)
    cv2.putText(canvas, "[POKEMON ARTWORK]", (150, 370), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.rectangle(canvas, (40, 780), (590, 840), (50, 50, 50), -1)
    cv2.putText(canvas, "199/165  *  RARE SIR", (60, 820), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return canvas

def generate_synthetic_dataset(num_samples: int = 150) -> Tuple[List[np.ndarray], np.ndarray]:
    sample_path = "sample_charizard.png"
    if os.path.exists(sample_path):
        base_img = cv2.imread(sample_path)
    else:
        base_img = create_synthetic_base_card()

    base_img = cv2.resize(base_img, (224, 224))
    images = []
    labels = []

    for _ in range(num_samples):
        # 1. Good sample
        noise = np.random.normal(0, 5, base_img.shape).astype(np.float32)
        good_sample = np.clip(base_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        images.append(cv2.cvtColor(good_sample, cv2.COLOR_BGR2RGB))
        labels.append([0.0, 0.0, 0.0, 1.0])

        # 2. Blurry sample
        blurry_sample = apply_gaussian_blur(base_img)
        images.append(cv2.cvtColor(blurry_sample, cv2.COLOR_BGR2RGB))
        labels.append([1.0, 0.0, 0.0, 0.0])

        # 3. Glare sample
        glare_sample = apply_glare_spot(base_img)
        images.append(cv2.cvtColor(glare_sample, cv2.COLOR_BGR2RGB))
        labels.append([0.0, 1.0, 0.0, 0.0])

        # 4. Occluded sample
        occluded_sample = apply_occlusion_mask(base_img)
        images.append(cv2.cvtColor(occluded_sample, cv2.COLOR_BGR2RGB))
        labels.append([0.0, 0.0, 1.0, 0.0])

        # 5. Blurry + Glare
        bg_sample = apply_glare_spot(apply_gaussian_blur(base_img))
        images.append(cv2.cvtColor(bg_sample, cv2.COLOR_BGR2RGB))
        labels.append([1.0, 1.0, 0.0, 0.0])

    return images, np.array(labels, dtype=np.float32)


class SyntheticCardDataset(Dataset):
    def __init__(self, images: List[np.ndarray], labels: np.ndarray):
        self.images = images
        self.labels = labels
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_rgb = self.images[idx]
        tensor_img = self.transform(img_rgb)
        target = torch.tensor(self.labels[idx], dtype=torch.float32)
        return tensor_img, target


class MobileNetV2QualityModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        try:
            from torchvision.models import MobileNet_V2_Weights
            self.backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        except Exception:
            self.backbone = models.mobilenet_v2(pretrained=True)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.backbone(x)


def train_pytorch_model(output_path: str = "models/quality_mobilenetv2.pt") -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print("[train_quality_classifier] Generating synthetic dataset...")
    images, labels = generate_synthetic_dataset(num_samples=100)

    dataset = SyntheticCardDataset(images, labels)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_quality_classifier] Device: {device}")

    model = MobileNetV2QualityModel(num_classes=NUM_CLASSES).to(device)

    # Freeze backbone parameters
    for param in model.backbone.features.parameters():
        param.requires_grad = False

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.backbone.classifier.parameters(), lr=1e-3)

    print("[train_quality_classifier] Fine-tuning MobileNetV2 head...")
    model.train()
    for epoch in range(6):
        total_loss = 0.0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x_batch)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/6 - Loss: {avg_loss:.4f}")

    print(f"[train_quality_classifier] Saving model weights to '{output_path}'...")
    torch.save(model.state_dict(), output_path)
    print("[train_quality_classifier] PyTorch MobileNetV2 training complete!")

if __name__ == "__main__":
    train_pytorch_model()
