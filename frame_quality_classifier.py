"""
MobileNetV2 Frame Quality Classifier Module
Replaces raw heuristic blur/glare thresholds with a trained MobileNetV2 multi-label classifier.

Labels evaluated:
- 'blurry': High spatial frequency degradation / out-of-focus capture.
- 'glare': Highlight clipping / over-exposed reflective spots on card surface.
- 'occluded': Finger/hand/object partially obscuring the card region.
- 'good': High quality, well-lit, fully visible card frame suitable for OCR.
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, Optional

class ModelNotAvailableError(Exception):
    """Raised when the requested deep-learning model or file is unavailable."""
    pass

class FrameQualityClassifier:
    """
    MobileNetV2-based Frame Quality Classifier.
    Evaluates input cropped card frames and outputs per-label confidence scores.
    Supports both PyTorch (.pt) and TensorFlow/Keras (.keras) models.
    """
    LABELS = ["blurry", "glare", "occluded", "good"]

    def __init__(
        self,
        pytorch_model_path: str = "models/quality_mobilenetv2.pt",
        keras_model_path: str = "models/quality_mobilenetv2.keras"
    ):
        self.pytorch_model_path = pytorch_model_path
        self.keras_model_path = keras_model_path
        self.model = None
        self.backend = None  # 'pytorch' or 'keras'
        self.device = None
        self._model_loaded = False

    def _load_model(self) -> None:
        """Loads available PyTorch or Keras model weights."""
        if self._model_loaded:
            return
        self._model_loaded = True
        # 1. Try PyTorch model first
        if os.path.exists(self.pytorch_model_path):
            try:
                import torch
                import torch.nn as nn
                import torchvision.models as models

                class MobileNetV2QualityModel(nn.Module):
                    def __init__(self, num_classes=4):
                        super().__init__()
                        # The checkpoint contains the complete backbone. Avoid a
                        # redundant network download before loading local weights.
                        self.backbone = models.mobilenet_v2(weights=None)
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

                model = MobileNetV2QualityModel(num_classes=len(self.LABELS))
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                state_dict = torch.load(self.pytorch_model_path, map_location=device)
                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()

                self.model = model
                self.device = device
                self.backend = "pytorch"
                return
            except Exception as e:
                print(f"[FrameQualityClassifier] Failed to load PyTorch model: {e}")

        # 2. Try Keras model fallback
        if os.path.exists(self.keras_model_path):
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(self.keras_model_path)
                self.backend = "keras"
                return
            except Exception as e:
                print(f"[FrameQualityClassifier] Failed to load Keras model: {e}")

    def is_available(self) -> bool:
        """Returns True if a model is loaded and ready for inference."""
        if not self._model_loaded:
            self._load_model()
        return self.model is not None

    def preprocess_image_pytorch(self, image_np: np.ndarray):
        import torch
        import torchvision.transforms as T

        rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)

        transform = T.Compose([
            T.ToTensor(),  # Convert HWC uint8 [0, 255] -> CHW float32 [0.0, 1.0]
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        tensor = transform(resized).unsqueeze(0)  # Add batch dimension (1, 3, 224, 224)
        return tensor.to(self.device)

    def preprocess_image_keras(self, image_np: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=0)

    def classify(self, cropped_card_image: np.ndarray) -> Dict[str, float]:
        """
        Classifies frame quality of cropped card image.
        Returns dictionary mapping label names ('blurry', 'glare', 'occluded', 'good') to confidence float scores.

        Raises:
            ModelNotAvailableError: If no model weights are loaded.
        """
        if not self.is_available():
            raise ModelNotAvailableError(
                f"MobileNetV2 quality model is not available at '{self.pytorch_model_path}' or '{self.keras_model_path}'."
            )

        if cropped_card_image is None or cropped_card_image.size == 0:
            raise ValueError("Empty or invalid image provided for quality classification.")

        if self.backend == "pytorch":
            import torch
            with torch.no_grad():
                tensor_input = self.preprocess_image_pytorch(cropped_card_image)
                outputs = self.model(tensor_input)[0].cpu().numpy()
            predictions = outputs
        elif self.backend == "keras":
            batch_input = self.preprocess_image_keras(cropped_card_image)
            predictions = self.model.predict(batch_input, verbose=0)[0]
        else:
            raise ModelNotAvailableError("No supported deep-learning backend active.")

        result = {}
        for idx, label in enumerate(self.LABELS):
            score = float(predictions[idx]) if idx < len(predictions) else 0.0
            result[label] = round(score, 4)

        return result


# Module singleton instance
_classifier_instance: Optional[FrameQualityClassifier] = None

def get_classifier_instance(
    pytorch_model_path: str = "models/quality_mobilenetv2.pt",
    keras_model_path: str = "models/quality_mobilenetv2.keras"
) -> FrameQualityClassifier:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = FrameQualityClassifier(
            pytorch_model_path=pytorch_model_path,
            keras_model_path=keras_model_path
        )
    return _classifier_instance

def classify_frame_quality(
    cropped_card_image: np.ndarray,
    pytorch_model_path: str = "models/quality_mobilenetv2.pt",
    keras_model_path: str = "models/quality_mobilenetv2.keras"
) -> Dict[str, float]:
    """
    Public entry-point function for frame quality classification.

    Args:
        cropped_card_image: BGR numpy image array of cropped card.

    Returns:
        Dict[str, float] with keys 'blurry', 'glare', 'occluded', 'good'.

    Raises:
        ModelNotAvailableError if model file does not exist or fails to load.
    """
    classifier = get_classifier_instance(
        pytorch_model_path=pytorch_model_path,
        keras_model_path=keras_model_path
    )
    return classifier.classify(cropped_card_image)
