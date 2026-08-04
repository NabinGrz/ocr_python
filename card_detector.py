"""
YOLOv8 Card and Region Detector Module
Replaces fragile contour-based cropping and fixed percentage crops with a trained YOLOv8 model.

Target Classes:
- 'card': Full Pokémon card bounding box.
- 'header_region': Top section containing Pokémon name and HP.
- 'id_region': Collector number and set symbol strip at card bottom.
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

class ModelNotAvailableError(Exception):
    """Raised when the YOLO detector model file or ultralytics library is unavailable."""
    pass

@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    label: str

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    def crop(self, image: np.ndarray) -> np.ndarray:
        """Crops ROI from image array using integer pixel coordinates."""
        h, w = image.shape[:2]
        x1_c = max(0, min(self.x1, w - 1))
        y1_c = max(0, min(self.y1, h - 1))
        x2_c = max(0, min(self.x2, w))
        y2_c = max(0, min(self.y2, h))

        if x2_c <= x1_c or y2_c <= y1_c:
            return np.empty((0, 0, 3), dtype=image.dtype)
        return image[y1_c:y2_c, x1_c:x2_c]

class YOLOCardDetector:
    """
    YOLOv8-based Card and Region Detector.
    Identifies 'card', 'header_region', and 'id_region' bounding boxes.
    """
    CLASS_NAMES = ["card", "header_region", "id_region"]

    def __init__(
        self,
        model_path: str = "models/yolov8_card_detector.pt",
        conf_threshold: float = 0.6
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Loads the YOLOv8 model if the weights file exists."""
        if not os.path.exists(self.model_path):
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
        except Exception as e:
            print(f"[YOLOCardDetector] Failed to load model '{self.model_path}': {e}")
            self.model = None

    def is_available(self) -> bool:
        """Returns True if the YOLO model is loaded and ready for inference."""
        return self.model is not None

    def detect_regions(
        self,
        image_np: np.ndarray,
        conf_threshold: Optional[float] = None
    ) -> Dict[str, BoundingBox]:
        """
        Runs object detection on input BGR image frame.

        Returns:
            Dict[str, BoundingBox] mapping class labels ('card', 'header_region', 'id_region')
            to their highest-confidence detection above threshold.

        Raises:
            ModelNotAvailableError: If YOLO model is missing or fails to load.
        """
        if not self.is_available():
            raise ModelNotAvailableError(
                f"YOLO card detector model is unavailable at path '{self.model_path}'."
            )

        if image_np is None or image_np.size == 0:
            raise ValueError("Empty or invalid image frame provided for card detection.")

        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold

        # Run YOLO inference
        results = self.model.predict(image_np, verbose=False, conf=threshold)
        detections: Dict[str, BoundingBox] = {}

        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            coords = box.xyxy[0].cpu().numpy()  # x1, y1, x2, y2
            conf = float(box.conf[0].cpu().numpy())
            cls_idx = int(box.cls[0].cpu().numpy())

            if conf >= threshold and 0 <= cls_idx < len(self.CLASS_NAMES):
                label = self.CLASS_NAMES[cls_idx]
                bbox = BoundingBox(
                    x1=int(coords[0]),
                    y1=int(coords[1]),
                    x2=int(coords[2]),
                    y2=int(coords[3]),
                    confidence=round(conf, 4),
                    label=label
                )

                # Keep the highest-confidence bounding box for each class
                if label not in detections or bbox.confidence > detections[label].confidence:
                    detections[label] = bbox

        return detections

    def warp_and_crop(
        self,
        image_np: np.ndarray,
        detections: Dict[str, BoundingBox]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Crops card region and ROIs according to YOLO bounding box detections.

        Returns:
            Tuple of (warped_card_image, rois_dict) where rois_dict contains 'header' and 'footer' arrays.
        """
        h, w = image_np.shape[:2]

        if "card" in detections:
            card_bbox = detections["card"]
            card_crop = card_bbox.crop(image_np)
            warped = cv2.resize(card_crop, (630, 880))
        else:
            warped = cv2.resize(image_np, (630, 880))

        rois: Dict[str, np.ndarray] = {}
        warped_h, warped_w = warped.shape[:2]

        # Use YOLO detected header ROI if present; otherwise default to upper 18% of warped card
        if "header_region" in detections:
            header_bbox = detections["header_region"]
            # Convert bounding box relative to card crop if card detected
            header_crop = header_bbox.crop(image_np)
            if header_crop.size > 0:
                rois["header"] = header_crop
            else:
                rois["header"] = warped[0:int(warped_h * 0.18), 0:warped_w]
        else:
            rois["header"] = warped[0:int(warped_h * 0.18), 0:warped_w]

        # Use YOLO detected ID region ROI if present; otherwise default to bottom 12% & 25%
        if "id_region" in detections:
            id_bbox = detections["id_region"]
            id_crop = id_bbox.crop(image_np)
            if id_crop.size > 0:
                rois["footer_tight"] = id_crop
                rois["footer"] = id_crop
            else:
                rois["footer_tight"] = warped[int(warped_h * 0.88):warped_h, 0:warped_w]
                rois["footer"] = warped[int(warped_h * 0.75):warped_h, 0:warped_w]
        else:
            rois["footer_tight"] = warped[int(warped_h * 0.88):warped_h, 0:warped_w]
            rois["footer"] = warped[int(warped_h * 0.75):warped_h, 0:warped_w]

        return warped, rois


# Module singleton instance
_detector_instance: Optional[YOLOCardDetector] = None

def get_card_detector(
    model_path: str = "models/yolov8_card_detector.pt",
    conf_threshold: float = 0.6
) -> YOLOCardDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = YOLOCardDetector(model_path=model_path, conf_threshold=conf_threshold)
    return _detector_instance

def detect_regions(
    image_np: np.ndarray,
    model_path: str = "models/yolov8_card_detector.pt",
    conf_threshold: float = 0.6
) -> Dict[str, BoundingBox]:
    """
    Public entry-point function for YOLO card and region detection.

    Args:
        image_np: BGR numpy image frame.
        conf_threshold: Minimum confidence score threshold.

    Returns:
        Dict[str, BoundingBox] mapping labels to bounding box objects.

    Raises:
        ModelNotAvailableError if model file is missing or ultralytics is unavailable.
    """
    detector = get_card_detector(model_path=model_path, conf_threshold=conf_threshold)
    return detector.detect_regions(image_np)
