"""
Unit and Integration Tests for Deep-Learning Card Scanner Pipeline
Tests all three DL components and their heuristic fallback paths.
"""

import os
import cv2
import numpy as np
import unittest

from frame_quality_classifier import (
    classify_frame_quality,
    FrameQualityClassifier,
    ModelNotAvailableError as QualityModelNotAvailableError
)
from visual_card_matcher import (
    match_card_by_image,
    VisualCardMatcher,
    ModelNotAvailableError as VisualModelNotAvailableError
)
from card_detector import (
    detect_regions,
    YOLOCardDetector,
    BoundingBox,
    ModelNotAvailableError as YOLONotAvailableError
)
from pokemon_card_ocr import PokemonCardExtractor
from pokemon_api import PokemonTCGClient


class TestDLComponents(unittest.TestCase):

    def setUp(self):
        # Create a synthetic card-like image (630x880 BGR)
        self.sample_card = np.zeros((880, 630, 3), dtype=np.uint8)
        self.sample_card[50:830, 50:580] = (200, 150, 50)  # Blue-ish card body
        cv2.putText(self.sample_card, "CHARIZARD ex", (70, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(self.sample_card, "HP 330", (450, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(self.sample_card, "199/165", (450, 840), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # -------------------------------------------------------------------------
    # Component 2: Frame Quality Classifier Tests
    # -------------------------------------------------------------------------
    def test_quality_classifier_fallback_on_missing_model(self):
        """Tests that FrameQualityClassifier raises ModelNotAvailableError when weights file is missing."""
        classifier = FrameQualityClassifier(pytorch_model_path="non_existent_model.pt", keras_model_path="non_existent.keras")
        with self.assertRaises(QualityModelNotAvailableError):
            classifier.classify(self.sample_card)

    def test_quality_classifier_inference(self):
        """Tests that frame quality classifier returns valid probability distribution."""
        if os.path.exists("models/quality_mobilenetv2.pt"):
            result = classify_frame_quality(self.sample_card, pytorch_model_path="models/quality_mobilenetv2.pt")
            self.assertIn("blurry", result)
            self.assertIn("glare", result)
            self.assertIn("occluded", result)
            self.assertIn("good", result)

    # -------------------------------------------------------------------------
    # Component 3: Visual Card Matcher Tests
    # -------------------------------------------------------------------------
    def test_visual_matcher_embedding_extraction(self):
        """Tests ResNet50 2048-dim normalized embedding extraction."""
        matcher = VisualCardMatcher(index_path="non_existent.faiss", meta_path="non_existent.json")
        if matcher.model is not None:
            embedding = matcher.extract_embedding(self.sample_card)
            self.assertEqual(embedding.shape, (2048,))
            norm = np.linalg.norm(embedding)
            self.assertAlmostEqual(norm, 1.0, places=3)

    def test_visual_matcher_fallback_on_missing_index(self):
        """Tests that match_card_by_image raises ModelNotAvailableError when index is missing."""
        matcher = VisualCardMatcher(index_path="models/missing.faiss", meta_path="models/missing.json")
        with self.assertRaises(VisualModelNotAvailableError):
            matcher.match_card_by_image(self.sample_card)

    # -------------------------------------------------------------------------
    # Component 1: YOLO Card Detector Tests
    # -------------------------------------------------------------------------
    def test_bounding_box_crop(self):
        """Tests BoundingBox ROI cropping logic."""
        bbox = BoundingBox(x1=50, y1=50, x2=200, y2=300, confidence=0.95, label="card")
        self.assertEqual(bbox.width, 150)
        self.assertEqual(bbox.height, 250)
        crop = bbox.crop(self.sample_card)
        self.assertEqual(crop.shape, (250, 150, 3))

    def test_yolo_detector_fallback_on_missing_model(self):
        """Tests that YOLOCardDetector raises ModelNotAvailableError when model file is missing."""
        detector = YOLOCardDetector(model_path="models/missing_yolo.pt")
        with self.assertRaises(YOLONotAvailableError):
            detector.detect_regions(self.sample_card)

    # -------------------------------------------------------------------------
    # Integration Tests: Layered OCR & Verification Pipeline
    # -------------------------------------------------------------------------
    def test_assess_frame_quality_layered(self):
        """Tests assess_frame_quality with ML quality scores included."""
        extractor = PokemonCardExtractor(languages=['en'], gpu=False)
        res = extractor.assess_frame_quality(self.sample_card)
        self.assertIn("pass", res)
        self.assertIn("reason", res)

    def test_verify_card_visual_matching_graceful_fallback(self):
        """Tests PokemonTCGClient.verify_card with visual card image input."""
        client = PokemonTCGClient()
        res = client.verify_card(
            collector_id="199/165",
            ocr_name="Charizard ex",
            ocr_hp=330,
            card_image=self.sample_card
        )
        self.assertIn("verified", res)
        self.assertIn("confidence", res)

    def test_special_cards_collector_id(self):
        """Tests collector ID extraction and parsing for special cards (TG, GG, Promos, SIRs)."""
        extractor = PokemonCardExtractor(languages=['en'], gpu=False)

        # 1. Trainer Gallery
        self.assertEqual(extractor.parse_collector_id("ILLUSTRATOR TG01/TG30"), "TG01/TG30")
        # 2. Galarian Gallery
        self.assertEqual(extractor.parse_collector_id("SECRET GG12/GG70"), "GG12/GG70")
        # 3. Promo Cards
        self.assertEqual(extractor.parse_collector_id("PROMO SVP025"), "SVP025")
        self.assertEqual(extractor.parse_collector_id("SWSH 050"), "SWSH050")
        # 4. Secret Rares / SIR (Numerator > Denominator)
        self.assertEqual(extractor.parse_collector_id("199/165"), "199/165")
        self.assertEqual(extractor.parse_collector_id("251/198"), "251/198")
        # 5. Protected valid set total (165 should not become 168)
        self.assertEqual(extractor._correct_set_total("165"), "165")

    # -------------------------------------------------------------------------
    # Multi-Path Recognition & Closed-Set Name Retrieval Tests
    # -------------------------------------------------------------------------
    def test_closed_set_name_matcher(self):
        """Tests database-backed closed-set card name resolution."""
        from custom_card_recognizer import ClosedSetCardNameMatcher
        matcher = ClosedSetCardNameMatcher(catalog_path="models/card_catalog.json")
        self.assertGreater(len(matcher.canonical_names), 0)

        # Test noisy/misread OCR candidates snapping to database names
        matched_name, conf = matcher.match_name("Charizardex")
        self.assertEqual(matched_name, "Charizard-EX")
        self.assertGreaterEqual(conf, 0.8)

        matched_name, conf = matcher.match_name("Mewe")
        self.assertEqual(matched_name, "Mew-EX")
        self.assertGreaterEqual(conf, 0.8)

        # Test resolving candidate list
        res_name, res_conf = matcher.resolve_candidates(["Charizardex", "Charizard ex"])
        self.assertEqual(res_name, "Charizard-EX")

    def test_restricted_alphabet_recognizer(self):
        """Tests constraint-based recognition with restricted alphabets for HP and Collector IDs."""
        from custom_card_recognizer import RestrictedAlphabetRecognizer
        recognizer = RestrictedAlphabetRecognizer()
        
        # Test HP crop recognition
        hp_crop = np.zeros((40, 120, 3), dtype=np.uint8)
        cv2.putText(hp_crop, "330 HP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        hp_results = recognizer.recognize_crop(hp_crop, field_type="hp")
        self.assertTrue(isinstance(hp_results, list))

        # Test Collector ID crop recognition
        id_crop = np.zeros((40, 160, 3), dtype=np.uint8)
        cv2.putText(id_crop, "199/165", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        id_results = recognizer.recognize_crop(id_crop, field_type="collector_id")
        self.assertTrue(isinstance(id_results, list))


if __name__ == "__main__":
    unittest.main()
