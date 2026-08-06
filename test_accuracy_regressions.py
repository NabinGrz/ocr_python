"""Regression tests for identification accuracy and confidence gating."""

import os
import unittest
from unittest.mock import patch

import cv2

from pokemon_api import PokemonTCGClient
from pokemon_card_ocr import PokemonCardExtractor


def make_card(
    card_id="sv3pt5-151",
    name="Mew ex",
    hp="180",
    number="151",
    printed_total=165,
):
    return {
        "id": card_id,
        "name": name,
        "hp": hp,
        "number": number,
        "rarity": "Double Rare",
        "set": {
            "name": "151",
            "series": "Scarlet & Violet",
            "printedTotal": printed_total,
        },
        "images": {"large": "https://example.test/card.png"},
    }


class StubPokemonTCGClient(PokemonTCGClient):
    def __init__(self, cards):
        super().__init__(catalog_path="models/nonexistent-test-catalog.json")
        self.cards = cards

    def _safe_get(self, params, timeout=5.0):
        return self.cards


class TestCandidateRanking(unittest.TestCase):
    def test_exact_collector_id_overcomes_minor_name_noise(self):
        cards = [
            make_card(),
            make_card(
                card_id="other-151",
                name="Mewtwo",
                hp="120",
                printed_total=198,
            ),
        ]
        result = StubPokemonTCGClient(cards).verify_card(
            collector_id="151/165", ocr_name="Mewe", ocr_hp=180
        )

        self.assertTrue(result["verified"])
        self.assertEqual(result["name"], "Mew ex")
        self.assertEqual(result["collector_id"], "151/165")
        self.assertGreaterEqual(result["confidence"], 0.9)
        self.assertGreaterEqual(len(result["candidates"]), 2)

    def test_name_only_does_not_create_false_verification(self):
        result = StubPokemonTCGClient([make_card(name="Mewtwo")]).verify_card(
            collector_id=None, ocr_name="Mew", ocr_hp=None
        )
        self.assertFalse(result["verified"])

    def test_name_and_hp_without_set_identity_remain_candidates(self):
        result = StubPokemonTCGClient([make_card()]).verify_card(
            collector_id=None, ocr_name="Mew ex", ocr_hp=180
        )
        self.assertFalse(result["verified"])
        self.assertGreaterEqual(len(result["candidates"]), 1)

    def test_collector_id_alone_is_not_enough_when_sets_can_overlap(self):
        result = StubPokemonTCGClient([make_card()]).verify_card(
            collector_id="151/165", ocr_name=None, ocr_hp=None
        )
        self.assertFalse(result["verified"])

    def test_visual_match_requires_independent_ocr_evidence(self):
        visual_candidate = {
            "name": "Charizard ex",
            "hp": 330,
            "collector_id": "199/165",
            "similarity_score": 0.99,
        }
        with patch("pokemon_api.match_card_by_image", return_value=[visual_candidate]):
            result = StubPokemonTCGClient([make_card()]).verify_card(
                collector_id=None,
                ocr_name=None,
                ocr_hp=None,
                card_image=cv2.imread("sample_mew.png"),
            )
        self.assertFalse(result["verified"])


@unittest.skipUnless(os.path.exists("sample_mew.png"), "reference card is unavailable")
class TestReferenceCard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = cv2.imread("sample_mew.png")
        cls.extractor = PokemonCardExtractor(gpu=False)

    def test_reference_image_is_not_rejected_by_quality_gate(self):
        quality = self.extractor.assess_frame_quality(self.image)
        self.assertTrue(quality["pass"], quality)
        self.assertGreater(quality["quality_score"], 60.0)

    def test_reference_card_core_fields(self):
        result = self.extractor.extract_from_image(self.image)
        self.assertEqual(result["hp"], 180)
        self.assertEqual(result["unique_id"], "151/165")
        self.assertIn("mew", (result["name"] or "").lower())
        self.assertGreaterEqual(len(result["ocr_id_candidates"]), 1)


class TestPrintedIDPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from custom_card_recognizer import PrintedIDParser, MultiFrameIDVoter
        cls.parser = PrintedIDParser()
        cls.voter = MultiFrameIDVoter(min_agreements=3)

    def test_printed_id_parser_formats(self):
        cases = [
            ("XY124", "XY124"),
            ("XY 124", "XY124"),
            ("4/102", "4/102"),
            ("4 of 102", "4/102"),
            ("SV124/198", "SV124/198"),
            ("TG01/TG30", "TG01/TG30"),
            ("SV01a", "SV01A"),
        ]
        for text, expected in cases:
            parsed, conf, pattern = self.parser.parse_printed_id(text)
            self.assertEqual(parsed, expected, f"Failed parsing '{text}', got '{parsed}'")
            self.assertGreaterEqual(conf, 0.85)

    def test_grammar_aware_confusion_normalization(self):
        cases = [
            ("X0124", "XY124"),   # 'O' -> 'Y' in Promo prefix
            ("XY I24", "XY124"),  # 'I' -> '1' in numeric slot
            ("S5050", "SWSH050"), # '5' -> 'W' in SWSH prefix
        ]
        for text, expected in cases:
            parsed, conf, pattern = self.parser.parse_printed_id(text)
            self.assertEqual(parsed, expected, f"Failed normalizing '{text}', got '{parsed}'")

    def test_multi_frame_voting(self):
        voter = self.voter.__class__(min_agreements=3)
        self.assertIsNone(voter.get_consensus())

        voter.add_observation("XY124", confidence=0.90)
        self.assertIsNone(voter.get_consensus())

        voter.add_observation("XY124", confidence=0.95)
        self.assertIsNone(voter.get_consensus())

        voter.add_observation("XY124", confidence=0.92)
        self.assertEqual(voter.get_consensus(), "XY124")

    def test_negative_unreadable_printed_id(self):
        bad_texts = ["ILLUSTRATOR KEN SUGIMORI", "FOIL NOISE 12345", "RANDOM JUNK TEXT"]
        for bad_text in bad_texts:
            parsed, conf, pattern = self.parser.parse_printed_id(bad_text)
            self.assertIsNone(parsed, f"Expected None for '{bad_text}', got '{parsed}'")


@unittest.skipUnless(os.path.exists("sample_xy124.png"), "sample_xy124.png is unavailable")
class TestSampleXY124Card(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = cv2.imread("sample_xy124.png")
        cls.extractor = PokemonCardExtractor(gpu=False)

    def test_xy124_extraction(self):
        result = self.extractor.extract_from_image(self.image)
        self.assertEqual(result["name"], "Pikachu ex")
        self.assertEqual(result["hp"], 130)
        self.assertEqual(result["normalized_printed_id"], "XY124")
        self.assertEqual(result["printed_id_roi_source"], "footer_right")
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["reason"], "success")
        self.assertTrue(os.path.exists(result["debug_crop_path"]))


if __name__ == "__main__":
    unittest.main()

