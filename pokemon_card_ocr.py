"""
Pokémon Card High-Accuracy OCR & Data Extraction Engine
Extracts: Pokémon Name, HP, and Unique Collector ID using OpenCV & EasyOCR
"""

import re
import cv2
import numpy as np
import easyocr
from typing import Dict, Any, Optional

class PokemonCardExtractor:
    def __init__(self, languages=['en'], gpu=False):
        # Initialize EasyOCR reader
        self.reader = easyocr.Reader(languages, gpu=gpu)
        
        # Character confusion map common in OCR
        self.char_fix_map = {
            'O': '0', 'o': '0',
            'I': '1', 'l': '1', '|': '1',
            'S': '5', 's': '5',
            'B': '8',
            'Z': '2', 'z': '2'
        }

        # Common noise words in header/footer to ignore when looking for card name
        self.ignore_name_words = {
            'basic', 'stage', 'stage1', 'stage2', 'vmax', 'vstar', 'tera', 'mega',
            'evolves', 'from', 'pokemon', 'pokémon', 'hp', 'len', 'duc', 'duc9'
        }

    def preprocess_and_warp(self, image: np.ndarray) -> np.ndarray:
        """
        Detects card boundary contours if present in photo, otherwise resizes directly.
        Normalizes card into standard dimensions (630 x 880).
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 200)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        for c in contours:
            area = cv2.contourArea(c)
            if area > (h * w * 0.3) and area < (h * w * 0.95):
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    pts = approx.reshape(4, 2)
                    rect = self._order_points(pts)
                    dst = np.array([[0, 0], [629, 0], [629, 879], [0, 879]], dtype="float32")
                    M = cv2.getPerspectiveTransform(rect, dst)
                    return cv2.warpPerspective(image, M, (630, 880))

        return cv2.resize(image, (630, 880))

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def crop_rois(self, card_img: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Crops specific Regions of Interest (ROIs):
        - Header (Name & HP): top 18% height
        - Footer (Collector ID): bottom 18% height
        """
        h, w = card_img.shape[:2]
        header_crop = card_img[0:int(h * 0.18), 0:w]
        footer_crop = card_img[int(h * 0.82):h, 0:w]
        return {
            "header": header_crop,
            "footer": footer_crop
        }

    def parse_hp(self, text: str) -> Optional[int]:
        hp_match = re.search(r'(?:HP\s*)?(\d{2,3})\s*(?:HP)?', text, re.IGNORECASE)
        if hp_match:
            val = int(hp_match.group(1))
            if 30 <= val <= 340:
                return val
        return None

    def parse_collector_id(self, text: str) -> Optional[str]:
        # Pattern 1: Pure numeric set ID format (e.g. "004/198" or "151/165"), allowing non-digits/newlines between slash and denominator
        num_match = re.search(r'(\d{1,4})\s*/(?:\D*?)(\d{1,4})', text)
        if num_match:
            num = num_match.group(1)
            total = num_match.group(2)
            return f"{num}/{total}"

        # Pattern 2: Alphanumeric set format (e.g. "TG01/TG30" or "GG04/GG70")
        std_match = re.search(r'([A-Z0-9]{1,4})\s*/(?:\D*?)([A-Z0-9]{1,4})', text)
        if std_match:
            num_raw, total_raw = std_match.group(1), std_match.group(2)
            num = "".join([self.char_fix_map.get(c, c) for c in num_raw])
            total = "".join([self.char_fix_map.get(c, c) for c in total_raw])
            return f"{num}/{total}"

        # Pattern 3: Promo format e.g. "SWSH050", "SVP025"
        promo_match = re.search(r'([A-Z]{2,4})\s*([0-9OISZB]{2,4})', text)
        if promo_match:
            digits = "".join([self.char_fix_map.get(c, c) for c in promo_match.group(2)])
            return f"{promo_match.group(1)}{digits}"

        return None

    def extract_from_image(self, image_np: np.ndarray) -> Dict[str, Any]:
        """
        Processes image array and returns extracted name, hp, unique collector id,
        warped card image, and header/footer crops.
        """
        warped = self.preprocess_and_warp(image_np)
        rois = self.crop_rois(warped)

        header_results = self.reader.readtext(rois["header"], detail=1)
        footer_results = self.reader.readtext(rois["footer"], detail=1)

        # Sort OCR results by top-to-bottom, left-to-right reading order
        header_results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))
        footer_results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))

        header_lines = [r[1] for r in header_results]
        footer_lines = [r[1] for r in footer_results]

        header_text = "\n".join(header_lines)
        footer_text = "\n".join(footer_lines)

        hp = self.parse_hp(header_text)
        collector_id = self.parse_collector_id(footer_text)
        name = self._extract_name_by_font_size(header_results)

        return {
            "name": name,
            "hp": hp,
            "unique_id": collector_id,
            "header_raw_ocr": header_text,
            "footer_raw_ocr": footer_text,
            "warped_card": warped,
            "header_crop": rois["header"],
            "footer_crop": rois["footer"]
        }

    def _extract_name_by_font_size(self, ocr_results) -> Optional[str]:
        """
        Ranks OCR bounding boxes by height. The card name is printed in the largest font in the header.
        """
        candidates = []
        for bbox, text, prob in ocr_results:
            clean_text = re.sub(r'[^a-zA-Z\s\'-]', '', text).strip()
            if not clean_text or len(clean_text) < 3:
                continue

            lowered = clean_text.lower()
            if any(w in lowered for w in self.ignore_name_words):
                continue

            # Calculate bounding box height (font size indicator)
            ys = [p[1] for p in bbox]
            height = max(ys) - min(ys)
            candidates.append((height, clean_text, prob))

        if not candidates:
            return None

        # Sort candidates descending by font height
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
