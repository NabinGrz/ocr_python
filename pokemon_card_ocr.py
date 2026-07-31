"""
Pokémon Card High-Accuracy OCR & Data Extraction Engine
Extracts: Pokémon Name, HP, and Unique Collector ID using OpenCV & EasyOCR

Supports:
- Standard cards (e.g. Breloom 004/198)
- ex / VMAX / VSTAR cards
- Mega Evolution cards (Mega Greninja ex)
- Full-art / Special Illustration Rare (SIR) cards where card number > set total (e.g. 116/086)
"""

import re
import cv2
import numpy as np
import easyocr
from typing import Dict, Any, Optional, List, Tuple

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
            'Z': '2', 'z': '2',
            '7': '/',
        }

        # Layout keywords that are NOT part of the Pokémon name.
        # NOTE: "mega" intentionally excluded — it IS part of names like "Mega Greninja ex"
        self.ignore_name_words = {
            'basic', 'stage', 'stage1', 'stage2', 'vmax', 'vstar', 'tera',
            'evolves', 'from', 'pokemon', 'pokémon', 'hp', 'len', 'duc', 'duc9',
            'restored', 'ability', 'attack', 'weakness', 'resistance', 'retreat',
            'rule', 'when', 'your', 'the', 'form', 'evolved', 'prize'
        }

        # Valid Pokémon name suffixes (appended to the base name)
        self.name_suffixes = {'ex', 'gx', 'vmax', 'vstar', 'v', 'mega'}

        # Patterns that OCR mistakes "ex" for
        self.ex_aliases = re.compile(r'[@\*]?[eE][xX3]$|[eE][xX3]$|\bex\b', re.IGNORECASE)

    def preprocess_and_warp(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 200)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        for c in contours:
            area = cv2.contourArea(c)
            if area > (h * w * 0.30) and area < (h * w * 0.95):
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
        Crops ROIs for OCR:
        - Header (Name & HP): top 18%
        - Footer (Collector ID): bottom 25% (wider to cover full-art SIR footer bars)
        """
        h, w = card_img.shape[:2]
        header_crop = card_img[0:int(h * 0.18), 0:w]
        footer_crop = card_img[int(h * 0.75):h, 0:w]  # Wide crop to catch dark full-art ID bars
        return {
            "header": header_crop,
            "footer": footer_crop
        }

    def _enhance_roi(self, roi: np.ndarray) -> np.ndarray:
        """
        Applies CLAHE + sharpening to boost OCR accuracy on low-contrast
        holographic / dark full-art card regions.
        Returns an enhanced BGR image.
        """
        # Upscale for better OCR resolution
        roi_large = cv2.resize(roi, (roi.shape[1] * 2, roi.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(roi_large, cv2.COLOR_BGR2GRAY)

        # CLAHE contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Sharpen
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    def parse_hp(self, text: str) -> Optional[int]:
        """
        Extracts HP from header text. Accepts up to 400 for Mega ex cards (350 HP).
        """
        # Also handles OCR noise like "0P350" → 350
        hp_match = re.search(r'(?:[A-Z]{0,2})?(\d{2,3})\s*(?:HP)?', text, re.IGNORECASE)
        if hp_match:
            val = int(hp_match.group(1))
            if 30 <= val <= 400:
                return val
        return None

    def _fix_collector_id_slashes(self, text: str) -> str:
        """
        OCR on dark holographic backgrounds sometimes reads '116/086' as '1167086' or '116l086'
        when the slash character is misread. This corrects those specific patterns.
        Only triggers when there is NO real '/' already present nearby the digits.
        """
        # Only fix if no actual '/' exists in the vicinity of digit clusters
        if '/' in text:
            return text
        # Pattern: 3-digit number, then '7' or 'l' or '1' as separator, then 2-3 digit number
        # e.g. "1167086" → "116/086", "1161086" → "116/086"
        text = re.sub(
            r'\b(\d{2,3})[7l](\d{2,3})\b',
            lambda m: f"{m.group(1)}/{m.group(2)}",
            text
        )
        return text

    def parse_collector_id(self, text: str) -> Optional[str]:
        """
        Extracts Collector ID from footer text.
        """
        # Pre-process: fix slash misreads on dark backgrounds
        fixed_text = self._fix_collector_id_slashes(text)

        # Pattern 1: Standard numeric ID (e.g. "004/198", "116/086")
        num_match = re.search(r'(\d{1,4})\s*/(?:\D*?)(\d{1,4})', fixed_text)
        if num_match:
            num = num_match.group(1)
            total = num_match.group(2)
            if 0 < int(num) <= 999 and 0 < int(total) <= 999:
                return f"{num}/{total}"

        # Pattern 2: Trainer Gallery / GX alphanumeric (e.g. "TG01/TG30")
        std_match = re.search(r'([A-Z]{1,3}\d{1,3})\s*/\s*([A-Z]{1,3}\d{1,3})', fixed_text)
        if std_match:
            return f"{std_match.group(1)}/{std_match.group(2)}"

        # Pattern 3: Promo codes (e.g. "SWSH050", "SVP025")
        promo_match = re.search(r'\b([A-Z]{2,4})(\d{2,4})\b', fixed_text)
        if promo_match:
            digits = "".join([self.char_fix_map.get(c, c) for c in promo_match.group(2)])
            return f"{promo_match.group(1)}{digits}"

        return None

    def _normalize_token(self, text: str) -> str:
        """
        Cleans OCR token for name parsing:
        - 'Greninja@X' → 'Greninja ex'
        - 'GreninjaeX' → 'Greninja ex'
        """
        # Fix ex suffix misreads (e.g. '@X', 'eX', '3X', '@x')
        fixed = re.sub(r'[@\*\s]?[eE3][xX]$', ' ex', text)
        # Remove remaining non-letter chars except space, hyphen, apostrophe
        fixed = re.sub(r'[^a-zA-Z\s\'-]', '', fixed).strip()
        return fixed

    def extract_from_image(self, image_np: np.ndarray) -> Dict[str, Any]:
        """
        Full extraction pipeline: dewarping → ROI crop → enhance → OCR → parse.
        """
        warped = self.preprocess_and_warp(image_np)
        rois = self.crop_rois(warped)

        # Enhance ROIs for better OCR on dark/holographic cards
        header_enhanced = self._enhance_roi(rois["header"])
        footer_enhanced = self._enhance_roi(rois["footer"])

        header_results = self.reader.readtext(header_enhanced, detail=1)
        footer_results = self.reader.readtext(footer_enhanced, detail=1)

        # Sort results left-to-right, top-to-bottom
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

    def _extract_name_by_font_size(self, ocr_results: List[Tuple]) -> Optional[str]:
        """
        Identifies the Pokémon name by selecting the largest-font OCR token(s).
        Adjacent large tokens (e.g. "Mega Greninja" + "ex") are merged when they:
        - Have a similar Y-coordinate (same line), OR
        - The adjacent token is a known suffix like 'ex', 'gx', 'v'
        """
        if not ocr_results:
            return None

        candidates = []
        for bbox, text, prob in ocr_results:
            # Normalize token — fix ex misreads like @X, GX with trailing noise
            clean_text = self._normalize_token(text)
            if not clean_text or len(clean_text) < 2:
                continue

            lowered = clean_text.lower().strip()
            if lowered in self.ignore_name_words and lowered not in self.name_suffixes:
                continue

            ys = [p[1] for p in bbox]
            xs = [p[0] for p in bbox]
            height = max(ys) - min(ys)
            mid_x = (min(xs) + max(xs)) / 2
            mid_y = (min(ys) + max(ys)) / 2

            candidates.append({
                "text": clean_text,
                "height": height,
                "mid_x": mid_x,
                "mid_y": mid_y,
                "prob": prob
            })

        if not candidates:
            return None

        # Sort by font height descending
        candidates.sort(key=lambda x: x["height"], reverse=True)
        primary = candidates[0]
        primary_height_threshold = primary["height"] * 0.55

        # Merge adjacent large tokens (same line or known suffixes)
        name_tokens = [primary]
        for c in candidates[1:]:
            token_lower = c["text"].lower().strip()
            if token_lower in self.name_suffixes:
                name_tokens.append(c)
            elif (
                c["height"] >= primary_height_threshold
                and abs(c["mid_y"] - primary["mid_y"]) < primary["height"] * 0.9
            ):
                name_tokens.append(c)

        # Sort merged tokens left to right
        name_tokens.sort(key=lambda x: x["mid_x"])
        return " ".join(t["text"] for t in name_tokens)
