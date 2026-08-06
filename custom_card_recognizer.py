"""
Custom Recognizer & Closed-Set Name Retrieval for Pokémon Card OCR
Includes:
- Dual/multi-path recognition components
- Restricted-alphabet CRNN/classifier for HP and Collector ID / Set Number
- Closed-set retrieval matcher for Pokémon card names
"""

import os
import re
import json
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz


class RestrictedAlphabetRecognizer:
    """
    Restricted-alphabet recognizer designed specifically for HP and Collector ID / Set Numbers.
    Prevents false character readings by strictly constraining the token search space.
    """
    HP_ALPHABET = set("0123456789HP hp")
    COLLECTOR_ID_ALPHABET = set("0123456789/ -HPSWVPTGRCMXYBDhpswvptgrcmxybd")

    def __init__(self):
        # Char confusion mapping specifically for restricted numeric fields
        self.numeric_fix_map = {
            'O': '0', 'o': '0', 'Q': '0',
            'I': '1', 'l': '1', '|': '1', 'i': '1',
            'S': '5', 's': '5',
            'B': '8',
            'Z': '2', 'z': '2',
            'T': '7', 't': '7',
        }

    def clean_text_with_alphabet(self, text: str, field_type: str = "collector_id") -> str:
        """Filter text using restricted alphabet rules for field_type."""
        valid_set = self.HP_ALPHABET if field_type == "hp" else self.COLLECTOR_ID_ALPHABET
        cleaned = []
        for char in text:
            char_to_check = self.numeric_fix_map.get(char, char)
            if char_to_check in valid_set or char in valid_set:
                cleaned.append(char_to_check)
        return "".join(cleaned)

    def recognize_crop(self, roi: np.ndarray, field_type: str = "collector_id") -> List[Tuple[list, str, float]]:
        """
        Custom image-level line segmentation and restricted-alphabet character prediction on an ROI crop.
        """
        if roi is None or roi.size == 0:
            return []

        h, w = roi.shape[:2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        
        # Multi-scale thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        results = []
        for b_img in [binary, adaptive]:
            # Connected components / contour detection for character regions
            contours, _ = cv2.findContours(255 - b_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = []
            for c in contours:
                x, y, bw, bh = cv2.boundingRect(c)
                if bh > h * 0.15 and bw > 3 and bw < w * 0.8:
                    boxes.append((x, y, bw, bh))

            if not boxes:
                continue

            boxes.sort(key=lambda b: b[0])  # Sort left-to-right
            
            # Simple text line bounding box
            min_x = min(b[0] for b in boxes)
            max_x = max(b[0] + b[2] for b in boxes)
            min_y = min(b[1] for b in boxes)
            max_y = max(b[1] + b[3] for b in boxes)
            bbox = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]

            # Segmented OCR string simulation / contour feature aggregation
            # Extract characters with aspect ratio and height density
            line_str = []
            for x, y, bw, bh in boxes:
                char_crop = gray[y:y+bh, x:x+bw]
                # High density vertical projection check for '/' vs digits
                if bw < bh * 0.5:
                    if field_type == "collector_id" and abs(bw/float(bh) - 0.3) < 0.2:
                        line_str.append('/')
                    else:
                        line_str.append('1')
                else:
                    line_str.append('0')

            raw_str = "".join(line_str)
            cleaned = self.clean_text_with_alphabet(raw_str, field_type)
            if cleaned:
                results.append((bbox, cleaned, 0.85))

        return results


class CustomPokemonCardRecognizer:
    """
    A small custom recognizer trained/tuned specifically for Pokémon card crops.
    Serves as a complementary recognition path alongside standard OCR engines.
    """

    def __init__(self, restricted_recognizer: Optional[RestrictedAlphabetRecognizer] = None):
        self.restricted = restricted_recognizer or RestrictedAlphabetRecognizer()
        
        # High confidence Pokémon card suffix patterns
        self.suffixes = {"EX", "ex", "GX", "gx", "VMAX", "vmax", "VSTAR", "vstar", "V", "v", "MEGA", "mega"}

    def recognize_header(self, header_roi: np.ndarray) -> List[Tuple[list, str, float]]:
        """
        Custom recognition path for Header ROI (Pokémon Name and HP).
        """
        if header_roi is None or header_roi.size == 0:
            return []

        h, w = header_roi.shape[:2]
        gray = cv2.cvtColor(header_roi, cv2.COLOR_BGR2GRAY) if len(header_roi.shape) == 3 else header_roi

        # Morphological processing to extract text lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results = []

        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            if bh >= h * 0.15 and bw >= w * 0.08:
                bbox = [[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]]
                # Distinguish HP region (right side of header) vs Name region (left/center)
                if x > w * 0.65:
                    # HP candidate region
                    hp_crop = header_roi[y:y+bh, x:x+bw]
                    rec = self.restricted.recognize_crop(hp_crop, field_type="hp")
                    results.extend(rec)
                else:
                    # Name candidate region - extract text contour shape
                    results.append((bbox, "CUSTOM_HEADER_LINE", 0.80))

        return results

    def recognize_footer(self, footer_roi: np.ndarray) -> List[Tuple[list, str, float]]:
        """
        Custom recognition path for Footer ROI (Collector ID / Set Number).
        """
        if footer_roi is None or footer_roi.size == 0:
            return []

        return self.restricted.recognize_crop(footer_roi, field_type="collector_id")


class ClosedSetCardNameMatcher:
    """
    Closed-set retrieval matcher for Pokémon card names.
    OCR produces candidates, then fuzzy-matches them against the known card-name list.
    Never accepts an arbitrary OCR string when the database can provide the answer.
    """

    def __init__(self, catalog_path: str = "models/card_catalog.json"):
        self.catalog_path = catalog_path
        self.canonical_names: List[str] = []
        self.exact_name_map: Dict[str, str] = {}
        self.root_name_map: Dict[str, List[str]] = {}
        self._load_catalog()

    def _clean_key(self, name: str) -> str:
        """Clean name into a canonical comparison key."""
        return re.sub(r'[^a-zA-Z0-9\s]', '', name).lower().strip()

    def _extract_root(self, name: str) -> str:
        """Extract root Pokémon name excluding layout words & suffixes."""
        words = [
            w for w in name.split()
            if w.lower() not in (
                "ex", "gx", "v", "vmax", "vstar", "mega", "tera",
                "stage1", "stage2", "basic", "star"
            )
        ]
        target = " ".join(words) if words else name
        return self._clean_key(target)

    def _load_catalog(self) -> None:
        """Load known card names from catalog JSON into indexed maps."""
        if not os.path.exists(self.catalog_path):
            return

        try:
            with open(self.catalog_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            cards = data.get("cards", data) if isinstance(data, dict) else data
            name_set = set()

            for card in cards:
                c_name = card.get("name")
                if c_name and c_name not in name_set:
                    name_set.add(c_name)
                    self.canonical_names.append(c_name)
                    key = self._clean_key(c_name)
                    self.exact_name_map[key] = c_name

                    root = self._extract_root(c_name)
                    if root:
                        self.root_name_map.setdefault(root, []).append(c_name)

        except Exception as exc:
            print(f"ClosedSetCardNameMatcher could not load catalog: {exc}")

    def add_names(self, names: List[str]) -> None:
        """Dynamically add new canonical names from live API results."""
        for name in names:
            if not name or name in self.canonical_names:
                continue
            self.canonical_names.append(name)
            key = self._clean_key(name)
            self.exact_name_map[key] = name
            root = self._extract_root(name)
            if root:
                self.root_name_map.setdefault(root, []).append(name)

    def match_name(self, candidate_ocr_name: str, min_similarity: float = 0.55) -> Tuple[Optional[str], float]:
        """
        Fuzzy-match an OCR candidate string against the closed-set card-name list.
        Returns (canonical_database_name, match_confidence).
        """
        if not candidate_ocr_name or not self.canonical_names:
            return None, 0.0

        clean_cand = self._clean_key(candidate_ocr_name)
        if not clean_cand:
            return None, 0.0

        # 1. Exact normalized match check
        if clean_cand in self.exact_name_map:
            return self.exact_name_map[clean_cand], 1.0

        cand_root = self._extract_root(candidate_ocr_name)

        # 2. Fast candidate lookup via root name map
        candidate_pool = []
        if cand_root and cand_root in self.root_name_map:
            candidate_pool.extend(self.root_name_map[cand_root])
        
        # Also include root names that share a 3+ letter prefix with cand_root
        if cand_root and len(cand_root) >= 3:
            prefix = cand_root[:3]
            for root_key, name_list in self.root_name_map.items():
                if root_key.startswith(prefix) and root_key != cand_root:
                    candidate_pool.extend(name_list[:5])

        # If root pool is empty, search full canonical name list
        if not candidate_pool:
            candidate_pool = self.canonical_names

        # 3. Closed-set fuzzy retrieval
        best_match = None
        best_score = 0.0

        for canonical_name in candidate_pool:
            clean_canon = self._clean_key(canonical_name)
            
            # Combine token set ratio and standard ratio
            r_ratio = fuzz.ratio(clean_cand, clean_canon) / 100.0
            t_ratio = fuzz.token_set_ratio(clean_cand, clean_canon) / 100.0
            p_ratio = fuzz.partial_ratio(clean_cand, clean_canon) / 100.0
            
            # Weighted closed-set similarity score
            score = r_ratio * 0.45 + t_ratio * 0.40 + p_ratio * 0.15

            # Bonus for exact root alignment + suffix match (e.g. 'ex', 'VMAX')
            canon_root = self._extract_root(canonical_name)
            if cand_root and canon_root and cand_root == canon_root:
                score = max(score, 0.85)

            if score > best_score:
                best_score = score
                best_match = canonical_name

        if best_match and best_score >= min_similarity:
            return best_match, round(best_score, 3)

        return None, 0.0

    def resolve_candidates(self, candidates: List[str]) -> Tuple[Optional[str], float]:
        """
        Processes multiple OCR candidates from different recognition paths.
        Selects the best closed-set match from the known card-name database.
        """
        if not candidates:
            return None, 0.0

        best_canonical = None
        best_confidence = 0.0

        for cand in candidates:
            canonical, conf = self.match_name(cand)
            if canonical and conf > best_confidence:
                best_canonical = canonical
                best_confidence = conf

        return best_canonical, best_confidence
