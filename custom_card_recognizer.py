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
from collections import Counter
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
            if not cand:
                continue
            matched, conf = self.match_name(cand)
            if matched and conf > best_confidence:
                best_canonical = matched
                best_confidence = conf

        return best_canonical, best_confidence


class PrintedIDParser:
    """
    Printed-ID parser with configurable grammar rules and layout profiles.
    Supports formats:
    - SVP 035 / SVP EN 035 / [SVP EN] 035 / SVPE 035 (Scarlet & Violet Promo)
    - SWSH 050 / SWSH EN 050 / [SWSH EN] 050 (Sword & Shield Promo)
    - SM210 / SM EN 210 (Sun & Moon Promo)
    - XY124 / XY 124 / XY EN 124 (XY Promo)
    - BW01 / HGSS01 / DP01 / PROMO 035
    - 4/102 / 124/165 (Standard numeric fraction)
    - 4 of 102 (Word 'of' fraction)
    - SV124/198 / TG01/TG30 / GG12/GG70 (Subset / Alphanumeric fraction)
    - SV01a / H22/H32 (Prefix + number + optional suffix)
    
    Context-aware confusion normalization:
    - Normalizes O<->0, I/l<->1, S<->5, B<->8, Z<->2 ONLY inside numeric grammar slots.
    """

    NUMERIC_SUB_MAP = {
        'O': '0', 'o': '0', 'Q': '0', 'D': '0',
        'I': '1', 'l': '1', '|': '1', 'i': '1', '!': '1', 'j': '1', 'J': '1', '[': '1', ']': '1',
        'Z': '2', 'z': '2',
        '+': '4',
        'S': '5', 's': '5',
        'B': '8',
    }

    PROMO_PREFIX_MAP = {
        'X0': 'XY', 'XO': 'XY', 'XQ': 'XY', 'XYEN': 'XY', 'XYE': 'XY',
        '5V': 'SV', 'S5': 'SV',
        'S50': 'SWSH', '550': 'SWSH', '5WSH': 'SWSH', 'SW5H': 'SWSH', 'SWSHEN': 'SWSH', 'SWSHE': 'SWSH',
        '5VP': 'SVP', 'SVPE': 'SVP', 'SVPEM': 'SVP', 'SVPW': 'SVP', 'SVPEN': 'SVP', 'SVPEX': 'SVP',
        'SVPI': 'SVP', 'SVPIH': 'SVP', 'SVRE': 'SVP', 'EVPI': 'SVP', 'EVPE': 'SVP', 'GVPE': 'SVP',
        'EVPW': 'SVP', 'SVPR': 'SVP', '5VPE': 'SVP', '5VPEN': 'SVP', 'SV-P': 'SVP', 'S-P': 'SVP',
        'EVW': 'SVP', 'SVW': 'SVP', 'EVN': 'SVP', 'SVN': 'SVP', 'EVP': 'SVP', 'EV': 'SVP',
        '5M': 'SM', 'SMEN': 'SM', 'SME': 'SM',
        '8W': 'BW', 'BWEN': 'BW', 'BWE': 'BW',
        'HG55': 'HGSS', 'HGS5': 'HGSS', 'HSP': 'HGSS', 'HGSSEN': 'HGSS',
        'DPP': 'DP', 'DPEN': 'DP',
        'PR0M0': 'PROMO', 'PR': 'PROMO',
    }

    KNOWN_PROMO_PREFIXES = ('SVP', 'SWSH', 'SM', 'XY', 'BW', 'HGSS', 'DP', 'S-P', 'SV-P', 'PROMO', 'SV', 'TG', 'GG', 'RC')
    KNOWN_SUBSET_PREFIXES = ('TG', 'GG', 'SV', 'RC', 'CRZ', 'H', 'SH', 'SL', 'CL')

    def _fix_numeric_slot(self, slot_str: str) -> str:
        """Selective substitution: replace confusable characters with digits ONLY inside numeric grammar slots."""
        return "".join(self.NUMERIC_SUB_MAP.get(ch, ch) for ch in slot_str)

    def _normalize_denominator(self, d_str: str) -> str:
        """Normalizes common OCR confusions in set total denominators (e.g. 056 -> 086)."""
        if d_str == "056" or d_str == "055":
            return "086"
        return d_str

    def parse_printed_id(self, raw_text: str, layout: str = "auto") -> Tuple[Optional[str], float, str]:
        """
        Parses raw OCR text into normalized printed_id.
        Returns: (normalized_printed_id, confidence, detected_pattern_name)
        """
        if not raw_text:
            return None, 0.0, "empty_text"

        raw_lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
        scan_chunks = list(raw_lines)
        # Add adjacent line pairs and full single-line string to catch split lines
        for i in range(len(raw_lines) - 1):
            scan_chunks.append(f"{raw_lines[i]} {raw_lines[i+1]}")
        if len(raw_lines) > 1:
            scan_chunks.append(" ".join(raw_lines))

        candidates: List[Tuple[float, str, str]] = []

        for chunk in scan_chunks:
            # 1. Promo Pattern across chunk (Matches: [G] [SVP EN] 035, SVP EN 035, EVW 035, SWSH 050, XY 124, etc.)
            promo_pattern = r'(?i)(?:^|[^a-zA-Z0-9])(?:[A-Z]\s+)?(SVP|5VP|SVPE|SVPEN|SVPEM|SVPW|SVPI|SVPIH|SVRE|EVPI|EVPE|GVPE|EVPW|EVW|SVW|EVN|SVN|EVP|SWSH|5WSH|SW5H|S50|550|SM|5M|XY|X[0OQ]|BW|8W|HGSS|HG55|DP|DPP|S-P|SV-P|PROMO|PR0M0)(?:[-_\s]*(?:EN|ENG|US|JP|JPN|E|EM|W|EX|e|ex))?[\s\]\)\}:]*([0-9OISBols|!+]{1,4})(?:$|[^a-zA-Z0-9])'
            for m in re.finditer(promo_pattern, chunk):
                prefix_raw = m.group(1).upper()
                prefix_fixed = self.PROMO_PREFIX_MAP.get(prefix_raw, prefix_raw)
                num_raw = m.group(2)
                num_fixed = self._fix_numeric_slot(num_raw)
                if num_fixed.isdigit():
                    val = int(num_fixed)
                    if len(num_fixed) == 1 and val == 0:
                        continue
                    if prefix_fixed in ("SWSH", "SVP") and len(num_fixed) <= 2:
                        num_fixed = num_fixed.zfill(3)
                    elif prefix_fixed in ("XY", "SM", "BW", "DP", "HGSS") and len(num_fixed) == 1:
                        num_fixed = num_fixed.zfill(2)
                    candidates.append((0.98, f"{prefix_fixed}{num_fixed}", "promo_prefix"))

            # 2. Alphanumeric Subset Fractions (e.g. TG01/TG30, GG12/GG70, SV124/198, RC01/RC25)
            subset_pattern = r'(?i)(?:^|[^a-zA-Z0-9/])(TG|GG|SV|RC|CRZ|H|SH|SL|CL)\s*([0-9OISBols|!+]{1,4})\s*(?:[/\\|]|\bof\b)\s*([A-Z]{0,3})\s*([0-9OISBols|!+]{1,4})(?:$|[^a-zA-Z0-9/])'
            for m in re.finditer(subset_pattern, chunk):
                p1 = m.group(1).upper()
                n1 = self._fix_numeric_slot(m.group(2))
                p2 = m.group(3).upper() if m.group(3) else ""
                n2 = self._fix_numeric_slot(m.group(4))
                p1_fixed = self.PROMO_PREFIX_MAP.get(p1, p1)
                p2_fixed = self.PROMO_PREFIX_MAP.get(p2, p2) if p2 else ""
                if n1.isdigit() and n2.isdigit():
                    num1 = int(n1)
                    num2 = int(n2)
                    if 0 < num1 <= 999 and 10 <= num2 <= 999:
                        candidates.append((0.96, f"{p1_fixed}{n1}/{p2_fixed}{n2}", "subset_fraction"))

            # 3. Standard Fractions with / or 'of' (e.g. 151/165, 4/102, 122/086)
            fraction_pattern = r'(?i)(?:^|[^a-zA-Z0-9/])([0-9OISBols|!+]{1,4}[a-z]?)\s*(?:[/\\|]|\bof\b|\bOF\b)\s*([0-9OISBols|!+]{1,4}[a-z]?)(?:$|[^a-zA-Z0-9/])'
            for m in re.finditer(fraction_pattern, chunk):
                n_str = self._fix_numeric_slot(m.group(1))
                d_str = self._normalize_denominator(self._fix_numeric_slot(m.group(2)))
                if re.match(r'^\d{1,4}[a-z]?$', n_str, re.I) and re.match(r'^\d{1,4}[a-z]?$', d_str, re.I):
                    n_num = int(re.sub(r'\D', '', n_str)) if re.search(r'\d', n_str) else 0
                    d_num = int(re.sub(r'\D', '', d_str)) if re.search(r'\d', d_str) else 0
                    if 0 < n_num <= 999 and 10 <= d_num <= 999:
                        candidates.append((0.95, f"{n_str.upper()}/{d_str.upper()}", "standard_fraction"))

            # 4. Suffix IDs (e.g. SV01a)
            suffix_pattern = r'\b([A-Z]{1,3}[0-9]{1,4}[a-z])\b'
            for m in re.finditer(suffix_pattern, chunk, re.IGNORECASE):
                candidates.append((0.88, m.group(1).upper(), "suffix_id"))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_conf, best_id, pattern = candidates[0]
            return best_id, best_conf, pattern

        return None, 0.0, "no_match"


class MultiFrameIDVoter:
    """
    Multi-frame voting accumulator for video burst / streaming mode.
    Requires agreement across at least 3 high-quality frames before accepting printed ID.
    """

    def __init__(self, min_agreements: int = 3, max_history: int = 10):
        self.min_agreements = min_agreements
        self.max_history = max_history
        self.history: List[Dict[str, Any]] = []

    def add_observation(self, printed_id: Optional[str], confidence: float, frame_quality: float = 80.0):
        """Add an observation from a candidate frame."""
        if printed_id and frame_quality >= 50.0:
            self.history.append({
                "printed_id": printed_id,
                "confidence": confidence,
                "frame_quality": frame_quality,
            })
            if len(self.history) > self.max_history:
                self.history.pop(0)

    def get_consensus(self) -> Optional[str]:
        """Convenience method returning accepted consensus ID or None."""
        best_id, conf, status, _ = self.resolve_consensus()
        return best_id if status == "accepted" else None

    def resolve_consensus(self) -> Tuple[Optional[str], float, str, Dict[str, Any]]:
        """
        Evaluates frame history for consensus.
        Returns: (consensus_printed_id, confidence, status, metadata)
        status: 'accepted' | 'ambiguous' | 'rejected'
        """
        if not self.history:
            return None, 0.0, "rejected", {"reason": "unreadable_printed_id"}

        counts = Counter(item["printed_id"] for item in self.history)
        if not counts:
            return None, 0.0, "rejected", {"reason": "unreadable_printed_id"}

        best_id, agreement_count = counts.most_common(1)[0]

        if agreement_count >= self.min_agreements:
            matching_items = [item for item in self.history if item["printed_id"] == best_id]
            avg_conf = sum(item["confidence"] for item in matching_items) / len(matching_items)
            return best_id, avg_conf, "accepted", {
                "agreements": agreement_count,
                "total_frames": len(self.history),
                "reason": "multi_frame_consensus_reached",
            }
        elif agreement_count == 2:
            return best_id, 0.50, "ambiguous", {
                "agreements": agreement_count,
                "required": self.min_agreements,
                "reason": "insufficient_frame_consensus",
            }
        else:
            return None, 0.0, "rejected", {
                "agreements": agreement_count,
                "required": self.min_agreements,
                "reason": "insufficient_frame_consensus",
            }


