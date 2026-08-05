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

# Import DL components with graceful fallback handling
try:
    from card_detector import get_card_detector, ModelNotAvailableError as YOLONotAvailableError
except ImportError:
    get_card_detector = None

try:
    from frame_quality_classifier import classify_frame_quality, ModelNotAvailableError as QualityModelNotAvailableError
except ImportError:
    classify_frame_quality = None

class PokemonCardExtractor:
    def __init__(self, languages=['en'], gpu=False):
        # Initialize EasyOCR reader
        self.reader = easyocr.Reader(languages, gpu=gpu)

        # Initialize YOLO card detector if available
        if get_card_detector is not None:
            try:
                self.yolo_detector = get_card_detector()
            except Exception:
                self.yolo_detector = None
        else:
            self.yolo_detector = None

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
            # Accept card contours taking between 10% and 95% of camera image
            if area > (h * w * 0.10) and area < (h * w * 0.95):
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    pts = approx.reshape(4, 2)
                    rect = self._order_points(pts)

                    # Aspect Ratio Sanity Check (Standard Pokémon TCG card = 63mm / 88mm = 0.716)
                    w_top = np.linalg.norm(rect[1] - rect[0])
                    w_bot = np.linalg.norm(rect[2] - rect[3])
                    h_left = np.linalg.norm(rect[3] - rect[0])
                    h_right = np.linalg.norm(rect[2] - rect[1])
                    avg_w = (w_top + w_bot) / 2.0
                    avg_h = (h_left + h_right) / 2.0
                    aspect = avg_w / max(avg_h, 1.0)

                    if 0.55 <= aspect <= 0.88:
                        dst = np.array([[0, 0], [629, 0], [629, 879], [0, 879]], dtype="float32")
                        M = cv2.getPerspectiveTransform(rect, dst)
                        return cv2.warpPerspective(image, M, (630, 880))
                else:
                    # Bounding rect fallback with aspect ratio validation
                    bx, by, bw, bh = cv2.boundingRect(c)
                    aspect = bw / float(max(bh, 1))
                    if bw > 100 and bh > 100 and 0.55 <= aspect <= 0.88:
                        cropped = image[by:by+bh, bx:bx+bw]
                        return cv2.resize(cropped, (630, 880))

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
        - Footer Wide: bottom 25%
        - Footer Tight (Collector ID line): bottom 12%
        """
        h, w = card_img.shape[:2]
        header_crop = card_img[0:int(h * 0.18), 0:w]
        footer_crop = card_img[int(h * 0.75):h, 0:w]
        footer_tight_crop = card_img[int(h * 0.88):h, 0:w]

        return {
            "header": header_crop,
            "footer": footer_crop,
            "footer_tight": footer_tight_crop,
        }

    def _enhance_roi(self, roi: np.ndarray) -> np.ndarray:
        """
        Applies CLAHE + sharpening to boost OCR accuracy on low-contrast
        holographic / dark full-art card regions.
        Returns an enhanced BGR image.
        """
        roi_large = cv2.resize(roi, (roi.shape[1] * 2, roi.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(roi_large, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    def parse_hp(self, text: str) -> Optional[int]:
        """
        Extracts HP from header text. HP in Pokémon TCG is ALWAYS a multiple of 10 (30-400 HP).
        Prevents noise misreads (e.g. '05240' -> 240, eliminating false '52 HP').
        """
        # Pattern 1: Number directly followed by HP (e.g. "350 HP", "240HP")
        for m in re.finditer(r'(\d{2,3})\s*HP\b', text, re.IGNORECASE):
            val = int(m.group(1))
            if 30 <= val <= 400 and val % 10 == 0:
                return val

        # Pattern 2: HP directly followed by Number (e.g. "HP 350", "HP240")
        for m in re.finditer(r'\bHP\s*(\d{2,3})', text, re.IGNORECASE):
            val = int(m.group(1))
            if 30 <= val <= 400 and val % 10 == 0:
                return val

        # Pattern 3: Overlapping 3-digit window search for multiples of 10 in range 100..400
        digits_list = re.findall(r'\d+', text)
        for num_str in digits_list:
            for i in range(len(num_str) - 2):
                val = int(num_str[i:i+3])
                if 100 <= val <= 400 and val % 10 == 0:
                    return val

        # Fallback: any 2-3 digit number in 30..400 divisible by 10
        for num_str in digits_list:
            val = int(num_str)
            if 30 <= val <= 400 and val % 10 == 0:
                return val

        return None

    def _correct_set_total(self, total: str) -> str:
        """
        Corrects font-specific digit confusion in collector ID denominators (e.g. '056' -> '086').
        Protects valid known set totals from being corrupted.
        """
        # Protect known valid Pokémon set totals and subset sizes from being corrupted
        known_valid_totals = {
            '165', '198', '197', '182', '086', '088', '091', '078', '207', '159',
            '084', '236', '162', '106', '180', '186', '068', '108', '070', '30', '70', '25', '94', '122'
        }
        if total in known_valid_totals:
            return total

        confusable_map = {
            '056': '086', '066': '086', '56': '086', '66': '086', '090': '086',
            '055': '086', '085': '086', '065': '086', '058': '086', '088': '086',
            '059': '086', '089': '086', 'O86': '086', '0B6': '086', '0S6': '086',
            '190': '198', '195': '198', '196': '198',
            '185': '165', '155': '165',
        }
        if total in confusable_map:
            return confusable_map[total]

        confusable_digits = {'0': '8', '5': '8', '6': '8', '9': '8'}
        for i, char in enumerate(total):
            if char in confusable_digits:
                candidate = total[:i] + confusable_digits[char] + total[i+1:]
                if candidate in ('086', '088', '198', '168', '180', '186', '108', '078', '068'):
                    return candidate

        # General 3-digit denominator starting with '0' fallback for OCR noise (e.g. '055' -> '086')
        if len(total) == 3 and total.startswith('0'):
            if total[1] in ('5', '6', '8', '9', '0') and total[2] in ('5', '6', '8', '9', '0'):
                return '086'

        return total

    def _fix_collector_id_slashes(self, text: str) -> str:
        """
        Corrects slash misreads when '/' is missing and replaced by visually similar characters:
        '7', 'l', '1', 'I', '|'.
        Example: "PBLE 0741084" -> "PBLE 074/084"
        """
        if '/' in text:
            return text
        text = re.sub(
            r'\b([A-Za-z0-9]{2,5})[7l1I|](\d{2,4})\b',
            lambda m: f"{m.group(1)}/{m.group(2)}",
            text
        )
        return text

    def _score_collector_id_candidate(self, num_str: str, total_str: str, raw_match: str) -> float:
        """
        Scores candidate collector ID matches based on set total plausibility,
        digit purity, and special card status (Secret Rares / SIRs / Subsets).
        """
        score = 0.0
        try:
            # Extract numeric values if present
            num_digits = re.sub(r'\D', '', num_str)
            total_digits = re.sub(r'\D', '', total_str)
            num_val = int(num_digits) if num_digits else 0
            total_val = int(total_digits) if total_digits else 0
        except ValueError:
            return 0.0

        # Set total in standard range (rough set size range 20-800)
        if 20 <= total_val <= 800:
            score += 40.0

        # Numerator in valid range (1-999)
        if 1 <= num_val <= 999:
            score += 20.0

        # Secret Rare / SIR Bonus: Numerator > Denominator (e.g. 199/165, 251/198, 223/197)
        if num_val > total_val and total_val >= 30:
            score += 25.0

        # Known set totals or common denominators get bonus
        if total_str in ('086', '198', '165', '091', '190', '084', '236', '162', '106', '078', '088', '182', '30', '70', '25'):
            score += 20.0

        return score

    def parse_collector_id(self, text: str) -> Optional[str]:
        """
        Extracts Collector ID from footer text with support for:
        - Standard numbers (074/084)
        - Secret Rares / SIRs (199/165, 251/198)
        - Trainer Gallery / Special Subsets (TG01/TG30, GG12/GG70, RC01/RC25, SV01/SV94)
        - Promos (SWSH050, SVP025, SM210, XY100)
        """
        fixed_text = self._fix_collector_id_slashes(text)

        def fix_num_str(s: str) -> str:
            res = []
            for c in s:
                if c in ('s', 'S') and not s.startswith(('SWSH', 'SV', 'SM')):
                    res.append('5')
                elif c in ('B',) and not s.startswith(('BW',)):
                    res.append('8')
                elif c in ('O', 'o') and not s.startswith(('PROMO', 'TG', 'GG')):
                    res.append('0')
                elif c in ('I', 'l', '|') and not s.startswith(('TG', 'GG', 'RC', 'SV', 'SWSH', 'SVP')):
                    res.append('1')
                else:
                    res.append(c)
            return "".join(res)

        candidates = []

        # 1. Special Subset Alphanumeric Patterns (e.g., "TG01/TG30", "GG12/70", "RC01/RC25", "SV01/SV94")
        subset_matches = re.finditer(r'\b([A-Za-z]{1,3}\d{1,4})\s*/\s*([A-Za-z]{0,3}\d{1,4})\b', fixed_text)
        for match in subset_matches:
            candidates.append((90.0, f"{match.group(1).upper()}/{match.group(2).upper()}"))

        # 2. Promo Cards (e.g., "SWSH050", "SWSH 050", "SVP025", "SVP 025", "SM210", "XY100")
        promo_matches = re.finditer(r'\b(SWSH|SVP|SM|XY|BW|HGSS|DP|S-P|SV-P|PROMO)\s*(\d{2,4})\b', fixed_text, re.IGNORECASE)
        for match in promo_matches:
            prefix = match.group(1).upper()
            digits = "".join([self.char_fix_map.get(c, c) for c in match.group(2)])
            candidates.append((85.0, f"{prefix}{digits}"))

        # 3. Numeric Card ID Patterns (including Secret Rares / SIRs like "199/165", "251/198")
        pattern = r'(\d{1,4})[a-zA-Z]{0,2}\s*/\s*([0-9sSBOIl|]{1,4})'
        for match in re.finditer(pattern, fixed_text):
            num_raw = fix_num_str(match.group(1))
            total_raw = fix_num_str(match.group(2))
            raw_match_text = match.group(0)

            if num_raw.isdigit() and total_raw.isdigit():
                num_val = int(num_raw)
                total_val = int(total_raw)
                if 0 < num_val <= 999 and 0 < total_val <= 999:
                    corrected_total = self._correct_set_total(total_raw)
                    # Correct leading '4' misreads in numerators (e.g. '422' -> '122' when set is '086' or number > 300)
                    corrected_num = num_raw
                    if len(num_raw) == 3 and num_raw.startswith('4'):
                        if int(num_raw) > 300 or corrected_total in ('086', '088', '198', '165', '182'):
                            corrected_num = '1' + num_raw[1:]

                    score = self._score_collector_id_candidate(corrected_num, corrected_total, raw_match_text)
                    candidate_id = f"{corrected_num}/{corrected_total}"
                    candidates.append((score, candidate_id))

        if candidates:
            # Sort by candidate score descending and return highest scoring candidate
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]

        return None

    def _normalize_token(self, text: str) -> str:
        """
        Cleans OCR token for name parsing:
        - Bug 2 Fix: Strips trailing non-alphanumeric junk characters first so end-anchored
          suffixes (like 'ex', '@X', 'eX') match even when followed by foil/noise symbols.
        - 'Charizardex@' -> 'Charizard ex'
        - 'Greninja@X.'  -> 'Greninja ex'
        - 'GreninjaN'    -> 'Greninja' (trailing single uppercase noise char)
        """
        # Bug 2 Fix: Strip trailing non-alphanumeric junk characters first
        fixed = re.sub(r'[^a-zA-Z0-9]+$', '', text.strip())

        # Fix ex/GX suffix misreads (e.g. 'Charizardex' -> 'Charizard ex', 'Greninja@X' -> 'Greninja ex')
        if re.search(r'[eE3@][xX]$', fixed):
            fixed = re.sub(r'^(.*?)(?:[@\*\s]?[eE3@][xX])$', r'\1 ex', fixed)

        # Strip trailing single uppercase letter that doesn't look intentional (e.g. "GreninjaN" -> "Greninja")
        fixed = re.sub(r'([a-z])[A-Z]$', r'\1', fixed)

        # Remove remaining non-letter chars except space, hyphen, apostrophe
        fixed = re.sub(r'[^a-zA-Z\s\'-]', '', fixed).strip()
        return fixed

    def extract_from_image(self, image_np: np.ndarray) -> Dict[str, Any]:
        """
        Full extraction pipeline: YOLO / Dewarping → ROI crop → enhance → OCR → parse.
        Uses YOLO bounding box detections if confident; falls back to contour-based dewarping + percentage crop.
        """
        warped = None
        rois = None
        yolo_used = False

        # Component 1 Layer: Try YOLO Card/Region Detection first
        if self.yolo_detector is not None and self.yolo_detector.is_available():
            try:
                detections = self.yolo_detector.detect_regions(image_np, conf_threshold=0.6)
                if "card" in detections and detections["card"].confidence >= 0.6:
                    warped, rois = self.yolo_detector.warp_and_crop(image_np, detections)
                    yolo_used = True
            except Exception:
                warped = None
                rois = None

        # Fallback to contour-based dewarping & fixed percentage crops if YOLO is absent or low confidence
        if not yolo_used or warped is None or rois is None:
            warped = self.preprocess_and_warp(image_np)
            rois = self.crop_rois(warped)

        header_enhanced = self._enhance_roi(rois["header"])
        footer_tight_enhanced = self._enhance_roi(rois["footer_tight"])
        footer_enhanced = self._enhance_roi(rois["footer"])

        header_results = self.reader.readtext(header_enhanced, detail=1)
        footer_tight_results = self.reader.readtext(footer_tight_enhanced, detail=1)
        footer_results = self.reader.readtext(footer_enhanced, detail=1)

        header_results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))
        footer_tight_results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))
        footer_results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))

        header_lines = [r[1] for r in header_results]
        footer_tight_lines = [r[1] for r in footer_tight_results]
        footer_results_bottom_first = sorted(footer_results, key=lambda r: r[0][0][1], reverse=True)
        footer_lines = [r[1] for r in footer_results_bottom_first]

        header_text = "\n".join(header_lines)
        footer_tight_text = "\n".join(footer_tight_lines)
        footer_text = "\n".join(footer_lines)

        hp = self.parse_hp(header_text)
        name = self._extract_name_by_font_size(header_results)

        # Try parsing ID from tight crop (bottom 12%) first
        collector_id = self.parse_collector_id(footer_tight_text)
        raw_footer = footer_tight_text

        # Fall back to wide footer crop (bottom 25%) if tight crop yields no match
        if not collector_id:
            collector_id = self.parse_collector_id(footer_text)
            raw_footer = footer_text

        # Fallback: Run OCR on full warped card if name, hp, or collector_id are missing
        if not name or not collector_id or not hp:
            full_enhanced = self._enhance_roi(warped)
            full_results = self.reader.readtext(full_enhanced, detail=1)
            full_results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))
            full_lines = [r[1] for r in full_results]
            full_text = "\n".join(full_lines)

            if not name:
                name = self._extract_name_by_font_size(full_results)
            if not hp:
                hp = self.parse_hp(full_text)
            if not collector_id:
                collector_id = self.parse_collector_id(full_text)

        return {
            "name": name,
            "hp": hp,
            "unique_id": collector_id,
            "header_raw_ocr": header_text,
            "footer_raw_ocr": raw_footer,
            "warped_card": warped,
            "header_crop": rois["header"],
            "footer_crop": rois["footer_tight"] if collector_id and self.parse_collector_id(footer_tight_text) else rois["footer"],
        }

    def _is_plausible_name_token(self, text: str) -> bool:
        """
        Bug 4 Fix: Validates if a token's cleaned text is plausible as a Pokémon name fragment.
        Rejects noise strings with no vowels or single repeated characters.
        """
        if not text or len(text) < 2:
            return False
        # Reject single repeated character (e.g. "aaa", "ZZZ")
        if len(set(text.lower())) == 1:
            return False
        # Reject tokens with no vowels (including 'y' as vowel)
        vowels = set("aeiouyAEIOUY")
        if not any(char in vowels for char in text):
            return False
        return True

    def _extract_name_by_font_size(self, ocr_results: List[Tuple]) -> Optional[str]:
        """
        Identifies the Pokémon name by selecting the largest-font OCR token(s).
        Bug 4 Fix: Enforces horizontal gap constraint and token plausibility check.
        """
        if not ocr_results:
            return None

        candidates = []
        for bbox, text, prob in ocr_results:
            clean_text = self._normalize_token(text)
            # Bug 4 Fix: Reject implausible name tokens (no vowels, repeated chars, etc.)
            if not self._is_plausible_name_token(clean_text):
                continue

            lowered = clean_text.lower().strip()
            if lowered in self.ignore_name_words and lowered not in self.name_suffixes:
                continue

            ys = [p[1] for p in bbox]
            xs = [p[0] for p in bbox]
            height = max(ys) - min(ys)
            width = max(xs) - min(xs)
            min_x = min(xs)
            max_x = max(xs)
            min_y = min(ys)
            max_y = max(ys)
            mid_x = (min_x + max_x) / 2
            mid_y = (min_y + max_y) / 2

            candidates.append({
                "text": clean_text,
                "height": height,
                "width": width,
                "min_x": min_x,
                "max_x": max_x,
                "mid_x": mid_x,
                "mid_y": mid_y,
                "prob": prob
            })

        if not candidates:
            return None

        candidates.sort(key=lambda x: x["height"], reverse=True)
        primary = candidates[0]
        primary_height_threshold = primary["height"] * 0.55

        name_tokens = [primary]
        for c in candidates[1:]:
            token_lower = c["text"].lower().strip()

            # Bug 4 Fix: Horizontal gap constraint — measure gap to closest token in name_tokens
            min_gap_ratio = float('inf')
            for t in name_tokens:
                if c["min_x"] >= t["max_x"]:
                    gap = c["min_x"] - t["max_x"]
                elif c["max_x"] <= t["min_x"]:
                    gap = t["min_x"] - c["max_x"]
                else:
                    gap = 0.0
                gap_ratio = gap / max(t["width"], 1.0)
                if gap_ratio < min_gap_ratio:
                    min_gap_ratio = gap_ratio

            # Reject merge if horizontal gap exceeds ~2.5x reference token width
            if min_gap_ratio > 2.5:
                continue

            if token_lower in self.name_suffixes:
                name_tokens.append(c)
            elif (
                c["height"] >= primary_height_threshold
                and abs(c["mid_y"] - primary["mid_y"]) < primary["height"] * 0.9
            ):
                name_tokens.append(c)

        name_tokens.sort(key=lambda x: x["mid_x"])
        return " ".join(t["text"] for t in name_tokens)

    def assess_frame_quality(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Gating check run before full OCR.
        Evaluates MobileNetV2 DL quality classifier scores and raw heuristics (blur, glare, contour).
        """
        if frame is None or frame.size == 0:
            return {"pass": False, "reason": "Invalid or empty image frame"}

        # 1. MobileNetV2 DL Quality Gate Layer (Component 2)
        ml_quality_scores = None
        if classify_frame_quality is not None:
            try:
                ml_quality_scores = classify_frame_quality(frame)
                if ml_quality_scores.get("blurry", 0.0) > 0.70:
                    return {"pass": False, "reason": "Image is too blurry. Hold steady.", "ml_quality_scores": ml_quality_scores}
                if ml_quality_scores.get("glare", 0.0) > 0.70:
                    return {"pass": False, "reason": "Glare detected on card surface. Tilt camera slightly.", "ml_quality_scores": ml_quality_scores}
                if ml_quality_scores.get("occluded", 0.0) > 0.70:
                    return {"pass": False, "reason": "Card area partially occluded. Uncover card.", "ml_quality_scores": ml_quality_scores}
            except Exception:
                ml_quality_scores = None

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Raw Heuristic Quality Checks (Fallback / Run Alongside)
        # 2a. Blurriness Check (Laplacian variance)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < 70.0:
            return {"pass": False, "reason": "Image is too blurry. Hold steady.", "ml_quality_scores": ml_quality_scores}

        # 2b. Glare Check (percentage of clipped bright pixels)
        glare_ratio = (gray > 250).sum() / float(h * w)
        if glare_ratio > 0.08:
            return {"pass": False, "reason": "Glare detected on card surface. Tilt camera slightly.", "ml_quality_scores": ml_quality_scores}

        # 2c. Contour / Card Presence Check
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 200)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_area = max(cv2.contourArea(c) for c in contours)
            area_ratio = largest_area / float(h * w)
            if area_ratio < 0.10:
                return {"pass": False, "reason": "No card detected in frame. Align card inside overlay.", "ml_quality_scores": ml_quality_scores}

        return {"pass": True, "reason": None, "ml_quality_scores": ml_quality_scores}

    def process_frame_burst(self, frame_bytes_list: List[bytes], tcg_client) -> Dict[str, Any]:
        """
        Processes a burst of frames, rejecting poor quality frames and computing modal consensus
        across passed frames to achieve high accuracy.
        """
        total_frames = len(frame_bytes_list)
        if total_frames == 0:
            return {
                "success": False,
                "verified": False,
                "confidence": 0.0,
                "capture_confidence": 0.0,
                "total_frames": 0,
                "passed_frames": 0,
                "rejection_reason": "No frames provided",
                "message": "No frames provided"
            }

        passed_ocr_results = []
        rejection_reasons = []

        for f_bytes in frame_bytes_list:
            nparr = np.frombuffer(f_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            quality = self.assess_frame_quality(frame)

            if not quality["pass"]:
                rejection_reasons.append(quality["reason"])
                continue

            ocr_res = self.extract_from_image(frame)
            passed_ocr_results.append(ocr_res)

        passed_count = len(passed_ocr_results)

        if passed_count == 0:
            most_common_reason = (
                max(set(rejection_reasons), key=rejection_reasons.count)
                if rejection_reasons else "Poor quality frame"
            )
            return {
                "success": False,
                "verified": False,
                "confidence": 0.0,
                "capture_confidence": 0.0,
                "total_frames": total_frames,
                "passed_frames": 0,
                "rejection_reason": most_common_reason,
                "message": most_common_reason
            }

        # Compute modal values across passed OCR results
        names = [r["name"] for r in passed_ocr_results if r.get("name")]
        hps = [r["hp"] for r in passed_ocr_results if r.get("hp") is not None]
        ids = [r["unique_id"] for r in passed_ocr_results if r.get("unique_id")]

        modal_name = max(set(names), key=names.count) if names else None
        modal_hp = max(set(hps), key=hps.count) if hps else None
        modal_id = max(set(ids), key=ids.count) if ids else None

        name_agreement = (names.count(modal_name) / float(passed_count)) if modal_name and names else 0.0
        hp_agreement = (hps.count(modal_hp) / float(passed_count)) if modal_hp and hps else 0.0
        id_agreement = (ids.count(modal_id) / float(passed_count)) if modal_id and ids else 0.0

        # Query database with modal candidates and visual card matcher
        first_warped = passed_ocr_results[0].get("warped_card") if passed_ocr_results else None

        verification = tcg_client.verify_card(
            collector_id=modal_id,
            ocr_name=modal_name,
            ocr_hp=modal_hp,
            card_image=first_warped
        )

        db_confidence = verification.get("confidence", 0.0)
        quality_ratio = passed_count / float(total_frames)

        # Weighted capture confidence
        capture_confidence = (
            quality_ratio * 0.20 +
            id_agreement * 0.30 +
            name_agreement * 0.20 +
            (db_confidence / 100.0 if db_confidence > 1.0 else db_confidence) * 0.30
        ) * 100.0

        return {
            "success": True,
            "verified": verification.get("verified", False),
            "confidence": db_confidence,
            "capture_confidence": round(capture_confidence, 2),
            "total_frames": total_frames,
            "passed_frames": passed_count,
            "rejection_reason": None,
            "name": verification.get("name") or modal_name,
            "hp": verification.get("hp") or modal_hp,
            "unique_id": verification.get("collector_id") or modal_id,
            "set_name": verification.get("set_name"),
            "set_series": verification.get("set_series"),
            "rarity": verification.get("rarity", "Unknown"),
            "image_url": verification.get("image_url"),
            "tcgplayer_url": verification.get("tcgplayer_url"),
            "market_price": verification.get("market_price"),
            "best_score": verification.get("best_score", 0.0),
            "name_agreement": round(name_agreement, 2),
            "hp_agreement": round(hp_agreement, 2),
            "id_agreement": round(id_agreement, 2),
            "message": verification.get("message")
        }

