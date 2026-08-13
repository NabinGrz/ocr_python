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
import json
import os
import cv2
import numpy as np
from collections import Counter
from difflib import SequenceMatcher
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

# PaddleOCR is imported lazily inside paddle_ocr property to keep startup RAM minimal
PaddleOCR = None

from custom_card_recognizer import (
    RestrictedAlphabetRecognizer,
    CustomPokemonCardRecognizer,
    ClosedSetCardNameMatcher,
    PrintedIDParser,
    MultiFrameIDVoter,
)

class PokemonCardExtractor:
    def __init__(self, languages=['en'], gpu=False):
        self.languages = languages
        self.gpu = gpu
        self._reader = None
        self._paddle_ocr = None
        self._paddle_initialized = False
        self._yolo_detector = None
        self._yolo_initialized = False

        # Path 2 Engine A & B: Initialize custom Pokémon recognizer and restricted alphabet classifier
        self.restricted_recognizer = RestrictedAlphabetRecognizer()
        self.custom_recognizer = CustomPokemonCardRecognizer(self.restricted_recognizer)

        # Closed-set retrieval matcher for Pokémon card names
        self.closed_set_matcher = ClosedSetCardNameMatcher(catalog_path="models/card_catalog.json")

        # Printed ID Parser and Multi-Frame Consensus Voter
        self.printed_id_parser = PrintedIDParser()
        self.multi_frame_voter = MultiFrameIDVoter(min_agreements=3)

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

        # Known valid Pokémon set totals / subset sizes — shared by
        # _correct_set_total (protection) and _score_collector_id_candidate
        # (bonus scoring), so both stay in sync.
        built_in_totals = {
            '165', '198', '197', '182', '086', '088', '091', '078', '207', '159',
            '084', '236', '162', '106', '180', '186', '068', '108', '070',
            '30', '70', '25', '94', '122',
        }
        catalog_path = "models/card_catalog.json"
        if os.path.exists(catalog_path):
            try:
                with open(catalog_path, "r", encoding="utf-8") as source:
                    catalog = json.load(source)
                catalog_cards = catalog.get("cards", []) if isinstance(catalog, dict) else catalog
                built_in_totals.update(
                    str(card.get("set", {}).get("printedTotal"))
                    for card in catalog_cards
                    if card.get("set", {}).get("printedTotal") is not None
                )
            except Exception:
                pass
        self.known_valid_totals = frozenset(built_in_totals)

        # Exact OCR-confusion strings -> correct total.
        self.confusable_total_map = {
            '056': '086', '066': '086', '56': '086', '66': '086', '090': '086',
            '055': '086', '085': '086', '065': '086', '058': '086',
            '059': '086', '089': '086', 'O86': '086', '0B6': '086', '0S6': '086',
            '190': '198', '195': '198', '196': '198',
            '185': '165', '155': '165',
        }
        self.confusable_digit_subs = {'0': '8', '5': '8', '6': '8', '9': '8'}
        self.total_substitution_targets = frozenset(
            {'086', '088', '198', '168', '180', '186', '108', '078', '068'}
        )

    @property
    def reader(self):
        """Lazy loader for EasyOCR reader instance."""
        if self._reader is None:
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("MKL_NUM_THREADS", "1")
            os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
            try:
                import torch
                torch.set_grad_enabled(False)
                torch.set_num_threads(1)
            except Exception:
                pass
            import easyocr
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
        return self._reader

    @property
    def paddle_ocr(self):
        """Lazy loader for PaddleOCR instance."""
        if not self._paddle_initialized:
            self._paddle_initialized = True
            try:
                from paddleocr import PaddleOCR
                self._paddle_ocr = PaddleOCR(lang='en')
            except Exception:
                self._paddle_ocr = None
        return self._paddle_ocr

    @paddle_ocr.setter
    def paddle_ocr(self, value):
        self._paddle_ocr = value
        self._paddle_initialized = True

    @property
    def yolo_detector(self):
        """Lazy loader for YOLO detector instance."""
        if not self._yolo_initialized:
            self._yolo_initialized = True
            if get_card_detector is not None:
                try:
                    self._yolo_detector = get_card_detector()
                except Exception:
                    self._yolo_detector = None
        return self._yolo_detector

    @yolo_detector.setter
    def yolo_detector(self, value):
        self._yolo_detector = value
        self._yolo_initialized = True

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
        Crops ROIs for OCR using normalized coordinates relative to the rectified card:
        - Header (Name & HP): ymin=0.00, ymax=0.18, xmin=0.00, xmax=1.00
        - Footer Left Surgical (Tight bottom-band for standard fraction / SVP promo): ymin=0.88, ymax=0.985, xmin=0.015, xmax=0.50
        - Footer Left (Standard ID & illustrator context): ymin=0.83, ymax=0.985, xmin=0.015, xmax=0.55
        - Footer Right Surgical (Tight bottom-band for SWSH/XY/BW promo): ymin=0.88, ymax=0.985, xmin=0.50, xmax=0.985
        - Footer Right (Promo/XY ID): ymin=0.83, ymax=0.985, xmin=0.48, xmax=0.985
        - Footer Wide (Full bottom context): ymin=0.86, ymax=0.985, xmin=0.02, xmax=0.98
        """
        h, w = card_img.shape[:2]

        def crop_norm(ymin, ymax, xmin, xmax):
            y1, y2 = int(h * ymin), int(h * ymax)
            x1, x2 = int(w * xmin), int(w * xmax)
            return card_img[y1:y2, x1:x2]

        header_crop = crop_norm(0.00, 0.18, 0.00, 1.00)
        footer_left_surgical = crop_norm(0.88, 0.985, 0.015, 0.50)
        footer_left_crop = crop_norm(0.83, 0.985, 0.015, 0.55)
        footer_right_surgical = crop_norm(0.88, 0.985, 0.50, 0.985)
        footer_right_crop = crop_norm(0.83, 0.985, 0.48, 0.985)
        footer_wide_crop = crop_norm(0.86, 0.985, 0.02, 0.98)

        return {
            "header": header_crop,
            "footer_left_surgical": footer_left_surgical,
            "footer_left": footer_left_crop,
            "footer_right_surgical": footer_right_surgical,
            "footer_right": footer_right_crop,
            "footer_wide": footer_wide_crop,
            "footer_tight": footer_left_surgical,
            "footer": footer_wide_crop,
        }

    def save_debug_roi_image(
        self, card_img: np.ndarray, output_path: str = "debug_crops/debug_rectified_roi.png"
    ) -> str:
        """
        Saves a debug crop image with colored ROI bounding boxes drawn on the rectified card.
        """
        if card_img is None or card_img.size == 0:
            return ""

        debug_img = card_img.copy()
        h, w = debug_img.shape[:2]

        rois_to_draw = [
            (0.00, 0.18, 0.00, 1.00, (255, 0, 0), "HEADER (0.00-0.18)"),
            (0.88, 0.985, 0.015, 0.50, (0, 255, 0), "FOOTER_LEFT_SURGICAL (0.88-0.985, 0.015-0.50)"),
            (0.83, 0.985, 0.015, 0.55, (0, 200, 100), "FOOTER_LEFT (0.83-0.985, 0.015-0.55)"),
            (0.88, 0.985, 0.50, 0.985, (0, 0, 255), "FOOTER_RIGHT_SURGICAL (0.88-0.985, 0.50-0.985)"),
            (0.83, 0.985, 0.48, 0.985, (100, 0, 200), "FOOTER_RIGHT (0.83-0.985, 0.48-0.985)"),
        ]

        for ymin, ymax, xmin, xmax, color, label in rois_to_draw:
            pt1 = (int(w * xmin), int(h * ymin))
            pt2 = (int(w * xmax), int(h * ymax))
            cv2.rectangle(debug_img, pt1, pt2, color, 2)
            cv2.putText(debug_img, label, (pt1[0] + 5, pt1[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, debug_img)
        return output_path

    def save_all_debug_pipeline_images(
        self,
        raw_image: np.ndarray,
        warped_card: np.ndarray,
        rois: Dict[str, np.ndarray],
        output_dir: str = "debug_crops/latest",
    ) -> Dict[str, str]:
        """
        Persists all stages of image transformation after client uploads:
        1. 01_raw_upload.png - Unmodified incoming client frame
        2. 02_warped_card.png - Dewarped rectified card
        3. 03_rectified_rois.png - Bounding box visualization across header and footer strips
        4. variant_{roi}_{filter}.png - Every manipulated preprocessing variant for OCR
        """
        os.makedirs(output_dir, exist_ok=True)

        # Clear previous run images in output_dir
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isfile(item_path) and item.lower().endswith((".png", ".jpg", ".jpeg")):
                try:
                    os.remove(item_path)
                except OSError:
                    pass

        saved_files = {}

        # 1. Raw upload
        if raw_image is not None and raw_image.size > 0:
            raw_path = os.path.join(output_dir, "01_raw_upload.png")
            cv2.imwrite(raw_path, raw_image)
            saved_files["raw_upload"] = raw_path

        # 2. Warped card
        if warped_card is not None and warped_card.size > 0:
            warped_path = os.path.join(output_dir, "02_warped_card.png")
            cv2.imwrite(warped_path, warped_card)
            saved_files["warped_card"] = warped_path

            # 3. Rectified ROIs visualizer
            roi_vis_path = os.path.join(output_dir, "03_rectified_rois.png")
            self.save_debug_roi_image(warped_card, output_path=roi_vis_path)
            saved_files["rectified_rois"] = roi_vis_path

        # 4. All manipulated preprocessing variants for each ROI
        for roi_name, roi_img in rois.items():
            if roi_name in ("footer_tight", "footer"):
                continue  # skip duplicate aliases
            if roi_img is None or roi_img.size == 0:
                continue

            variants = self.generate_preprocessing_variants(roi_img)
            for var_name, var_img in variants.items():
                if var_img is not None and var_img.size > 0:
                    var_file_name = f"variant_{roi_name}_{var_name}.png"
                    var_path = os.path.join(output_dir, var_file_name)
                    cv2.imwrite(var_path, var_img)
                    saved_files[f"{roi_name}_{var_name}"] = var_path

        return saved_files

    def generate_preprocessing_variants(self, crop: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Takes crop and upscales to ensure target character height >= 36-48px.
        Generates dynamic polarity and morphology-aware preprocessing variants:
        - original_color / orig
        - grayscale_norm
        - clahe
        - blackhat / inv_th_bh (for gold foil / dark text on bright foil)
        - tophat / th_tophat (for light text on dark backgrounds)
        - dark_text_light_bg (Otsu)
        - light_text_dark_bg (Inv-Otsu)
        - adaptive_thresh
        """
        if crop is None or crop.size == 0:
            return {}

        ch, cw = crop.shape[:2]
        scale = max(2.5, 42.0 / max(ch / 2.5, 8.0))
        target_w = int(cw * scale)
        target_h = int(ch * scale)
        upscaled = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY) if len(upscaled.shape) == 3 else upscaled.copy()
        norm_gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        clahe_engine = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        clahe_gray = clahe_engine.apply(norm_gray)

        variants = {
            "original_color": upscaled,
            "grayscale_norm": cv2.cvtColor(norm_gray, cv2.COLOR_GRAY2BGR),
            "clahe": cv2.cvtColor(clahe_gray, cv2.COLOR_GRAY2BGR),
        }

        # Adaptive threshold
        adaptive_thresh = cv2.adaptiveThreshold(
            clahe_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4
        )
        variants["adaptive_thresh"] = cv2.cvtColor(adaptive_thresh, cv2.COLOR_GRAY2BGR)

        # Black-Hat Morphology (crucial for gold/foil cards like Mega Greninja)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        bh_norm = cv2.normalize(bh, None, 0, 255, cv2.NORM_MINMAX)
        _, th_bh = cv2.threshold(bh_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants["blackhat"] = cv2.cvtColor(255 - bh_norm, cv2.COLOR_GRAY2BGR)
        variants["inv_th_bh"] = cv2.cvtColor(255 - th_bh, cv2.COLOR_GRAY2BGR)

        # Top-Hat Morphology (for dark background foil)
        th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        th_norm = cv2.normalize(th, None, 0, 255, cv2.NORM_MINMAX)
        _, th_th = cv2.threshold(th_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants["tophat"] = cv2.cvtColor(255 - th_norm, cv2.COLOR_GRAY2BGR)
        variants["th_tophat"] = cv2.cvtColor(255 - th_th, cv2.COLOR_GRAY2BGR)

        # Otsu thresholding
        _, dark_text = cv2.threshold(clahe_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        light_text = cv2.bitwise_not(dark_text)
        variants["dark_text_light_bg"] = cv2.cvtColor(dark_text, cv2.COLOR_GRAY2BGR)
        variants["light_text_dark_bg"] = cv2.cvtColor(light_text, cv2.COLOR_GRAY2BGR)

        return variants

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

    def _run_paddle_ocr(self, roi: np.ndarray) -> List[Tuple[list, str, float]]:
        """Run PaddleOCR/PP-OCR recognizer as standard OCR Path 1 Engine B."""
        if self.paddle_ocr is None or roi is None or roi.size == 0:
            return []
        try:
            res = self.paddle_ocr.ocr(roi)
            formatted = []
            if not res:
                return formatted
            for item in res:
                if isinstance(item, dict):
                    texts = item.get("rec_texts", [])
                    scores = item.get("rec_scores", [])
                    polys = item.get("rec_polys", item.get("rec_boxes", []))
                    for i, text in enumerate(texts):
                        score = float(scores[i]) if i < len(scores) else 0.9
                        poly = polys[i].tolist() if hasattr(polys[i], "tolist") else (polys[i] if i < len(polys) else [[0, 0], [10, 0], [10, 10], [0, 10]])
                        formatted.append((poly, str(text), score))
                elif isinstance(item, list):
                    for sub_item in item:
                        if isinstance(sub_item, (list, tuple)) and len(sub_item) == 2:
                            box, val = sub_item
                            if isinstance(val, (list, tuple)) and len(val) == 2:
                                text, prob = val
                                box = box.tolist() if hasattr(box, "tolist") else box
                                formatted.append((box, str(text), float(prob)))
            return formatted
        except Exception:
            return []

    def _ocr_variants(
        self,
        roi: np.ndarray,
        allowlist: Optional[str] = None,
        field_type: str = "all",
    ) -> List[List[Tuple]]:
        """
        Runs multiple recognition paths for each ROI:
        - Path 1: Standard OCR Recognizers (EasyOCR & PaddleOCR/PP-OCR)
        - Path 2: Custom Pokémon Card Recognizer & Restricted Alphabet Recognizer
        """
        if roi is None or roi.size == 0:
            return []

        upscaled = cv2.resize(
            roi,
            (roi.shape[1] * 2, roi.shape[0] * 2),
            interpolation=cv2.INTER_CUBIC,
        )
        enhanced = self._enhance_roi(roi)
        variants = [enhanced, upscaled]
        results = []

        # Path 1 Engine A: EasyOCR
        for variant in variants:
            kwargs = {
                "detail": 1,
                "decoder": "beamsearch",
                "beamWidth": 5,
                "paragraph": False,
            }
            if allowlist:
                kwargs["allowlist"] = allowlist
            ocr_result = self.reader.readtext(variant, **kwargs)
            ocr_result.sort(key=lambda r: (r[0][0][1], r[0][0][0]))
            results.append(ocr_result)

        # Path 1 Engine B: PaddleOCR / PP-OCR
        if self.paddle_ocr is not None:
            for variant in variants:
                paddle_res = self._run_paddle_ocr(variant)
                if paddle_res:
                    paddle_res.sort(key=lambda r: (r[0][0][1], r[0][0][0]))
                    results.append(paddle_res)

        # Path 2: Custom Recognizer & Restricted Alphabet Classifier
        if field_type in ("collector_id", "hp"):
            restricted_res = self.restricted_recognizer.recognize_crop(roi, field_type=field_type)
            if restricted_res:
                results.append(restricted_res)

        if field_type == "header":
            custom_res = self.custom_recognizer.recognize_header(roi)
            if custom_res:
                results.append(custom_res)

        return results

    @staticmethod
    def _modal_value(values: List[Any]) -> Optional[Any]:
        """Return a deterministic mode while preserving first-seen tie-breaking."""
        if not values:
            return None
        counts = Counter(values)
        best_count = max(counts.values())
        return next(value for value in values if counts[value] == best_count)

    @staticmethod
    def _fuzzy_name_consensus(names: List[str]) -> Tuple[Optional[str], float]:
        """Choose the name most similar to all observations and report fuzzy agreement."""
        if not names:
            return None, 0.0
        if len(names) == 1:
            return names[0], 1.0

        def normalized(value: str) -> str:
            return re.sub(r"[^a-z]", "", value.lower())

        similarities = []
        for candidate in names:
            candidate_norm = normalized(candidate)
            mean_similarity = sum(
                SequenceMatcher(None, candidate_norm, normalized(other)).ratio()
                for other in names
            ) / len(names)
            similarities.append(mean_similarity)

        best_index = max(
            range(len(names)),
            key=lambda idx: (
                similarities[idx],
                bool(re.search(r"\s(?:ex|gx|v|vmax|vstar)$", names[idx], re.IGNORECASE)),
            ),
        )
        consensus = names[best_index]
        consensus_norm = normalized(consensus)
        agreeing = sum(
            SequenceMatcher(None, consensus_norm, normalized(other)).ratio() >= 0.78
            for other in names
        )
        return consensus, agreeing / float(len(names))

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
        if total in self.known_valid_totals:
            return total

        corrected = self.confusable_total_map.get(total)
        if corrected is not None:
            return corrected

        for i, char in enumerate(total):
            repl = self.confusable_digit_subs.get(char)
            if repl is not None:
                candidate = total[:i] + repl + total[i + 1:]
                if candidate in self.total_substitution_targets:
                    return candidate

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
        if total_str in self.known_valid_totals:
            score += 20.0

        return score

    def parse_collector_id(self, text: str) -> Optional[str]:
        """
        Extracts Collector ID from footer text using PrintedIDParser.
        Supports formats:
        - XY124 / XY 124 (Promo prefix)
        - 4/102, 124/165 (Standard numeric fraction)
        - 4 of 102 (Word 'of' fraction)
        - SV124/198, TG01/TG30 (Subset / Alphanumeric fraction)
        - SV01a (Suffix ID)
        """
        parsed_id, conf, pattern = self.printed_id_parser.parse_printed_id(text)
        if parsed_id and conf >= 0.85:
            return parsed_id

        # Fallback to pattern rules
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
        subset_matches = re.finditer(r'\b(TG|GG|SV|RC|CRZ|H|SH|SL|CL)(\d{1,4})\s*/\s*([A-Za-z]{0,3}\d{1,4})\b', fixed_text, re.IGNORECASE)
        for match in subset_matches:
            candidates.append((90.0, f"{match.group(1).upper()}{match.group(2)}/{match.group(3).upper()}"))

        # 2. Promo Cards (e.g., "SWSH050", "SWSH 050", "SVP025", "SVP 025", "SM210", "XY100")
        promo_matches = re.finditer(r'\b(SWSH|SVP|SM|XY|BW|HGSS|DP|S-P|SV-P|PROMO)\s*(\d{2,4})\b', fixed_text, re.IGNORECASE)
        for match in promo_matches:
            prefix = match.group(1).upper()
            digits = "".join([self.char_fix_map.get(c, c) for c in match.group(2)])
            candidates.append((95.0, f"{prefix}{digits}"))

        # 3. Numeric Card ID Patterns (including Secret Rares / SIRs like "199/165", "251/198")
        pattern = r'(\d{1,4})[a-zA-Z]{0,2}\s*/\s*([0-9sSBOIl|]{1,4})'
        for match in re.finditer(pattern, fixed_text):
            num_raw = fix_num_str(match.group(1))
            total_raw = fix_num_str(match.group(2))
            raw_match_text = match.group(0)

            if num_raw.isdigit() and total_raw.isdigit():
                num_val = int(num_raw)
                total_val = int(total_raw)
                if 0 < num_val <= 999 and 10 <= total_val <= 999:
                    corrected_total = self._correct_set_total(total_raw)
                    corrected_num = num_raw
                    if len(num_raw) == 3 and num_raw.startswith('4'):
                        if int(num_raw) > 300 or corrected_total in ('086', '088', '198', '165', '182'):
                            corrected_num = '1' + num_raw[1:]

                    score = self._score_collector_id_candidate(corrected_num, corrected_total, raw_match_text)
                    candidate_id = f"{corrected_num}/{corrected_total}"
                    candidates.append((score, candidate_id))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]

        return None

    def _normalize_token(self, text: str) -> str:
        fixed = re.sub(r'[^a-zA-Z0-9]+$', '', text.strip())

        if re.search(r'[eE3@][xX]$', fixed):
            fixed = re.sub(r'^(.*?)(?:[@\*\s]?[eE3@][xX])$', r'\1 ex', fixed)

        fixed = re.sub(r'([a-z])[A-Z]$', r'\1', fixed)

        fixed = re.sub(r'[^a-zA-Z\s\'-]', '', fixed).strip()
        return fixed

    def extract_from_image(self, image_np: np.ndarray, save_debug: bool = False) -> Dict[str, Any]:
        """
        Full extraction pipeline: YOLO / Dewarping → ROI crop → 6-variant enhance → OCR → parse.
        Uses YOLO bounding box detections if confident; falls back to contour-based dewarping + ratio crop.
        """
        warped = None
        rois = None
        yolo_used = False

        if self.yolo_detector is not None and self.yolo_detector.is_available():
            try:
                detections = self.yolo_detector.detect_regions(image_np, conf_threshold=0.6)
                if "card" in detections and detections["card"].confidence >= 0.6:
                    warped, rois = self.yolo_detector.warp_and_crop(image_np, detections)
                    yolo_used = True
            except Exception:
                warped = None
                rois = None

        if not yolo_used or warped is None or rois is None:
            warped = self.preprocess_and_warp(image_np)
            rois = self.crop_rois(warped)

        # 1. Optionally save complete debug images after client sends upload
        debug_crop_path = ""
        debug_files: Dict[str, str] = {}
        if save_debug:
            debug_files = self.save_all_debug_pipeline_images(image_np, warped, rois, output_dir="debug_crops/latest")
            debug_crop_path = debug_files.get("rectified_rois", "debug_crops/latest/03_rectified_rois.png")

        # 2. Header Extraction
        header_variants = self._ocr_variants(rois["header"], field_type="header")
        header_texts = ["\n".join(r[1] for r in results) for results in header_variants]

        name_candidates = [
            candidate
            for candidate in (self._extract_name_by_font_size(results) for results in header_variants)
            if candidate
        ]
        hp_candidates = [
            candidate for candidate in (self.parse_hp(text) for text in header_texts) if candidate is not None
        ]

        closed_set_name, closed_set_conf = self.closed_set_matcher.resolve_candidates(name_candidates)
        if closed_set_name and closed_set_conf >= 0.55:
            name = closed_set_name
        else:
            name, _ = self._fuzzy_name_consensus(name_candidates)

        hp = self._modal_value(hp_candidates)
        header_text = "\n--- OCR PASS ---\n".join(header_texts)

        # 3. Printed ID Extraction with Morphology Variants & Multi-Path Footer ROIs
        raw_ocr_outputs: Dict[str, str] = {}
        id_observations: List[str] = []
        printed_id_roi_source: str = "none"
        collector_id: Optional[str] = None
        parsed_candidates: List[Tuple[float, str, str]] = []

        id_allowlist = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/- :[]|"
        eval_rois = [
            ("footer_left_surgical", rois.get("footer_left_surgical")),
            ("footer_left", rois.get("footer_left")),
            ("footer_right_surgical", rois.get("footer_right_surgical")),
            ("footer_right", rois.get("footer_right")),
            ("footer_wide", rois.get("footer_wide")),
        ]

        for roi_name, roi_crop in eval_rois:
            if roi_crop is None or roi_crop.size == 0:
                continue

            variants_dict = self.generate_preprocessing_variants(roi_crop)
            for v_name, v_img in variants_dict.items():
                ocr_results = self.reader.readtext(
                    v_img,
                    detail=1,
                    decoder="beamsearch",
                    beamWidth=5,
                    paragraph=False,
                    allowlist=id_allowlist,
                )
                ocr_results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))
                variant_text = "\n".join(r[1] for r in ocr_results)
                raw_ocr_outputs[f"{roi_name}_{v_name}"] = variant_text

                parsed_id, conf, pattern = self.printed_id_parser.parse_printed_id(variant_text)
                if not parsed_id:
                    parsed_id = self.parse_collector_id(variant_text)
                    if parsed_id:
                        conf = 0.85
                        pattern = "legacy_pattern"

                if parsed_id:
                    id_observations.append(parsed_id)
                    score = conf
                    # Path weighting:
                    if "surgical" in roi_name:
                        score += 0.10
                    if pattern in ("standard_fraction", "subset_fraction", "of_fraction"):
                        if "left" in roi_name:
                            score += 0.15
                        elif "right" in roi_name:
                            score -= 0.10
                    elif pattern == "promo_prefix":
                        if parsed_id.startswith("SVP"):
                            if "left" in roi_name:
                                score += 0.15
                        elif "right" in roi_name:
                            score += 0.15
                    parsed_candidates.append((score, parsed_id, roi_name))

        candidate_scores: Counter = Counter()
        candidate_sources: Dict[str, str] = {}
        for score, parsed_id, roi_name in parsed_candidates:
            candidate_scores[parsed_id] += score
            if parsed_id not in candidate_sources:
                candidate_sources[parsed_id] = roi_name

        if candidate_scores:
            collector_id = candidate_scores.most_common(1)[0][0]
            printed_id_roi_source = candidate_sources.get(collector_id, "footer_left")
        elif id_observations:
            collector_id = self._modal_value(id_observations)

        if collector_id:
            self.multi_frame_voter.add_observation(collector_id, confidence=0.90)

        # Fallback: Run OCR on full warped card if fields missing
        if not name or not collector_id or not hp:
            full_variants = self._ocr_variants(warped, field_type="all")
            full_texts = ["\n".join(r[1] for r in results) for results in full_variants]

            if not name:
                full_names = [
                    candidate
                    for candidate in (self._extract_name_by_font_size(results) for results in full_variants)
                    if candidate
                ]
                closed_set_full, _ = self.closed_set_matcher.resolve_candidates(full_names)
                if closed_set_full:
                    name = closed_set_full
                else:
                    name = self._modal_value(full_names)
            if not hp:
                full_hps = [self.parse_hp(text) for text in full_texts]
                hp = self._modal_value([candidate for candidate in full_hps if candidate is not None])
            if not collector_id:
                full_ids = [self.parse_collector_id(text) for text in full_texts]
                collector_id = self._modal_value([candidate for candidate in full_ids if candidate])
                if collector_id:
                    printed_id_roi_source = "warped_full"

        raw_footer = "\n--- RAW OCR PASS ---\n".join(
            f"[{k}]: {v}" for k, v in raw_ocr_outputs.items()
        )

        status = "accepted" if collector_id else "rejected"
        reason = "success" if collector_id else "unreadable_printed_id"

        return {
            "name": name,
            "hp": hp,
            "unique_id": collector_id,
            "normalized_printed_id": collector_id,
            "printed_id_roi_source": printed_id_roi_source,
            "raw_ocr_outputs": raw_ocr_outputs,
            "debug_crop_path": debug_crop_path,
            "debug_files": debug_files,
            "status": status,
            "reason": reason,
            "header_raw_ocr": header_text,
            "footer_raw_ocr": raw_footer,
            "warped_card": warped,
            "header_crop": rois["header"],
            "footer_crop": rois.get(printed_id_roi_source, rois["footer_left"]),
            "ocr_name_candidates": name_candidates,
            "ocr_hp_candidates": hp_candidates,
            "ocr_id_candidates": id_observations,
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

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Raw measurements are intentionally computed before the learned model.
        # The bundled classifier is trained on synthetic degradations, so it is
        # useful corroborating evidence but not reliable enough to be a sole gate.
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        glare_ratio = float((gray > 250).sum() / float(h * w))

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 200)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_area = max((cv2.contourArea(c) for c in contours), default=0.0)
        area_ratio = float(largest_area / float(h * w))
        normalized_aspect = min(w, h) / float(max(w, h))
        full_frame_card_like = 0.62 <= normalized_aspect <= 0.80

        # 1. MobileNetV2 quality evidence (Component 2)
        ml_quality_scores = None
        if classify_frame_quality is not None:
            try:
                ml_quality_scores = classify_frame_quality(frame)
            except Exception:
                ml_quality_scores = None

        metrics = {
            "laplacian_variance": round(lap_var, 2),
            "glare_ratio": round(glare_ratio, 4),
            "card_area_ratio": round(area_ratio, 4),
            "full_frame_card_like": full_frame_card_like,
        }

        # 2. Hard gates require measurable image evidence. Learned predictions
        # only tighten borderline decisions when they agree with the heuristics.
        if lap_var < 70.0:
            return {
                "pass": False,
                "reason": "Image is too blurry. Hold steady.",
                "ml_quality_scores": ml_quality_scores,
                "metrics": metrics,
                "quality_score": 0.0,
            }
        if (
            ml_quality_scores
            and ml_quality_scores.get("blurry", 0.0) > 0.90
            and lap_var < 140.0
        ):
            return {
                "pass": False,
                "reason": "Image is probably blurry. Hold steady.",
                "ml_quality_scores": ml_quality_scores,
                "metrics": metrics,
                "quality_score": 0.0,
            }

        if glare_ratio > 0.08:
            return {
                "pass": False,
                "reason": "Glare detected on card surface. Tilt camera slightly.",
                "ml_quality_scores": ml_quality_scores,
                "metrics": metrics,
                "quality_score": 0.0,
            }
        if (
            ml_quality_scores
            and ml_quality_scores.get("glare", 0.0) > 0.95
            and glare_ratio > 0.03
        ):
            return {
                "pass": False,
                "reason": "Image probably contains glare. Tilt camera slightly.",
                "ml_quality_scores": ml_quality_scores,
                "metrics": metrics,
                "quality_score": 0.0,
            }

        if area_ratio < 0.10 and not full_frame_card_like:
            return {
                "pass": False,
                "reason": "No card detected in frame. Align card inside overlay.",
                "ml_quality_scores": ml_quality_scores,
                "metrics": metrics,
                "quality_score": 0.0,
            }

        sharpness_score = min(lap_var / 350.0, 1.0)
        glare_score = max(0.0, 1.0 - glare_ratio / 0.08)
        presence_score = 1.0 if full_frame_card_like else min(area_ratio / 0.45, 1.0)
        quality_score = 100.0 * (
            sharpness_score * 0.45 + glare_score * 0.30 + presence_score * 0.25
        )

        return {
            "pass": True,
            "reason": None,
            "ml_quality_scores": ml_quality_scores,
            "metrics": metrics,
            "quality_score": round(quality_score, 2),
        }

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
            ocr_res["frame_quality_score"] = quality.get("quality_score", 0.0)
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

        modal_name, fuzzy_name_agreement = self._fuzzy_name_consensus(names)
        modal_hp = self._modal_value(hps)
        modal_id = self._modal_value(ids)

        name_agreement = (
            fuzzy_name_agreement * (len(names) / float(passed_count)) if modal_name else 0.0
        )
        hp_agreement = (hps.count(modal_hp) / float(passed_count)) if modal_hp and hps else 0.0
        id_agreement = (ids.count(modal_id) / float(passed_count)) if modal_id and ids else 0.0

        # Query using the sharpest/cleanest accepted frame, rather than whichever
        # frame happened to arrive first.
        best_frame_result = max(
            passed_ocr_results,
            key=lambda result: result.get("frame_quality_score", 0.0),
        )
        best_warped = best_frame_result.get("warped_card")

        verification = tcg_client.verify_card(
            collector_id=modal_id,
            ocr_name=modal_name,
            ocr_hp=modal_hp,
            card_image=best_warped
        )

        db_confidence = verification.get("confidence", 0.0)
        quality_ratio = passed_count / float(total_frames)

        # Weighted capture confidence
        capture_confidence = (
            quality_ratio * 0.15 +
            id_agreement * 0.30 +
            name_agreement * 0.15 +
            hp_agreement * 0.10 +
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
            "candidates": verification.get("candidates"),
            "message": verification.get("message")
        }
