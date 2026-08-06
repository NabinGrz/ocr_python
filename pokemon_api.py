"""
Pokémon TCG API Verification Module
Queries Pokemon TCG API to verify extracted Collector ID, Name, and HP
Supports SIR (Special Illustration Rare) cards where card number > set total (e.g. 116/086)
"""

import requests
import time
import re
import json
import os
import numpy as np
from typing import Optional, Dict, Any, List
from fuzzywuzzy import fuzz

try:
    from visual_card_matcher import match_card_by_image, ModelNotAvailableError as VisualMatcherNotAvailableError
except ImportError:
    match_card_by_image = None


class PokemonTCGClient:
    BASE_URL = "https://api.pokemontcg.io/v2/cards"

    def __init__(
        self,
        api_key: Optional[str] = None,
        catalog_path: str = "models/card_catalog.json",
    ):
        if api_key is None:
            api_key = os.getenv("POKEMON_TCG_API_KEY")

        self.session = requests.Session()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        if api_key:
            self.headers["X-Api-Key"] = api_key
        self.session.headers.update(self.headers)
        self._cache: Dict[str, List[Dict]] = {}
        self.catalog_path = catalog_path
        self._local_cards: List[Dict] = []
        self._cards_by_number: Dict[str, List[Dict]] = {}
        self._cards_by_name: Dict[str, List[Dict]] = {}
        self._load_local_catalog()

    @staticmethod
    def _number_keys(value: str) -> List[str]:
        raw = str(value or "").upper().strip()
        if not raw:
            return []
        keys = {raw, raw.lstrip("0") or "0"}
        digits = re.sub(r"\D", "", raw)
        if digits:
            keys.update({digits, digits.lstrip("0") or "0"})
        return list(keys)

    def _load_local_catalog(self) -> None:
        if not os.path.exists(self.catalog_path):
            return
        try:
            with open(self.catalog_path, "r", encoding="utf-8") as source:
                payload = json.load(source)
            self._local_cards = payload.get("cards", payload) if isinstance(payload, dict) else payload
            for card in self._local_cards:
                for key in self._number_keys(card.get("number", "")):
                    self._cards_by_number.setdefault(key, []).append(card)
                root_name = self._extract_root_name(str(card.get("name", ""))).lower()
                if root_name:
                    self._cards_by_name.setdefault(root_name, []).append(card)
        except Exception as exc:
            print(f"Local card catalog could not be loaded: {exc}")
            self._local_cards = []
            self._cards_by_number = {}
            self._cards_by_name = {}

    def _search_local_catalog(
        self,
        ocr_name: Optional[str],
        num_query: Optional[str],
    ) -> List[Dict]:
        """Return a compact union of number and fuzzy-name candidates."""
        if not self._local_cards:
            return []

        matches: Dict[str, Dict] = {}
        if num_query:
            for key in self._number_keys(num_query):
                for card in self._cards_by_number.get(key, []):
                    matches[str(card.get("id"))] = card

        if ocr_name:
            root_name = self._extract_root_name(ocr_name).lower()
            exact_name_cards = self._cards_by_name.get(root_name, [])
            if exact_name_cards:
                for card in exact_name_cards:
                    matches[str(card.get("id"))] = card
            elif root_name:
                ranked_names = sorted(
                    (
                        (fuzz.ratio(root_name, known_name), known_name)
                        for known_name in self._cards_by_name
                    ),
                    reverse=True,
                )[:8]
                for score, known_name in ranked_names:
                    if score < 65:
                        continue
                    for card in self._cards_by_name[known_name]:
                        matches[str(card.get("id"))] = card

        return list(matches.values())

    def _safe_get(self, params: dict, timeout: float = 5.0) -> List[Dict]:
        """Cached API request with pooling; transient failures are never cached."""
        cache_key = str(sorted(params.items()))
        if cache_key in self._cache:
            return self._cache[cache_key]

        for attempt in range(3):
            try:
                res = self.session.get(
                    self.BASE_URL, params=params, timeout=timeout
                )
                if res.status_code == 200 and res.text.strip():
                    try:
                        result = res.json().get("data", [])
                        self._cache[cache_key] = result
                        return result
                    except Exception:
                        pass
                if res.status_code in (429, 500, 502, 503):
                    time.sleep(0.4 * (attempt + 1))
            except requests.exceptions.Timeout as e:
                print(f"API timeout ({params}): {e}")
                if attempt < 2:
                    time.sleep(0.4 * (attempt + 1))
            except Exception as e:
                if attempt < 2:
                    time.sleep(0.4 * (attempt + 1))
                else:
                    print(f"API error ({params}): {e}")
        return []

    def _extract_root_name(self, name: str) -> str:
        """Extract core Pokémon name (e.g. 'Mega GreninjaN' -> 'Greninja', 'Mewe' -> 'Mew', 'Backtrack Badge' -> 'Backtrack Badge')."""
        words = [w for w in name.split() if w.lower() not in ("mega", "ex", "v", "vmax", "star", "gx", "stage1", "stage2", "basic")]
        target = " ".join(words) if words else name
        cleaned = re.sub(r'[^a-zA-Z\s]', '', target).strip()
        return cleaned

    @staticmethod
    def _format_candidate(card: Dict[str, Any], score: float) -> Dict[str, Any]:
        """Create a stable, API-facing summary for ranked alternatives."""
        card_number = str(card.get("number", ""))
        printed_total = str(card.get("set", {}).get("printedTotal", ""))
        collector_id = f"{card_number}/{printed_total}" if printed_total else card_number
        return {
            "name": card.get("name"),
            "hp": int(card["hp"]) if str(card.get("hp", "")).isdigit() else None,
            "unique_id": collector_id,
            "set_name": card.get("set", {}).get("name"),
            "set_series": card.get("set", {}).get("series"),
            "rarity": card.get("rarity", "Unknown"),
            "image_url": card.get("images", {}).get("large"),
            "confidence": round(max(0.0, min(score / 100.0, 1.0)), 2),
        }

    def verify_card(
        self,
        collector_id: Optional[str],
        ocr_name: Optional[str] = None,
        ocr_hp: Optional[int] = None,
        card_image: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        candidates: List[Dict] = []
        target_num_raw = None
        target_total_raw = None

        # Strategy 0 (Component 3 Layer): ResNet Visual Matching in parallel with OCR
        visual_matches = []
        top_visual_match = None
        if card_image is not None and match_card_by_image is not None:
            try:
                visual_matches = match_card_by_image(card_image, top_k=5)
                if visual_matches:
                    top_visual_match = visual_matches[0]
            except Exception:
                visual_matches = []
                top_visual_match = None

        if collector_id and "/" in collector_id:
            target_num_raw, target_total_raw = collector_id.split("/", 1)
            target_num_raw = target_num_raw.strip()
            target_total_raw = target_total_raw.strip()
            num_query = target_num_raw.lstrip("0") or "0"
        elif collector_id:
            target_num_raw = collector_id.strip()
            num_query = target_num_raw.lstrip("0") or "0"
        else:
            num_query = None

        candidates.extend(self._search_local_catalog(ocr_name, num_query))

        # Strategy 1: Search by root Pokémon name
        if ocr_name and not candidates:
            root_name = self._extract_root_name(ocr_name)
            if root_name:
                data = self._safe_get({"q": f'name:"{root_name}"'})
                if not data:
                    data = self._safe_get({"q": f"name:{root_name}*"})
                if not data and len(root_name) >= 3:
                    # Progressive prefix fallback handles trailing OCR noise such
                    # as "Mewe" -> "Mew*" without immediately issuing a broad query.
                    for prefix_length in (4, 3):
                        if len(root_name) < prefix_length:
                            continue
                        data = self._safe_get({"q": f"name:{root_name[:prefix_length]}*"})
                        if data:
                            break
                candidates.extend(data)

        # Strategy 2: Search by Collector Number
        if num_query and not candidates:
            data = self._safe_get({"q": f"number:{num_query}"})
            numeric_query = re.sub(r"\D", "", num_query).lstrip("0") or "0"
            if not data and numeric_query != num_query:
                data = self._safe_get({"q": f"number:{numeric_query}"})
            existing_ids = {c.get("id") for c in candidates}
            for c in data:
                if c.get("id") not in existing_ids:
                    candidates.append(c)

        if not candidates:
            return {
                "verified": False,
                "match": None,
                "confidence": 0.0,
                "best_score": 0.0,
                "message": "No cards found in API database.",
            }

        # Candidate Ranking System
        best_match = None
        best_score = -1.0
        best_subscores_count = 0
        best_strong_match = False
        best_has_set_identity = False
        ranked_candidates = []

        for card in candidates:
            card_name = card.get("name", "")
            card_hp_str = card.get("hp", "0")
            card_hp = int(card_hp_str) if card_hp_str.isdigit() else None
            set_total = str(card.get("set", {}).get("printedTotal", ""))
            card_number = str(card.get("number", ""))

            target_digits = re.sub(r'\D', '', target_total_raw) if target_total_raw else ""
            set_digits = re.sub(r'\D', '', set_total) if set_total else ""

            # Check if this card belongs to a special subset / promo (TG, GG, SV, RC, SWSH, SVP)
            is_special_subset = any(
                card_number.upper().startswith(p) or (num_query and num_query.upper().startswith(p))
                for p in ('TG', 'GG', 'SV', 'RC', 'SWSH', 'SVP', 'W', 'PROMO')
            )

            # Total score calculation
            total_score = 0.0
            if target_total_raw and set_total:
                if target_total_raw.upper() == set_total.upper() or (target_digits and target_digits == set_digits):
                    total_score = 100.0
                elif is_special_subset:
                    # Special subsets (e.g. TG01/TG30 vs printedTotal 186)
                    total_score = 90.0
                elif target_digits.isdigit() and set_digits.isdigit():
                    try:
                        t_val = int(target_digits)
                        s_val = int(set_digits)
                        if abs(s_val - t_val) <= 2:
                            total_score = 60.0
                        elif len(target_digits) == len(set_digits):
                            diffs = sum(1 for a, b in zip(target_digits, set_digits) if a != b)
                            if diffs == 1:
                                total_score = 50.0
                    except ValueError:
                        pass
            elif is_special_subset:
                # Promo cards or subset cards with missing denominator
                total_score = 80.0

            hp_score = 100.0 if (ocr_hp and card_hp and ocr_hp == card_hp) else 0.0
            name_score = (
                float(max(
                    fuzz.ratio(ocr_name.lower(), card_name.lower()),
                    fuzz.token_set_ratio(ocr_name.lower(), card_name.lower()),
                ))
                if ocr_name else 0.0
            )

            # Number match score (exact match = 100, suffix/subset match = 80)
            num_score = 0.0
            if num_query:
                q_clean = num_query.upper().strip()
                c_clean = card_number.upper().strip()
                if c_clean == q_clean:
                    num_score = 100.0
                elif c_clean.lstrip("0") == q_clean.lstrip("0") and c_clean.lstrip("0"):
                    num_score = 100.0
                elif c_clean.endswith(q_clean) or q_clean.endswith(c_clean):
                    num_score = 80.0

            # If exact number match on promo/special card, ensure total_score is high
            if num_score == 100.0 and (is_special_subset or not target_total_raw):
                total_score = 100.0

            # Weighted final score
            final_score = (
                total_score * 0.30
                + num_score * 0.30
                + name_score * 0.25
                + hp_score * 0.15
            )

            # Count only fields actually observed by OCR. Defaults for absent
            # fields must never manufacture verification evidence.
            collector_evidence = bool(
                num_query
                and num_score > 0
                and (not target_total_raw or total_score > 0)
            )
            subscores_count = sum((
                collector_evidence,
                bool(ocr_name and name_score >= 60),
                bool(ocr_hp and hp_score > 0),
            ))
            strong_collector_match = bool(num_score == 100.0 and total_score >= 90.0)
            strong_match = bool(
                strong_collector_match
                and (name_score >= 60.0 or hp_score == 100.0)
            )
            has_set_identity = bool(
                target_total_raw
                or (is_special_subset and num_score == 100.0)
            )
            ranked_candidates.append((final_score, card))

            if final_score > best_score:
                best_score = final_score
                best_match = card
                best_subscores_count = subscores_count
                best_strong_match = strong_match
                best_has_set_identity = has_set_identity

        ranked_candidates.sort(key=lambda item: item[0], reverse=True)
        candidate_summaries = [
            self._format_candidate(card, score) for score, card in ranked_candidates[:5]
        ]

        # Check for disagreement between Visual Match and OCR text match
        disagreement_warning = None
        if top_visual_match and top_visual_match.get("similarity_score", 0.0) >= 0.80 and best_match:
            vis_name = top_visual_match.get("name", "")
            vis_hp = top_visual_match.get("hp")
            best_name = best_match.get("name", "")
            best_hp_str = best_match.get("hp", "0")
            best_hp = int(best_hp_str) if best_hp_str.isdigit() else None

            # If visual match disagrees with OCR match in name or HP
            if vis_hp and ocr_hp and abs(vis_hp - ocr_hp) > 30:
                disagreement_warning = f"Visual candidate '{vis_name}' (HP {vis_hp}) differs from OCR candidate '{best_name}' (HP {ocr_hp})."
            elif vis_name and ocr_name and fuzz.partial_ratio(vis_name.lower(), ocr_name.lower()) < 40:
                disagreement_warning = f"Visual candidate '{vis_name}' differs from OCR text '{ocr_name}'."

        # If top visual match has high similarity (>0.85) AND agrees or cross-checks with OCR
        visual_name_agreement = bool(
            top_visual_match
            and ocr_name
            and fuzz.ratio(
                str(top_visual_match.get("name", "")).lower(), ocr_name.lower()
            ) >= 75
        )
        visual_hp_agreement = bool(
            top_visual_match
            and ocr_hp
            and top_visual_match.get("hp") == ocr_hp
        )
        visual_id_agreement = bool(
            top_visual_match
            and collector_id
            and top_visual_match.get("collector_id") == collector_id
        )

        if (
            top_visual_match
            and top_visual_match.get("similarity_score", 0.0) >= 0.88
            and not disagreement_warning
            and (visual_name_agreement or visual_hp_agreement or visual_id_agreement)
        ):
            vis_score = top_visual_match["similarity_score"]
            return {
                "verified": True,
                "confidence": round(float(vis_score), 2),
                "best_score": round(vis_score * 100.0, 1),
                "name": top_visual_match.get("name"),
                "hp": top_visual_match.get("hp"),
                "collector_id": top_visual_match.get("collector_id"),
                "set_name": top_visual_match.get("set_name"),
                "set_series": top_visual_match.get("set_series"),
                "rarity": top_visual_match.get("rarity", "Unknown"),
                "image_url": top_visual_match.get("image_url"),
                "tcgplayer_url": top_visual_match.get("tcgplayer_url"),
                "market_price": top_visual_match.get("market_price"),
                "visual_match_applied": True,
                "visual_similarity": round(float(vis_score), 4),
                "disagreement_warning": None,
                "candidates": candidate_summaries,
            }

        # Accept either a strong full collector-ID/name+HP match, or a high
        # aggregate score supported by at least two independently observed fields.
        if best_match and (
            best_strong_match
            or (
                best_has_set_identity
                and best_score >= 58.0
                and best_subscores_count >= 2
            )
        ):
            card_number_str   = str(best_match.get("number", ""))
            printed_total_str = str(best_match.get("set", {}).get("printedTotal", ""))

            if target_num_raw and len(target_num_raw) >= 3 and len(card_number_str) < 3:
                formatted_num = card_number_str.zfill(len(target_num_raw))
            else:
                formatted_num = card_number_str

            formatted_id = f"{formatted_num}/{printed_total_str}" if printed_total_str else formatted_num

            prices = best_match.get("tcgplayer", {}).get("prices", {})
            market_price = (
                (prices.get("holofoil") or {}).get("market")
                or (prices.get("normal") or {}).get("market")
                or (prices.get("reverseHolofoil") or {}).get("market")
                or (prices.get("1stEditionHolofoil") or {}).get("market")
            )

            res = {
                "verified": True,
                "confidence": round(min(best_score / 100.0, 1.0), 2),
                "best_score": round(best_score, 1),
                "name": best_match.get("name"),
                "hp": best_match.get("hp"),
                "collector_id": formatted_id,
                "set_name": best_match.get("set", {}).get("name"),
                "set_series": best_match.get("set", {}).get("series"),
                "rarity": best_match.get("rarity", "Unknown"),
                "image_url": best_match.get("images", {}).get("large"),
                "tcgplayer_url": best_match.get("tcgplayer", {}).get("url"),
                "market_price": market_price,
                "visual_match_applied": False,
                "disagreement_warning": disagreement_warning,
                "candidates": candidate_summaries,
            }
            if top_visual_match:
                res["visual_candidate"] = {
                    "name": top_visual_match.get("name"),
                    "hp": top_visual_match.get("hp"),
                    "similarity": round(float(top_visual_match.get("similarity_score", 0.0)), 4)
                }
            return res

        return {
            "verified": False,
            "match": None,
            "confidence": 0.0,
            "best_score": round(best_score, 1) if best_score > 0 else 0.0,
            "message": "No confident database match found.",
            "candidates": candidate_summaries,
            "disagreement_warning": disagreement_warning,
            "visual_candidate": {
                "name": top_visual_match.get("name"),
                "hp": top_visual_match.get("hp"),
                "similarity": round(float(top_visual_match.get("similarity_score", 0.0)), 4)
            } if top_visual_match else None
        }
