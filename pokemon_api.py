"""
Pokémon TCG API Verification Module
Queries Pokemon TCG API to verify extracted Collector ID, Name, and HP
Supports SIR (Special Illustration Rare) cards where card number > set total (e.g. 116/086)
"""

import requests
import time
import re
from typing import Optional, Dict, Any, List
from fuzzywuzzy import fuzz


class PokemonTCGClient:
    BASE_URL = "https://api.pokemontcg.io/v2/cards"

    def __init__(self, api_key: Optional[str] = None):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        if api_key:
            self.headers["X-Api-Key"] = api_key
        self._cache: Dict[str, List[Dict]] = {}

    def _safe_get(self, params: dict, timeout: int = 10) -> List[Dict]:
        """Cached API request with exponential backoff on rate-limit errors."""
        cache_key = str(sorted(params.items()))
        if cache_key in self._cache:
            return self._cache[cache_key]

        for attempt in range(3):
            try:
                res = requests.get(
                    self.BASE_URL, params=params,
                    headers=self.headers, timeout=timeout
                )
                if res.status_code == 200 and res.text.strip():
                    result = res.json().get("data", [])
                    self._cache[cache_key] = result
                    return result
                if res.status_code in (429, 500, 502, 503):
                    time.sleep(1.5 * (attempt + 1))
            except Exception as e:
                if attempt < 2:
                    time.sleep(1.0 * (attempt + 1))
                else:
                    print(f"API error ({params}): {e}")
        self._cache[cache_key] = []
        return []

    def _extract_root_name(self, name: str) -> str:
        """Extract core Pokémon name (e.g. 'Mega GreninjaN' -> 'Greninja', 'Mewe' -> 'Mew', 'Backtrack Badge' -> 'Backtrack Badge')."""
        words = [w for w in name.split() if w.lower() not in ("mega", "ex", "v", "vmax", "star", "gx", "stage1", "stage2", "basic")]
        target = " ".join(words) if words else name
        cleaned = re.sub(r'[^a-zA-Z\s]', '', target).strip()
        return cleaned

    def verify_card(
        self,
        collector_id: Optional[str],
        ocr_name: Optional[str] = None,
        ocr_hp: Optional[int] = None,
    ) -> Dict[str, Any]:
        candidates: List[Dict] = []
        target_num_raw = None
        target_total_raw = None

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

        # Strategy 1: Search by root Pokémon name
        if ocr_name:
            root_name = self._extract_root_name(ocr_name)
            if root_name:
                data = self._safe_get({"q": f'name:"{root_name}"'})
                if not data:
                    data = self._safe_get({"q": f"name:{root_name}*"})
                candidates.extend(data)

        # Strategy 2: Search by Collector Number
        if num_query:
            data = self._safe_get({"q": f"number:{num_query}"})
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

        for card in candidates:
            card_name = card.get("name", "")
            card_hp_str = card.get("hp", "0")
            card_hp = int(card_hp_str) if card_hp_str.isdigit() else None
            set_total = str(card.get("set", {}).get("printedTotal", ""))
            card_number = str(card.get("number", ""))

            # Total score calculation (compare stripped leading zeros so '84' matches '084')
            total_score = 0.0
            if target_total_raw and set_total:
                target_stripped = target_total_raw.lstrip("0") or "0"
                set_stripped = set_total.lstrip("0") or "0"
                if set_stripped == target_stripped or set_total == target_total_raw:
                    total_score = 100.0
                elif abs(int(set_stripped) - int(target_stripped)) <= 2:
                    total_score = 60.0
                elif len(set_stripped) == len(target_stripped):
                    diffs = sum(1 for a, b in zip(set_stripped, target_stripped) if a != b)
                    if diffs == 1:
                        total_score = 50.0  # Single-digit misread partial credit

            hp_score    = 100.0 if (ocr_hp and card_hp and ocr_hp == card_hp) else (40.0 if not ocr_hp else 0.0)
            name_score  = float(fuzz.partial_ratio(ocr_name.lower(), card_name.lower())) if ocr_name else 50.0

            # Number match score (exact match = 100, suffix/delta match = 70, misread = 0)
            num_score = 0.0
            if num_query:
                if card_number == num_query:
                    num_score = 100.0
                elif num_query.endswith(card_number) or card_number.endswith(num_query[-2:]):
                    num_score = 70.0

            # Weighted final score
            final_score = (total_score * 0.40) + (hp_score * 0.20) + (name_score * 0.20) + (num_score * 0.20)

            # Count non-zero sub-scores for verification threshold gate
            subscores_count = sum(1 for score in (total_score, hp_score, name_score, num_score) if score > 0)

            if final_score > best_score:
                best_score = final_score
                best_match = card
                best_subscores_count = subscores_count

        # Bug 5 Fix: Require best_score >= 45.0 AND at least 2 non-zero sub-scores to prevent weak/false matches
        if best_match and best_score >= 45.0 and best_subscores_count >= 2:
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

            return {
                "verified": True,
                "confidence": round(min(best_score / 100.0, 1.0), 2),
                "best_score": round(best_score, 1),  # Bug 5 Fix: Return raw best_score for visibility
                "name": best_match.get("name"),
                "hp": best_match.get("hp"),
                "collector_id": formatted_id,
                "set_name": best_match.get("set", {}).get("name"),
                "set_series": best_match.get("set", {}).get("series"),
                "rarity": best_match.get("rarity", "Unknown"),
                "image_url": best_match.get("images", {}).get("large"),
                "tcgplayer_url": best_match.get("tcgplayer", {}).get("url"),
                "market_price": market_price,
            }

        return {
            "verified": False,
            "match": None,
            "confidence": 0.0,
            "best_score": round(best_score, 1) if best_score > 0 else 0.0,  # Bug 5 Fix: Expose best_score on failure
            "message": "No confident database match found.",
        }
