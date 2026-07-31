"""
Pokémon TCG API Verification Module
Queries Pokemon TCG API to verify extracted Collector ID, Name, and HP
Supports SIR (Special Illustration Rare) cards where card number > set total (e.g. 116/086)
"""

import requests
from typing import Optional, Dict, Any, List
from fuzzywuzzy import fuzz

class PokemonTCGClient:
    BASE_URL = "https://api.pokemontcg.io/v2/cards"

    def __init__(self, api_key: Optional[str] = None):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        if api_key:
            self.headers["X-Api-Key"] = api_key

    def _safe_get(self, params: dict, timeout: int = 10) -> List[Dict]:
        """Make a safe API request, returns empty list on failure."""
        try:
            res = requests.get(self.BASE_URL, params=params, headers=self.headers, timeout=timeout)
            if res.status_code == 200 and res.text:
                return res.json().get("data", [])
        except Exception as e:
            print(f"API Request error: {e}")
        return []

    def verify_card(self, collector_id: Optional[str], ocr_name: Optional[str] = None, ocr_hp: Optional[int] = None) -> Dict[str, Any]:
        """
        Multi-strategy Pokémon card lookup:
        1. Exact number search → filter by set total → rank by HP + name
        2. Adjacent number search (±1, ±10) filtered by set total (handles OCR digit misreads like 216→116)
        3. Name search fallback
        """
        candidates: List[Dict[str, Any]] = []
        target_num_raw = None
        target_total_raw = None

        if collector_id and '/' in collector_id:
            target_num_raw, target_total_raw = collector_id.split('/')
            target_num_raw = target_num_raw.strip()
            target_total_raw = target_total_raw.strip()
            num_query = target_num_raw.lstrip('0') or '0'
        elif collector_id:
            target_num_raw = collector_id.strip()
            num_query = target_num_raw.lstrip('0') or '0'
        else:
            num_query = None

        # Strategy 1: Search by exact collector number
        if num_query:
            data = self._safe_get({"q": f"number:{num_query}"})
            candidates.extend(data)

        # Strategy 1b: If we have a set total, try adjacent numbers to recover OCR digit misreads.
        # e.g. OCR reads "216" but real number is "116" — try 116, 126, 106, etc.
        # Strictly filtered by set total to avoid false positives.
        if num_query and target_total_raw:
            num_int = int(num_query)
            # Candidates whose set total already matches — stop early if found
            already_matched = any(
                str(c.get("set", {}).get("printedTotal", "")) == target_total_raw
                for c in candidates
            )
            if not already_matched:
                for delta in (-1, +1, -10, +10, -100, +100):
                    adj_num = num_int + delta
                    if adj_num > 0:
                        adj_data = self._safe_get({"q": f"number:{adj_num}"}, timeout=6)
                        # Only keep cards whose set total matches the extracted total
                        matching = [
                            c for c in adj_data
                            if str(c.get("set", {}).get("printedTotal", "")) == target_total_raw
                        ]
                        if matching:
                            candidates.extend(matching)
                            break  # Stop at first useful delta to avoid over-querying

        # Strategy 2: Name-based fallback if no candidates yet
        if not candidates and ocr_name:
            data = self._safe_get({"q": f'name:"{ocr_name}"'})
            candidates.extend(data)

        if not candidates:
            return {"verified": False, "match": None, "confidence": 0.0, "message": "No cards found in API database."}

        # Rank candidates: set total (40%) + HP (30%) + fuzzy name (30%)
        best_match = None
        best_score = -1.0

        for card in candidates:
            card_name = card.get("name", "")
            card_hp_str = card.get("hp", "0")
            card_hp = int(card_hp_str) if card_hp_str.isdigit() else None
            set_total = str(card.get("set", {}).get("printedTotal", ""))

            total_score = 100.0 if (target_total_raw and set_total == target_total_raw) else 0.0
            hp_score    = 100.0 if (ocr_hp and card_hp and ocr_hp == card_hp) else (40.0 if ocr_hp is None else 0.0)
            name_score  = float(fuzz.partial_ratio(ocr_name.lower(), card_name.lower())) if ocr_name else 50.0

            final_score = (total_score * 0.40) + (hp_score * 0.30) + (name_score * 0.30)

            if final_score > best_score:
                best_score = final_score
                best_match = card

        # Require minimum confidence before accepting
        if best_match and best_score >= 40.0:
            card_number_str = str(best_match.get('number', ''))
            printed_total_str = str(best_match.get('set', {}).get('printedTotal', ''))

            # Preserve zero-padding (e.g. "004" not "4") based on extracted ID length
            if target_num_raw and len(target_num_raw) >= 3 and len(card_number_str) < 3:
                formatted_num = card_number_str.zfill(len(target_num_raw))
            else:
                formatted_num = card_number_str

            formatted_id = f"{formatted_num}/{printed_total_str}" if printed_total_str else formatted_num

            return {
                "verified": True,
                "confidence": round(min(best_score / 100.0, 1.0), 2),
                "name": best_match.get("name"),
                "hp": best_match.get("hp"),
                "collector_id": formatted_id,
                "set_name": best_match.get("set", {}).get("name"),
                "set_series": best_match.get("set", {}).get("series"),
                "rarity": best_match.get("rarity", "Unknown"),
                "image_url": best_match.get("images", {}).get("large"),
                "tcgplayer_url": best_match.get("tcgplayer", {}).get("url"),
                "market_price": (
                    (best_match.get("tcgplayer", {}).get("prices", {}).get("holofoil", {}) or {}).get("market") or
                    (best_match.get("tcgplayer", {}).get("prices", {}).get("normal", {}) or {}).get("market") or
                    (best_match.get("tcgplayer", {}).get("prices", {}).get("reverseHolofoil", {}) or {}).get("market")
                )
            }

        return {"verified": False, "match": None, "confidence": 0.0, "message": "No confident database match found."}
