"""
Pokémon TCG API Verification Module
Queries Pokemon TCG API to verify extracted Collector ID, Name, and HP
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

    def verify_card(self, collector_id: Optional[str], ocr_name: Optional[str] = None, ocr_hp: Optional[int] = None) -> Dict[str, Any]:
        """
        Queries API using Collector ID or Name to find exact matching Pokémon card.
        """
        candidates: List[Dict[str, Any]] = []
        target_num_raw = None
        target_total_raw = None

        if collector_id and '/' in collector_id:
            target_num_raw, target_total_raw = collector_id.split('/')
            target_num_raw = target_num_raw.strip()
            target_total_raw = target_total_raw.strip()
            # Strip leading zeros for API query (e.g. "004" -> "4")
            num_query = target_num_raw.lstrip('0') or '0'
        elif collector_id:
            target_num_raw = collector_id.strip()
            num_query = target_num_raw.lstrip('0') or '0'
        else:
            num_query = None

        # Strategy 1: Search by Collector Number (e.g., number:4)
        if num_query:
            query = f"number:{num_query}"
            try:
                res = requests.get(self.BASE_URL, params={"q": query}, headers=self.headers, timeout=8)
                if res.status_code == 200:
                    data = res.json().get("data", [])
                    candidates.extend(data)
            except Exception as e:
                print(f"API Request error for number query: {e}")

        # Strategy 2: Search by Name if no candidates found yet
        if not candidates and ocr_name:
            query = f'name:"{ocr_name}"'
            try:
                res = requests.get(self.BASE_URL, params={"q": query}, headers=self.headers, timeout=8)
                if res.status_code == 200:
                    data = res.json().get("data", [])
                    candidates.extend(data)
            except Exception as e:
                print(f"API Request error for name query: {e}")

        if not candidates:
            return {"verified": False, "match": None, "confidence": 0.0, "message": "No cards found in API database."}

        # Rank candidates using Set Total matching + HP matching + Fuzzy Name matching
        best_match = None
        best_score = -1.0

        for card in candidates:
            card_name = card.get("name", "")
            card_hp_str = card.get("hp", "0")
            card_hp = int(card_hp_str) if card_hp_str.isdigit() else None
            set_total = str(card.get("set", {}).get("printedTotal", ""))
            card_num = str(card.get("number", ""))

            # Score components
            total_match_score = 100.0 if (target_total_raw and set_total == target_total_raw) else 0.0
            hp_score = 100.0 if (ocr_hp and card_hp and ocr_hp == card_hp) else (40.0 if ocr_hp is None else 0.0)
            name_score = float(fuzz.partial_ratio(ocr_name.lower(), card_name.lower())) if ocr_name else 50.0

            # Weighted final score
            final_score = (total_match_score * 0.45) + (hp_score * 0.25) + (name_score * 0.30)

            if final_score > best_score:
                best_score = final_score
                best_match = card

        if best_match and best_score >= 35.0:
            card_number_str = str(best_match.get('number', ''))
            printed_total_str = str(best_match.get('set', {}).get('printedTotal', ''))

            # Format formatted collector ID (e.g. preserve 3-digit padding "004/198")
            if target_num_raw and len(target_num_raw) == 3 and len(card_number_str) < 3:
                formatted_num = card_number_str.zfill(3)
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
                "market_price": (best_match.get("tcgplayer", {}).get("prices", {}).get("holofoil", {}) or {}).get("market") or 
                                (best_match.get("tcgplayer", {}).get("prices", {}).get("normal", {}) or {}).get("market") or
                                (best_match.get("tcgplayer", {}).get("prices", {}).get("reverseHolofoil", {}) or {}).get("market")
            }

        return {"verified": False, "match": None, "confidence": 0.0, "message": "No confident database match found."}
