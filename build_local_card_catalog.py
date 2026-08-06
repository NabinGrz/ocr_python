"""Build a compact offline catalog from the official Pokémon TCG data repository."""

import argparse
import io
import json
import os
import zipfile
from datetime import datetime, timezone

import requests


DEFAULT_SOURCE = (
    "https://codeload.github.com/PokemonTCG/pokemon-tcg-data/zip/refs/heads/master"
)
DEFAULT_OUTPUT = "models/card_catalog.json"


def compact_card(card, set_info=None):
    """Retain identification and display fields used by the scanner."""
    compact = {
        key: card.get(key)
        for key in (
            "id",
            "name",
            "hp",
            "number",
            "rarity",
            "set",
            "images",
            "tcgplayer",
        )
        if card.get(key) is not None
    }
    if "set" not in compact and set_info:
        compact["set"] = {
            key: set_info.get(key)
            for key in ("id", "name", "series", "printedTotal", "total")
            if set_info.get(key) is not None
        }
    return compact


def build_catalog(source_url=DEFAULT_SOURCE, output_path=DEFAULT_OUTPUT):
    response = requests.get(source_url, timeout=(10, 120))
    response.raise_for_status()

    cards = []
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        sets_filename = next(
            name for name in archive.namelist() if name.endswith("/sets/en.json")
        )
        with archive.open(sets_filename) as source:
            sets_by_id = {item["id"]: item for item in json.load(source)}

        english_card_files = sorted(
            name
            for name in archive.namelist()
            if "/cards/en/" in name and name.endswith(".json")
        )
        for filename in english_card_files:
            set_id = os.path.splitext(os.path.basename(filename))[0]
            with archive.open(filename) as source:
                cards.extend(
                    compact_card(card, sets_by_id.get(set_id)) for card in json.load(source)
                )

    cards.sort(key=lambda card: str(card.get("id", "")))
    payload = {
        "source": "PokemonTCG/pokemon-tcg-data",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "card_count": len(cards),
        "cards": cards,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary_path = f"{output_path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as destination:
        json.dump(payload, destination, separators=(",", ":"), ensure_ascii=False)
    os.replace(temporary_path, output_path)
    return len(cards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    count = build_catalog(arguments.source, arguments.output)
    print(f"Wrote {count:,} English cards to {arguments.output}")
