"""
CLI Test Runner for Pokemon Card High-Accuracy Extraction
Uses a sample Mew ex card image and tests extraction + API verification.
"""

import os
import cv2
import numpy as np
import requests
from pokemon_card_ocr import PokemonCardExtractor
from pokemon_api import PokemonTCGClient

def main():
    print("=" * 60)
    print("⚡ POKEMON CARD HIGH-ACCURACY EXTRACTION TEST")
    print("=" * 60)

    sample_url = "https://images.pokemontcg.io/sv3pt5/151_hires.png"
    img_path = "sample_mew.png"

    if not os.path.exists(img_path):
        print(f"Downloading sample card image from {sample_url}...")
        resp = requests.get(sample_url)
        with open(img_path, "wb") as f:
            f.write(resp.content)
        print("Download complete.")

    print("\n[Step 1] Loading Image & Initializing OCR Engine...")
    image = cv2.imread(img_path)
    extractor = PokemonCardExtractor(gpu=False)

    print("\n[Step 2] Running Crop, Dewarping, and OCR Extraction...")
    res = extractor.extract_from_image(image)

    print("-" * 40)
    print(f"Extracted Card Name  : {res['name']}")
    print(f"Extracted HP         : {res['hp']} HP")
    print(f"Extracted Unique ID  : {res['unique_id']}")
    print(f"Header Raw OCR       : '{res['header_raw_ocr']}'")
    print(f"Footer Raw OCR       : '{res['footer_raw_ocr']}'")
    print("-" * 40)

    print("\n[Step 3] Verifying against Pokémon TCG API...")
    api_client = PokemonTCGClient()
    verification = api_client.verify_card(
        collector_id=res['unique_id'],
        ocr_name=res['name'],
        ocr_hp=res['hp']
    )

    print("-" * 40)
    if verification.get("verified"):
        print("✅ DATABASE MATCH VERIFIED!")
        print(f"Verified Name     : {verification.get('name')}")
        print(f"Verified HP       : {verification.get('hp')} HP")
        print(f"Verified Set      : {verification.get('set_name')} ({verification.get('set_series')})")
        print(f"Verified Unique ID: {verification.get('collector_id')}")
        print(f"Rarity            : {verification.get('rarity')}")
        print(f"Market Price      : ${verification.get('market_price')}")
        print(f"Confidence        : {int(verification['confidence']*100)}%")
    else:
        print(f"⚠️ Verification failed: {verification.get('message')}")
    print("=" * 60)

if __name__ == "__main__":
    main()
