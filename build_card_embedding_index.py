"""
Index Building Script for ResNet50 Visual Card Matcher
Fetches Pokémon card images from the TCG API (or local sample images),
extracts 2048-d ResNet50 embeddings, and constructs a FAISS index.

Outputs:
- models/card_index.faiss
- models/card_metadata.json
"""

import os
import json
import requests
import cv2
import numpy as np
import faiss
import torch
import torchvision.models as models
import torchvision.transforms as T
from typing import List, Dict, Any, Optional

from pokemon_api import PokemonTCGClient

def get_resnet_extractor():
    try:
        from torchvision.models import ResNet50_Weights
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    except Exception:
        resnet = models.resnet50(pretrained=True)

    model = torch.nn.Sequential(*list(resnet.children())[:-1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    transform = T.Compose([
        T.ToTensor(),
        T.Resize((224, 224), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return model, device, transform

def fetch_card_image(url: str) -> Optional[np.ndarray]:
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if resp.status_code == 200:
            arr = np.frombuffer(resp.content, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img
    except Exception as e:
        print(f"[build_index] Failed to download image from {url}: {e}")
    return None

def build_index(
    max_cards: int = 15,
    index_path: str = "models/card_index.faiss",
    meta_path: str = "models/card_metadata.json"
):
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    tcg_client = PokemonTCGClient()

    print("[build_index] Querying Pokémon TCG API for card data...")
    cards_data = []
    queries = ["Charizard", "Pikachu", "Mew", "Greninja"]

    for q in queries:
        res = tcg_client._safe_get({"q": f'name:"{q}"', "pageSize": 5})
        cards_data.extend(res)
        if len(cards_data) >= max_cards:
            break

    print(f"[build_index] Total cards retrieved: {len(cards_data)}")

    model, device, transform = get_resnet_extractor()

    embeddings = []
    metadata = []

    # Include local sample image if available
    sample_path = "sample_mew.png"
    if os.path.exists(sample_path):
        sample_img = cv2.imread(sample_path)
        if sample_img is not None:
            rgb = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
            tensor = transform(rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = torch.flatten(model(tensor), 1).cpu().numpy()[0]
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat = feat / norm
            embeddings.append(feat)
            metadata.append({
                "card_id": "sv3pt5-151",
                "collector_id": "151/165",
                "name": "Mew ex",
                "hp": 180,
                "set_name": "151",
                "set_series": "Scarlet & Violet",
                "rarity": "Double Rare",
                "image_url": "sample_mew.png",
                "embedding": feat.tolist(),
            })

    for card in cards_data:
        image_url = card.get("images", {}).get("large") or card.get("images", {}).get("small")
        if not image_url:
            continue

        card_img = fetch_card_image(image_url)
        if card_img is None:
            continue

        rgb = cv2.cvtColor(card_img, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = torch.flatten(model(tensor), 1).cpu().numpy()[0]

        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm

        hp_str = card.get("hp", "0")
        card_hp = int(hp_str) if hp_str.isdigit() else None
        printed_total = str(card.get("set", {}).get("printedTotal", ""))
        num_str = str(card.get("number", ""))
        formatted_id = f"{num_str}/{printed_total}" if printed_total else num_str

        meta_item = {
            "card_id": card.get("id"),
            "collector_id": formatted_id,
            "name": card.get("name"),
            "hp": card_hp,
            "set_name": card.get("set", {}).get("name"),
            "set_series": card.get("set", {}).get("series"),
            "rarity": card.get("rarity", "Unknown"),
            "image_url": image_url,
            "embedding": feat.tolist()
        }

        embeddings.append(feat)
        metadata.append(meta_item)
        print(f"Processed: {card.get('name')} ({formatted_id})")

    if not embeddings:
        print("[build_index] No embeddings generated.")
        return

    emb_matrix = np.array(embeddings, dtype=np.float32)
    dim = emb_matrix.shape[1]

    print(f"[build_index] Constructing FAISS IndexFlatIP for {len(embeddings)} vectors of dimension {dim}...")
    index = faiss.IndexFlatIP(dim)
    index.add(emb_matrix)

    faiss.write_index(index, index_path)
    print(f"[build_index] FAISS index saved to '{index_path}'")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[build_index] Metadata saved to '{meta_path}'")

if __name__ == "__main__":
    build_index(max_cards=15)
