"""
ResNet Visual Card Matcher Module
Identifies Pokémon cards by visual appearance matching using pretrained ResNet50 embeddings
and FAISS approximate nearest-neighbor search.

Key features:
- Pretrained ResNet50 backbone (ImageNet weights) extracts 2048-dimensional normalized feature vectors.
- FAISS vector similarity index (Inner Product / Cosine Similarity).
- Graceful error handling with ModelNotAvailableError if index or model files are missing.
"""

import os
import json
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

class ModelNotAvailableError(Exception):
    """Raised when the visual matching index or embedding model is unavailable."""
    pass

class VisualCardMatcher:
    """
    Visual Card Matcher using ResNet50 feature extractor and FAISS embedding index.
    """
    def __init__(
        self,
        index_path: str = "models/card_index.faiss",
        meta_path: str = "models/card_metadata.json"
    ):
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self.model = None
        self.device = None
        self.transform = None

        self._load_resnet_model()
        self._load_index_and_metadata()

    def _load_resnet_model(self) -> None:
        """Loads pretrained ResNet50 feature extractor (outputting 2048-dim vectors)."""
        try:
            import torch
            import torch.nn as nn
            import torchvision.models as models
            import torchvision.transforms as T

            try:
                from torchvision.models import ResNet50_Weights
                resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            except Exception:
                resnet = models.resnet50(pretrained=True)

            # Remove classification layer; use avgpool output (2048 features)
            self.model = nn.Sequential(*list(resnet.children())[:-1])
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()

            self.transform = T.Compose([
                T.ToTensor(),
                T.Resize((224, 224), antialias=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        except Exception as e:
            print(f"[VisualCardMatcher] Failed to initialize ResNet50 backbone: {e}")
            self.model = None

    def _load_index_and_metadata(self) -> None:
        """Loads FAISS index and metadata JSON from disk."""
        if not os.path.exists(self.meta_path):
            return

        try:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        except Exception as e:
            print(f"[VisualCardMatcher] Error reading metadata '{self.meta_path}': {e}")
            self.metadata = []

        if os.path.exists(self.index_path):
            try:
                import faiss
                self.index = faiss.read_index(self.index_path)
            except Exception as e:
                print(f"[VisualCardMatcher] Failed to read FAISS index '{self.index_path}': {e}")
                self.index = None

    def is_available(self) -> bool:
        """Returns True if both ResNet50 backbone and embedding index are loaded."""
        return self.model is not None and len(self.metadata) > 0

    def extract_embedding(self, image_np: np.ndarray) -> np.ndarray:
        """
        Extracts L2-normalized 2048-dimensional feature embedding from input BGR image.
        """
        if self.model is None:
            raise ModelNotAvailableError("ResNet50 feature extractor model is not initialized.")

        if image_np is None or image_np.size == 0:
            raise ValueError("Invalid or empty image provided for embedding extraction.")

        import torch

        rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.model(tensor)  # Shape: (1, 2048, 1, 1)
            feat = torch.flatten(feat, 1).cpu().numpy()[0]

        # L2 normalization for Cosine Similarity
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        return feat.astype(np.float32)

    def match_card_by_image(
        self,
        warped_card_image: np.ndarray,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Queries FAISS embedding index with input image.

        Returns:
            List of candidate matches sorted by visual similarity score descending:
            [
                {
                    "card_id": "sv3pt5-199",
                    "collector_id": "199/165",
                    "name": "Charizard ex",
                    "hp": 330,
                    "similarity_score": 0.942,
                    "image_url": "https://...",
                    "set_name": "151"
                },
                ...
            ]
        """
        if not self.is_available():
            raise ModelNotAvailableError(
                f"Visual card matching index or metadata is unavailable at '{self.index_path}'."
            )

        query_vector = self.extract_embedding(warped_card_image)

        # 1. FAISS Search
        if self.index is not None:
            import faiss
            query_matrix = np.expand_dims(query_vector, axis=0)
            k_search = min(top_k, self.index.ntotal)
            distances, indices = self.index.search(query_matrix, k_search)

            results = []
            for score, idx in zip(distances[0], indices[0]):
                if 0 <= idx < len(self.metadata):
                    meta = self.metadata[idx].copy()
                    meta["similarity_score"] = round(float(score), 4)
                    results.append(meta)
            return results

        # 2. NumPy Cosine Similarity Search Fallback
        cached_embeddings = [m.get("embedding") for m in self.metadata if "embedding" in m]
        if cached_embeddings:
            matrix = np.array(cached_embeddings, dtype=np.float32)
            scores = np.dot(matrix, query_vector)
            top_indices = np.argsort(scores)[::-1][:top_k]

            results = []
            for idx in top_indices:
                meta = {k: v for k, v in self.metadata[idx].items() if k != "embedding"}
                meta["similarity_score"] = round(float(scores[idx]), 4)
                results.append(meta)
            return results

        return []


# Module singleton instance
_matcher_instance: Optional[VisualCardMatcher] = None

def get_visual_matcher(
    index_path: str = "models/card_index.faiss",
    meta_path: str = "models/card_metadata.json"
) -> VisualCardMatcher:
    global _matcher_instance
    if _matcher_instance is None:
        _matcher_instance = VisualCardMatcher(index_path=index_path, meta_path=meta_path)
    return _matcher_instance

def match_card_by_image(
    warped_card_image: np.ndarray,
    top_k: int = 5,
    index_path: str = "models/card_index.faiss",
    meta_path: str = "models/card_metadata.json"
) -> List[Dict[str, Any]]:
    """
    Public entry-point function for visual card matching.

    Args:
        warped_card_image: BGR numpy image array of cropped/warped card.
        top_k: Number of nearest candidates to return.

    Returns:
        List of dicts with matching metadata and similarity scores.

    Raises:
        ModelNotAvailableError if index or model is missing.
    """
    matcher = get_visual_matcher(index_path=index_path, meta_path=meta_path)
    return matcher.match_card_by_image(warped_card_image, top_k=top_k)
