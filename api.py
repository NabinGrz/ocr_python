"""
FastAPI REST Server for Pokémon TCG OCR & Card Identification
Exposes API endpoints for consumption by Flutter Mobile App.
"""

import io
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pokemon_card_ocr import PokemonCardExtractor
from pokemon_api import PokemonTCGClient

app = FastAPI(
    title="Pokémon TCG Card OCR API",
    description="Backend OCR and verification service for Flutter Mobile App",
    version="1.0.0"
)

# Enable CORS for Flutter web / mobile cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from typing import Optional, Dict, Any, List

# Global singleton instances
ocr_extractor = PokemonCardExtractor()
tcg_client = PokemonTCGClient()


class CardScanResponse(BaseModel):
    success: bool
    verified: bool
    confidence: float
    capture_confidence: float = 0.0
    total_frames: int = 0
    passed_frames: int = 0
    rejection_reason: Optional[str] = None
    name: Optional[str] = None
    hp: Optional[int] = None
    unique_id: Optional[str] = None
    set_name: Optional[str] = None
    set_series: Optional[str] = None
    rarity: Optional[str] = "Unknown"
    image_url: Optional[str] = None
    tcgplayer_url: Optional[str] = None
    market_price: Optional[float] = None
    best_score: float = 0.0
    name_agreement: float = 0.0
    hp_agreement: float = 0.0
    id_agreement: float = 0.0
    message: Optional[str] = None


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Pokémon TCG Card OCR API",
        "docs": "/docs",
        "health": "/health",
        "scan_endpoint": "POST /api/v1/scan",
        "stream_endpoint": "POST /api/v1/scan/stream"
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "pokemon-tcg-ocr"}


@app.post("/api/v1/scan/stream", response_model=CardScanResponse)
async def scan_card_stream(files: List[UploadFile] = File(...)):
    """
    Accepts a burst stream of images (JPEG/PNG), filters quality-rejected frames,
    computes modal consensus across passed frames, and returns real-time confidence scores.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided in burst stream.")

    try:
        frame_bytes_list = []
        for file in files:
            content = await file.read()
            if content:
                frame_bytes_list.append(content)

        result_dict = ocr_extractor.process_frame_burst(frame_bytes_list, tcg_client)

        return CardScanResponse(
            success=result_dict.get("success", False),
            verified=result_dict.get("verified", False),
            confidence=result_dict.get("confidence", 0.0),
            capture_confidence=result_dict.get("capture_confidence", 0.0),
            total_frames=result_dict.get("total_frames", len(files)),
            passed_frames=result_dict.get("passed_frames", 0),
            rejection_reason=result_dict.get("rejection_reason"),
            name=result_dict.get("name"),
            hp=result_dict.get("hp"),
            unique_id=result_dict.get("unique_id"),
            set_name=result_dict.get("set_name"),
            set_series=result_dict.get("set_series"),
            rarity=result_dict.get("rarity", "Unknown"),
            image_url=result_dict.get("image_url"),
            tcgplayer_url=result_dict.get("tcgplayer_url"),
            market_price=result_dict.get("market_price"),
            best_score=result_dict.get("best_score", 0.0),
            name_agreement=result_dict.get("name_agreement", 0.0),
            hp_agreement=result_dict.get("hp_agreement", 0.0),
            id_agreement=result_dict.get("id_agreement", 0.0),
            message=result_dict.get("message")
        )

    except Exception as e:
        return CardScanResponse(
            success=False,
            verified=False,
            confidence=0.0,
            message=f"Stream processing error: {str(e)}"
        )


@app.post("/api/v1/scan", response_model=CardScanResponse)
async def scan_card(file: UploadFile = File(...)):
    """
    Accepts an image upload (JPEG/PNG), extracts card text via OpenCV & EasyOCR,
    queries the Pokémon TCG API, and returns verified card metadata.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded must be an image.")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image_np is None:
            raise HTTPException(status_code=400, detail="Could not decode image file.")

        # 1. OCR Extraction
        ocr_result = ocr_extractor.extract_from_image(image_np)
        extracted_name = ocr_result.get("name")
        extracted_hp = ocr_result.get("hp")
        extracted_id = ocr_result.get("unique_id")

        # 2. Database Verification
        verification = tcg_client.verify_card(
            collector_id=extracted_id,
            ocr_name=extracted_name,
            ocr_hp=extracted_hp
        )

        verified = verification.get("verified", False)

        return CardScanResponse(
            success=True,
            verified=verified,
            confidence=verification.get("confidence", 0.0),
            name=verification.get("name") or extracted_name,
            hp=verification.get("hp") or extracted_hp,
            unique_id=verification.get("collector_id") or extracted_id,
            set_name=verification.get("set_name"),
            set_series=verification.get("set_series"),
            rarity=verification.get("rarity", "Unknown"),
            image_url=verification.get("image_url"),
            tcgplayer_url=verification.get("tcgplayer_url"),
            market_price=verification.get("market_price"),
            best_score=verification.get("best_score", 0.0),
            message=verification.get("message")
        )

    except Exception as e:
        return CardScanResponse(
            success=False,
            verified=False,
            confidence=0.0,
            message=f"Processing error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
