"""
FastAPI REST Server for Pokémon TCG OCR & Card Identification
Exposes API endpoints for consumption by Flutter Mobile App.
"""

import asyncio
import time
import cv2
import logging
import numpy as np
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from pokemon_card_ocr import PokemonCardExtractor
from pokemon_api import PokemonTCGClient

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pokemon_tcg_api")

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


@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    """
    Middleware to log request details, client IP, endpoint, status code, and latency.
    """
    start_time = time.time()
    client_host = request.client.host if request.client else "unknown"
    method = request.method
    url_path = request.url.path

    logger.info("🌐 [HTTP START] %s %s from %s", method, url_path, client_host)

    try:
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000.0
        logger.info("🌐 [HTTP END] %s %s -> Status %d (took %.2f ms)", method, url_path, response.status_code, duration_ms)
        return response
    except Exception as exc:
        duration_ms = (time.time() - start_time) * 1000.0
        logger.error("❌ [HTTP ERROR] %s %s -> Exception: %s (took %.2f ms)", method, url_path, str(exc), duration_ms)
        raise exc


# Global singleton instances
ocr_extractor = PokemonCardExtractor()
tcg_client = PokemonTCGClient()
inference_lock = asyncio.Lock()

MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_BURST_FRAMES = 20


class CardCandidate(BaseModel):
    name: Optional[str] = None
    hp: Optional[int] = None
    unique_id: Optional[str] = None
    set_name: Optional[str] = None
    set_series: Optional[str] = None
    rarity: Optional[str] = "Unknown"
    image_url: Optional[str] = None
    confidence: float = 0.0


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
    candidates: Optional[List[CardCandidate]] = None
    message: Optional[str] = None


@app.get("/")
def root():
    logger.info("Root endpoint hit")
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
    logger.info("Health check endpoint hit")
    return {"status": "ok", "service": "pokemon-tcg-ocr"}


@app.post("/api/v1/scan/stream", response_model=CardScanResponse)
async def scan_card_stream(files: List[UploadFile] = File(...)):
    """
    Accepts a burst stream of images (JPEG/PNG), filters quality-rejected frames,
    computes modal consensus across passed frames, and returns real-time confidence scores.
    """
    start_time = time.time()
    logger.info("📥 [/api/v1/scan/stream] Received frame burst with %d files", len(files) if files else 0)

    if not files:
        logger.warning("⚠️ [/api/v1/scan/stream] No files provided in burst stream")
        raise HTTPException(status_code=400, detail="No files provided in burst stream.")
    if len(files) > MAX_BURST_FRAMES:
        raise HTTPException(
            status_code=413,
            detail=f"At most {MAX_BURST_FRAMES} frames are allowed per burst.",
        )

    try:
        frame_bytes_list = []
        for file in files:
            content = await file.read()
            if len(content) > MAX_IMAGE_BYTES:
                raise HTTPException(status_code=413, detail="An uploaded frame is too large.")
            if content:
                frame_bytes_list.append(content)

        logger.info("🔍 [/api/v1/scan/stream] Processing burst stream (%d non-empty frames)...", len(frame_bytes_list))
        async with inference_lock:
            result_dict = await run_in_threadpool(
                ocr_extractor.process_frame_burst, frame_bytes_list, tcg_client
            )

        raw_candidates = result_dict.get("candidates")
        cand_models = [CardCandidate(**c) for c in raw_candidates] if raw_candidates else None

        passed_count = result_dict.get("passed_frames", 0)
        verified = result_dict.get("verified", False)
        card_name = result_dict.get("name")
        card_id = result_dict.get("unique_id")
        best_score = result_dict.get("best_score", 0.0)

        elapsed_ms = (time.time() - start_time) * 1000.0
        logger.info(
            "✅ [/api/v1/scan/stream] Complete in %.2f ms -> Verified: %s | Name: '%s' | ID: '%s' | Score: %.1f | Passed Frames: %d/%d | Candidates: %d",
            elapsed_ms, verified, card_name, card_id, best_score, passed_count, len(files), len(cand_models) if cand_models else 0
        )

        return CardScanResponse(
            success=result_dict.get("success", False),
            verified=verified,
            confidence=result_dict.get("confidence", 0.0),
            capture_confidence=result_dict.get("capture_confidence", 0.0),
            total_frames=result_dict.get("total_frames", len(files)),
            passed_frames=passed_count,
            rejection_reason=result_dict.get("rejection_reason"),
            name=card_name,
            hp=result_dict.get("hp"),
            unique_id=card_id,
            set_name=result_dict.get("set_name"),
            set_series=result_dict.get("set_series"),
            rarity=result_dict.get("rarity", "Unknown"),
            image_url=result_dict.get("image_url"),
            tcgplayer_url=result_dict.get("tcgplayer_url"),
            market_price=result_dict.get("market_price"),
            best_score=best_score,
            name_agreement=result_dict.get("name_agreement", 0.0),
            hp_agreement=result_dict.get("hp_agreement", 0.0),
            id_agreement=result_dict.get("id_agreement", 0.0),
            candidates=cand_models,
            message=result_dict.get("message")
        )

    except HTTPException:
        raise
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000.0
        logger.error("❌ [/api/v1/scan/stream] Stream processing error after %.2f ms: %s", elapsed_ms, str(e), exc_info=True)
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
    start_time = time.time()
    filename = file.filename or "unknown.jpg"
    logger.info("📥 [/api/v1/scan] Received image upload: '%s' (content_type=%s)", filename, file.content_type)

    if not file.content_type or not file.content_type.startswith("image/"):
        logger.warning("⚠️ [/api/v1/scan] Rejected non-image upload: '%s' (%s)", filename, file.content_type)
        raise HTTPException(status_code=400, detail="File uploaded must be an image.")

    try:
        contents = await file.read()
        if len(contents) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Uploaded image is too large.")
        file_size_kb = len(contents) / 1024.0
        logger.info("   [1/3 Load Image] File size: %.2f KB", file_size_kb)

        nparr = np.frombuffer(contents, np.uint8)
        image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image_np is None:
            logger.error("⚠️ [/api/v1/scan] Could not decode image file: '%s'", filename)
            raise HTTPException(status_code=400, detail="Could not decode image file.")

        img_h, img_w = image_np.shape[:2]
        logger.info("   [1/3 Load Image] Image decoded successfully: %dx%d px", img_w, img_h)

        # 1. OCR Extraction
        logger.info("   [2/3 OCR Pipeline] Running OpenCV crop & EasyOCR text extraction...")
        ocr_start = time.time()
        async with inference_lock:
            ocr_result = await run_in_threadpool(ocr_extractor.extract_from_image, image_np)
            ocr_ms = (time.time() - ocr_start) * 1000.0

        extracted_name = ocr_result.get("name")
        extracted_hp = ocr_result.get("hp")
        extracted_id = ocr_result.get("unique_id")
        logger.info("   [2/3 OCR Pipeline] OCR completed in %.2f ms -> Extracted Name: '%s', HP: %s, ID: '%s'",
                    ocr_ms, extracted_name, extracted_hp, extracted_id)

        # 2. Database Verification
        logger.info("   [3/3 DB Verification] Querying Pokémon card database snapshot & scoring candidates...")
        verify_start = time.time()
        async with inference_lock:
            verification = await run_in_threadpool(
                tcg_client.verify_card,
                extracted_id,
                extracted_name,
                extracted_hp,
                ocr_result.get("warped_card"),
            )
        verify_ms = (time.time() - verify_start) * 1000.0

        verified = verification.get("verified", False)
        final_name = verification.get("name") or extracted_name
        final_hp = verification.get("hp") or extracted_hp
        final_id = verification.get("collector_id") or extracted_id
        best_score = verification.get("best_score", 0.0)
        confidence = verification.get("confidence", 0.0)
        raw_candidates = verification.get("candidates")
        cand_models = [CardCandidate(**c) for c in raw_candidates] if raw_candidates else None

        logger.info(
            "   [3/3 DB Verification] Verification completed in %.2f ms -> Verified: %s | Name: '%s' | HP: %s | ID: '%s' | Score: %.1f | Conf: %.2f | Candidates: %d",
            verify_ms, verified, final_name, final_hp, final_id, best_score, confidence, len(cand_models) if cand_models else 0
        )

        total_ms = (time.time() - start_time) * 1000.0
        logger.info("✅ [/api/v1/scan] Scan request for '%s' finished successfully in %.2f ms", filename, total_ms)

        return CardScanResponse(
            success=True,
            verified=verified,
            confidence=confidence,
            name=final_name,
            hp=final_hp,
            unique_id=final_id,
            set_name=verification.get("set_name"),
            set_series=verification.get("set_series"),
            rarity=verification.get("rarity", "Unknown"),
            image_url=verification.get("image_url"),
            tcgplayer_url=verification.get("tcgplayer_url"),
            market_price=verification.get("market_price"),
            best_score=best_score,
            candidates=cand_models,
            message=verification.get("message")
        )

    except HTTPException:
        raise
    except Exception as e:
        total_ms = (time.time() - start_time) * 1000.0
        logger.error("❌ [/api/v1/scan] Processing error for '%s' after %.2f ms: %s", filename, total_ms, str(e), exc_info=True)
        return CardScanResponse(
            success=False,
            verified=False,
            confidence=0.0,
            message=f"Processing error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
