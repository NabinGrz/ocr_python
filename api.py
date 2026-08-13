"""
FastAPI REST Server for Pokémon TCG OCR & Card Identification
Exposes API endpoints for consumption by Flutter Mobile App.
"""

import asyncio
import os
import time
import cv2
import logging
import numpy as np
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, Response
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

# Ensure debug directory exists
os.makedirs("debug_crops/latest", exist_ok=True)

app = FastAPI(
    title="Pokémon TCG Card OCR API",
    description="Backend OCR and verification service for Flutter Mobile App",
    version="1.0.0"
)

# Mount debug crops static directory to serve images to client/browser
app.mount("/debug_crops", StaticFiles(directory="debug_crops"), name="debug_crops")

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


# Environment optimization for low-memory environments
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# Global singleton lazy holders
_ocr_extractor: Optional[PokemonCardExtractor] = None
_tcg_client: Optional[PokemonTCGClient] = None
inference_lock = asyncio.Lock()

def get_ocr_extractor() -> PokemonCardExtractor:
    """Lazy initializer for PokemonCardExtractor to keep boot RAM minimal."""
    global _ocr_extractor
    if _ocr_extractor is None:
        logger.info("⚡ [Lazy Load] Initializing PokemonCardExtractor on first request...")
        _ocr_extractor = PokemonCardExtractor(gpu=False)
        logger.info("⚡ [Lazy Load] PokemonCardExtractor successfully loaded.")
    return _ocr_extractor

def get_tcg_client() -> PokemonTCGClient:
    """Lazy initializer for PokemonTCGClient to keep boot RAM minimal."""
    global _tcg_client
    if _tcg_client is None:
        logger.info("⚡ [Lazy Load] Initializing PokemonTCGClient on first request...")
        _tcg_client = PokemonTCGClient()
        logger.info("⚡ [Lazy Load] PokemonTCGClient successfully loaded.")
    return _tcg_client

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


@app.api_route("/", methods=["GET", "HEAD"])
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


@app.api_route("/health", methods=["GET", "HEAD"])
def health_check():
    logger.info("Health check endpoint hit")
    return {"status": "ok", "service": "pokemon-tcg-ocr"}


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.api_route("/scan", methods=["GET", "HEAD"], include_in_schema=False)
@app.api_route("/api/v1/scan", methods=["GET", "HEAD"], include_in_schema=False)
def scan_info():
    return {
        "status": "ok",
        "message": "Use POST with multipart/form-data ('file') to scan Pokémon cards.",
        "endpoints": {
            "single_scan": "POST /api/v1/scan",
            "burst_stream": "POST /api/v1/scan/stream"
        }
    }


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
        extractor = get_ocr_extractor()
        client = get_tcg_client()
        async with inference_lock:
            result_dict = await run_in_threadpool(
                extractor.process_frame_burst, frame_bytes_list, client
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
@app.post("/scan", response_model=CardScanResponse, include_in_schema=False)
async def scan_card(
    file: UploadFile = File(...),
    save_debug: bool = Query(True, description="Whether to persist all transformed pipeline images to debug_crops/latest"),
):
    """
    Accepts an image upload (JPEG/PNG), extracts card text via OpenCV & EasyOCR,
    queries the Pokémon TCG API, and returns verified card metadata.
    """
    start_time = time.time()
    filename = file.filename or "unknown.jpg"
    logger.info("📥 [/api/v1/scan] Received image upload: '%s' (content_type=%s, save_debug=%s)", filename, file.content_type, save_debug)

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
        extractor = get_ocr_extractor()
        client = get_tcg_client()
        async with inference_lock:
            ocr_result = await run_in_threadpool(
                extractor.extract_from_image, image_np, save_debug=save_debug
            )
            ocr_ms = (time.time() - ocr_start) * 1000.0

        extracted_name = ocr_result.get("name")
        extracted_hp = ocr_result.get("hp")
        extracted_id = ocr_result.get("unique_id")
        saved_debug_count = len(ocr_result.get("debug_files", {}))
        if save_debug and saved_debug_count > 0:
            logger.info("   [2/3 OCR Pipeline] Saved %d debug images to /debug_crops/latest", saved_debug_count)

        logger.info("   [2/3 OCR Pipeline] OCR completed in %.2f ms -> Extracted Name: '%s', HP: %s, ID: '%s'",
                    ocr_ms, extracted_name, extracted_hp, extracted_id)

        # 2. Database Verification
        logger.info("   [3/3 DB Verification] Querying Pokémon card database snapshot & scoring candidates...")
        verify_start = time.time()
        async with inference_lock:
            verification = await run_in_threadpool(
                client.verify_card,
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


@app.get("/api/v1/debug/pipeline")
def get_debug_pipeline_images():
    """
    Returns a list of all images currently saved in debug_crops/latest.
    """
    latest_dir = "debug_crops/latest"
    if not os.path.exists(latest_dir):
        return {"images": [], "count": 0}

    images = []
    for fname in sorted(os.listdir(latest_dir)):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            fpath = os.path.join(latest_dir, fname)
            stat = os.stat(fpath)
            images.append({
                "name": fname,
                "url": f"/debug_crops/latest/{fname}",
                "size_bytes": stat.st_size,
                "modified_time": stat.st_mtime
            })

    return {"images": images, "count": len(images)}


@app.get("/debug", response_class=HTMLResponse)
def debug_gallery_view():
    """
    Renders an HTML gallery to inspect the received raw upload and all manipulated OCR variants.
    """
    latest_dir = "debug_crops/latest"
    files = sorted(os.listdir(latest_dir)) if os.path.exists(latest_dir) else []
    image_files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    cards_html = ""
    for img in image_files:
        url = f"/debug_crops/latest/{img}?t={int(time.time() * 1000)}"
        cards_html += f"""
        <div style="background: #1e1e2e; border: 1px solid #313244; border-radius: 12px; padding: 16px; display: flex; flex-direction: column; align-items: center; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
            <h4 style="color: #cdd6f4; margin: 0 0 12px 0; font-family: monospace; font-size: 14px; word-break: break-all; text-align: center;">{img}</h4>
            <div style="max-height: 260px; overflow: auto; display: flex; justify-content: center; align-items: center; background: #11111b; border-radius: 8px; padding: 8px; width: 100%; box-sizing: border-box;">
                <img src="{url}" alt="{img}" style="max-width: 100%; max-height: 240px; object-fit: contain; border-radius: 4px;" />
            </div>
            <a href="{url}" target="_blank" style="margin-top: 12px; color: #89b4fa; text-decoration: none; font-size: 13px; font-weight: bold;">View Full Size →</a>
        </div>
        """

    if not cards_html:
        cards_html = '<p style="color: #a6adc8; grid-column: 1 / -1; text-align: center; padding: 40px;">No debug images saved yet. Scan a card via the client or API to populate this view.</p>'

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Pokémon OCR Pipeline Debug Gallery</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                background-color: #11111b;
                color: #cdd6f4;
                margin: 0;
                padding: 24px;
            }}
            .header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid #313244;
                padding-bottom: 16px;
                margin-bottom: 24px;
            }}
            .grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
                gap: 20px;
            }}
            .btn {{
                background: #89b4fa;
                color: #11111b;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
                cursor: pointer;
                text-decoration: none;
            }}
            .btn:hover {{
                background: #b4befe;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <div>
                <h1 style="margin: 0; font-size: 24px; color: #89b4fa;">🔍 Pokémon OCR Debug Visualizer</h1>
                <p style="margin: 4px 0 0 0; color: #a6adc8; font-size: 14px;">Raw client uploads and all manipulated preprocessing filter variants</p>
            </div>
            <div>
                <button class="btn" onclick="location.reload()">🔄 Refresh Images</button>
            </div>
        </div>
        <div class="grid">
            {cards_html}
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

