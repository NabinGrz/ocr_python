# Pokémon Card OCR

Python OCR and card-identification service with Streamlit and FastAPI entry points.

## Accuracy pipeline

1. Detect and normalize the card using YOLO when trained weights are available, otherwise use contour-based perspective correction.
2. Run EasyOCR over enhanced and natural-resolution header/footer variants.
3. Parse and vote on name, HP, and collector-ID observations.
4. Rank matches from the bundled offline English-card catalog using independent evidence from number, set total, name, and HP.
5. Use visual matching only when a FAISS catalog exists and its result agrees with OCR evidence.
6. For burst scans, reject measurably poor frames and form fuzzy consensus from the accepted frames.

The service intentionally returns unverified ranked candidates when the evidence cannot identify an exact printing. Name and HP alone are not enough because reprints can share both values.

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000
```

For the Streamlit interface:

```bash
.venv/bin/streamlit run app.py
```

## Refresh the offline catalog

The generated `models/card_catalog.json` contains the English cards published by the official `PokemonTCG/pokemon-tcg-data` repository at build time.

```bash
.venv/bin/python build_local_card_catalog.py
```

Set `POKEMON_TCG_API_KEY` to allow live API fallback with higher rate limits when a local match is unavailable.

## Verify

```bash
.venv/bin/python -m unittest discover -v
.venv/bin/python test_extraction.py
```

The optional `models/yolov8_card_detector.pt`, `models/card_index.faiss`, and `models/card_metadata.json` assets are not included. Their layers fail closed and use the OCR/catalog pipeline until properly trained artifacts are supplied.
