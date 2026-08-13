# Multi-stage / optimized Dockerfile for Python ML & OCR FastAPI backend
FROM python:3.10-slim

# Prevent Python from writing .pyc files & enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install system dependencies required by OpenCV, EasyOCR, and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first to keep image lightweight (avoids 4GB+ CUDA wheels)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-warm EasyOCR model cache so inference is instantaneous on cold start
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"

# Copy application source code and models
COPY . .

# Expose standard port
EXPOSE 8000

# Start FastAPI server using uvicorn binding to dynamic $PORT
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}
