FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    DERMATRIAGE_CHECKPOINT=/app/ml/models/best_model.pth

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt /app/backend/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /app/backend/requirements.txt

COPY backend/app /app/backend/app
COPY ml/src /app/ml/src
COPY ml/models/best_model.pth /app/ml/models/best_model.pth
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1
CMD ["sh", "-c", "python -m uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

