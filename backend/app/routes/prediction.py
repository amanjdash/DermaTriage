"""Prediction API routes with strict in-memory image validation."""

from __future__ import annotations

import logging
import time
from io import BytesIO

from backend.app.core.config import MAX_IMAGE_PIXELS, MAX_UPLOAD_BYTES, MC_PASSES
from backend.app.schemas.prediction import HealthResponse, ModelInfoResponse, PredictionResponse
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from ml.src.labels import CLASS_NAMES
from PIL import Image, ImageOps, UnidentifiedImageError

router = APIRouter()
logger = logging.getLogger("dermatriage.api")
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


def _service(request: Request):
    return request.app.state.inference_service


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    service = _service(request)
    metrics = request.app.state.metrics
    return HealthResponse(
        status="ok",
        model_loaded=service.is_ready,
        model_version=service.model_version,
        requests_total=metrics["requests"],
        errors_total=metrics["errors"],
    )


@router.get("/model-info", response_model=ModelInfoResponse)
async def model_info(request: Request):
    service = _service(request)
    checkpoint = service.predictor["checkpoint"] if service.predictor else {}
    return ModelInfoResponse(
        model_name="EfficientNet-B3",
        model_version=service.model_version,
        model_available=service.is_ready,
        classes=list(CLASS_NAMES),
        uncertainty_method="predictive_entropy",
        mc_passes=MC_PASSES,
        dataset_reference=checkpoint.get("dataset_reference", "HAM10000, Harvard Dataverse DOI:10.7910/DVN/DBW86T"),
        message=None if service.is_ready else service.load_error,
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, image: UploadFile = File(...)):
    service = _service(request)
    request.app.state.metrics["requests"] += 1
    if not service.is_ready:
        request.app.state.metrics["errors"] += 1
        raise HTTPException(status_code=503, detail={"code": "model_unavailable", "message": service.load_error})
    if image.content_type not in ALLOWED_MIME:
        request.app.state.metrics["errors"] += 1
        raise HTTPException(status_code=415, detail={"code": "unsupported_media_type", "message": "Upload a JPEG, PNG, or WebP image."})
    started = time.perf_counter()
    try:
        content = await image.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            request.app.state.metrics["errors"] += 1
            raise HTTPException(status_code=413, detail={"code": "file_too_large", "message": "Image exceeds the upload size limit."})
        try:
            with Image.open(BytesIO(content)) as candidate:
                if candidate.format not in ALLOWED_FORMATS:
                    raise HTTPException(status_code=415, detail={"code": "unsupported_image", "message": "Image format must be JPEG, PNG, or WebP."})
                if candidate.width * candidate.height > MAX_IMAGE_PIXELS:
                    raise HTTPException(status_code=413, detail={"code": "image_dimensions_too_large", "message": "Image dimensions exceed the processing limit."})
                candidate.verify()
            with Image.open(BytesIO(content)) as candidate:
                decoded = ImageOps.exif_transpose(candidate).convert("RGB")
        except HTTPException:
            raise
        except (OSError, UnidentifiedImageError, ValueError, Image.DecompressionBombError) as exc:
            raise HTTPException(status_code=400, detail={"code": "invalid_image", "message": "The uploaded file is not a readable image."}) from exc
        result = await run_in_threadpool(service.predict, decoded, MC_PASSES)
        result["inference_time_ms"] = max(result["inference_time_ms"], (time.perf_counter() - started) * 1000)
        deterministic = result.get("deterministic_prediction", result["prediction"])
        response = {**result, "deterministic_prediction": deterministic}
        return PredictionResponse(**response)
    except HTTPException:
        request.app.state.metrics["errors"] += 1
        raise
    except Exception as exc:
        request.app.state.metrics["errors"] += 1
        logger.exception("Prediction request failed")
        raise HTTPException(status_code=500, detail={"code": "inference_failed", "message": "Image analysis failed. Please try again."}) from exc
    finally:
        await image.close()

