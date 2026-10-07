"""FastAPI application; the checkpoint is loaded once during lifespan."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from time import perf_counter

from backend.app.core.config import CHECKPOINT_PATH, CORS_ORIGINS, MAX_UPLOAD_BYTES
from backend.app.routes.prediction import router
from backend.app.services.inference_service import InferenceService
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.inference_service = InferenceService(CHECKPOINT_PATH)
    app.state.metrics = {"requests": 0, "errors": 0}
    yield
    app.state.inference_service = None


app = FastAPI(
    title="DermaTriage API",
    description="Research prototype for dermoscopic-image classification. Not for clinical diagnosis.",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "DermaTriage API",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model-info",
    }


app.include_router(router)


@app.middleware("http")
async def log_request_timing(request: Request, call_next):
    started = perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(perf_counter() - started) * 1000:.2f}"
    return response


@app.middleware("http")
async def reject_oversized_predict_request(request: Request, call_next):
    if request.method == "POST" and request.url.path == "/predict":
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                request_bytes = int(content_length)
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"detail": {"code": "invalid_content_length", "message": "Invalid request size header."}},
                )
            # Multipart boundaries and headers add a small amount beyond the image bytes.
            if request_bytes > MAX_UPLOAD_BYTES + 64 * 1024:
                return JSONResponse(
                    status_code=413,
                    content={"detail": {"code": "file_too_large", "message": "Image exceeds the upload size limit."}},
                )
    return await call_next(request)

