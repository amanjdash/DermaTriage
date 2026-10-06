"""One in-memory model instance shared by API requests."""

from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock

from PIL import Image

logger = logging.getLogger("dermatriage.inference")


class InferenceService:
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = Path(checkpoint_path)
        self._inference_lock = Lock()
        self.predictor = None
        self.load_error: str | None = None
        if not self.checkpoint_path.is_file():
            self.load_error = "Trained model checkpoint is not loaded."
            logger.warning("Trained model checkpoint is not available at %s", self.checkpoint_path)
            return
        try:
            # Imported only when a checkpoint exists so health/docs can run before training.
            from ml.src.inference import load_predictor

            self.predictor = load_predictor(self.checkpoint_path, device="cpu")
        except Exception:  # startup remains useful for /health and /model-info
            logger.exception("Could not load the model checkpoint")
            self.load_error = "Model checkpoint could not be loaded."

    @property
    def is_ready(self) -> bool:
        return self.predictor is not None

    @property
    def model_version(self) -> str | None:
        if not self.predictor:
            return None
        return str(self.predictor["checkpoint"].get("model_version", "unknown"))

    def predict(self, image: Image.Image, mc_passes: int):
        if not self.predictor:
            raise RuntimeError(self.load_error or "Model is not available")
        from ml.src.inference import predict_image

        # MC inference changes Dropout module modes; serialize access to this shared model.
        with self._inference_lock:
            return predict_image(self.predictor, image, mc_passes=mc_passes, stochastic=True)

