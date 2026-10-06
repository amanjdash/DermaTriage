from math import isclose

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
    requests_total: int = 0
    errors_total: int = 0


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str | None
    model_available: bool
    classes: list[str]
    uncertainty_method: str
    mc_passes: int
    dataset_reference: str
    message: str | None = None


class PredictionResponse(BaseModel):
    prediction: str
    deterministic_prediction: str
    confidence: float = Field(ge=0, le=1)
    uncertainty: float = Field(ge=0)
    uncertainty_normalized: float = Field(ge=0, le=1)
    uncertainty_method: str
    mc_passes: int
    probabilities: dict[str, float]
    class_probability_variance: dict[str, float]
    inference_time_ms: float = Field(ge=0)
    preprocessing_time_ms: float = Field(ge=0)
    model_inference_time_ms: float = Field(ge=0)
    model_name: str
    model_version: str
    dataset_reference: str
    device: str

    @field_validator("probabilities")
    @classmethod
    def validate_probability_distribution(cls, value: dict[str, float]) -> dict[str, float]:
        if len(value) != 7 or any(item < 0 or item > 1 for item in value.values()):
            raise ValueError("probabilities must contain seven values between zero and one")
        if not isclose(sum(value.values()), 1.0, rel_tol=0.0, abs_tol=1e-4):
            raise ValueError("probabilities must sum to one")
        return value

