from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class Detection(BaseModel):
    x: float = Field(..., ge=0.0, le=1.0, description="Normalized X coordinate")
    y: float = Field(..., ge=0.0, le=1.0, description="Normalized Y coordinate")
    radius_px: int = Field(..., ge=1, description="Radius in pixels")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class QualityMetrics(BaseModel):
    plate_found: bool = Field(..., description="Whether plate was detected in image")
    focus_score: float = Field(..., ge=0.0, le=1.0, description="Image focus quality")
    glare_score: float = Field(..., ge=0.0, le=1.0, description="Amount of glare detected")
    overgrowth_detected: bool = Field(..., description="Whether overgrowth/TNTC condition detected")


class ConfidenceMetrics(BaseModel):
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    needs_review: bool = Field(..., description="Whether manual review is recommended")
    reason_codes: list[str] = Field(default_factory=list, description="Reasons for review flag")


class Artifacts(BaseModel):
    annotated_image_url: Optional[str] = Field(
        None, description="Signed URL for annotated image (null if no detections)"
    )


class ModelMetadata(BaseModel):
    model_name: str = Field(..., description="Name of the detection model")
    model_version: str = Field(..., description="Version of the detection model")
    pipeline_hash: str = Field(..., description="SHA256 hash of the pipeline configuration")


class JobResultResponse(BaseModel):
    job_id: UUID
    cfu_count_total: Optional[int] = Field(
        None, description="CFU count (null for invalid/TNTC)"
    )
    detections: list[Detection] = Field(default_factory=list)
    quality: QualityMetrics
    confidence: ConfidenceMetrics
    artifacts: Artifacts
    model_metadata: ModelMetadata

    class Config:
        from_attributes = True
