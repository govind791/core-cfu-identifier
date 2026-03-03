from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class PlateType(str, Enum):
    TFA_90MM = "TFA_90MM"
    TFA_100MM = "TFA_100MM"


class CaptureMethod(str, Enum):
    PHONE = "PHONE"
    SCANNER = "SCANNER"
    CAMERA_RIG = "CAMERA_RIG"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class JobCreate(BaseModel):
    sample_id: str = Field(..., min_length=1, max_length=100, description="External reference ID")
    plate_type: PlateType = Field(..., description="Type of TFA plate")
    capture_method: CaptureMethod = Field(..., description="How the image was captured")
    captured_at: datetime = Field(..., description="When the image was captured")

    # Optional but recommended fields
    operator_id: Optional[str] = Field(None, max_length=100, description="Operator identifier")
    facility_id: Optional[str] = Field(None, max_length=100, description="Facility identifier")
    dilution: Optional[str] = Field(None, max_length=50, description="Dilution factor")
    incubation_hours: Optional[float] = Field(None, ge=0, description="Hours of incubation")
    lighting_type: Optional[str] = Field(None, max_length=50, description="Type of lighting used")


class JobResponse(BaseModel):
    job_id: UUID
    status: JobStatus

    class Config:
        from_attributes = True


class JobStatusResponse(BaseModel):
    job_id: UUID
    status: JobStatus
    progress: float = Field(ge=0.0, le=1.0)
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class JobListResponse(BaseModel):
    jobs: list[JobStatusResponse]
    total: int
    page: int
    page_size: int


class BatchJobCreate(BaseModel):
    jobs: list[JobCreate]


class BatchJobResponse(BaseModel):
    job_ids: list[UUID]
    count: int
