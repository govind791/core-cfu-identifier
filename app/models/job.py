import enum
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Enum, Float, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class PlateJob(Base):
    __tablename__ = "plate_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    client_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    sample_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus), default=JobStatus.QUEUED, nullable=False
    )
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    error_message: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)

    # Metadata
    plate_type: Mapped[str] = mapped_column(String(50), nullable=False)
    capture_method: Mapped[str] = mapped_column(String(50), nullable=False)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Optional metadata
    operator_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    facility_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    dilution: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    incubation_hours: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    lighting_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    image: Mapped["PlateImage"] = relationship(
        "PlateImage", back_populates="job", uselist=False, cascade="all, delete-orphan"
    )
    result: Mapped["PlateResult"] = relationship(
        "PlateResult", back_populates="job", uselist=False, cascade="all, delete-orphan"
    )
    audit_logs: Mapped[list["AuditLog"]] = relationship(
        "AuditLog", back_populates="job", cascade="all, delete-orphan"
    )


from app.models.image import PlateImage
from app.models.result import PlateResult
from app.models.audit import AuditLog
