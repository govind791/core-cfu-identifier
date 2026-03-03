import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, ForeignKey, Integer, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.job import PlateJob


class PlateResult(Base):
    __tablename__ = "plate_results"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("plate_jobs.id", ondelete="CASCADE"), unique=True
    )

    # CFU count (null means invalid/TNTC)
    cfu_count_total: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Detections as JSONB array
    detections: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True, default=list)

    # Quality metrics as JSONB
    quality: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Confidence metrics as JSONB
    confidence: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Artifacts (annotated image URL, etc.)
    artifacts: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Model metadata
    model_metadata: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationship
    job: Mapped["PlateJob"] = relationship("PlateJob", back_populates="result")
