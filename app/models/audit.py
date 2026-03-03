import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.job import PlateJob


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("plate_jobs.id", ondelete="CASCADE"), index=True
    )

    action: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    actor: Mapped[str] = mapped_column(String(200), nullable=False)
    actor_type: Mapped[str] = mapped_column(String(50), nullable=False)  # "system", "client", "operator"

    # Additional context
    details: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    # Relationship
    job: Mapped["PlateJob"] = relationship("PlateJob", back_populates="audit_logs")
