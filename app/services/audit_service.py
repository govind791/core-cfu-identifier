from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import AuditLog
from app.core.logging import get_logger

logger = get_logger(__name__)


class AuditAction:
    JOB_CREATED = "job_created"
    JOB_QUEUED = "job_queued"
    JOB_STARTED = "job_started"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"
    IMAGE_UPLOADED = "image_uploaded"
    IMAGE_PROCESSED = "image_processed"
    RESULT_SAVED = "result_saved"
    ANNOTATED_IMAGE_CREATED = "annotated_image_created"


class ActorType:
    SYSTEM = "system"
    CLIENT = "client"
    OPERATOR = "operator"


class AuditService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def log(
        self,
        job_id: UUID,
        action: str,
        actor: str,
        actor_type: str = ActorType.SYSTEM,
        details: Optional[dict] = None,
    ) -> AuditLog:
        audit_log = AuditLog(
            job_id=job_id,
            action=action,
            actor=actor,
            actor_type=actor_type,
            details=details,
        )
        self.db.add(audit_log)
        await self.db.commit()
        await self.db.refresh(audit_log)

        logger.info(
            "Audit log created",
            job_id=str(job_id),
            action=action,
            actor=actor,
        )
        return audit_log

    async def get_logs_for_job(self, job_id: UUID, client_id: str) -> list[AuditLog]:
        stmt = (
            select(AuditLog)
            .join(AuditLog.job)
            .where(AuditLog.job_id == job_id)
            .order_by(AuditLog.timestamp.asc())
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())
