from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import PlateJob, PlateImage, PlateResult, JobStatus
from app.schemas.job import JobCreate
from app.core.logging import get_logger

logger = get_logger(__name__)


class JobService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_job(
        self,
        client_id: str,
        job_data: JobCreate,
        image_path: str,
        original_filename: str,
        content_type: str,
        file_size: int,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ) -> PlateJob:
        job = PlateJob(
            client_id=client_id,
            sample_id=job_data.sample_id,
            plate_type=job_data.plate_type.value,
            capture_method=job_data.capture_method.value,
            captured_at=job_data.captured_at,
            operator_id=job_data.operator_id,
            facility_id=job_data.facility_id,
            dilution=job_data.dilution,
            incubation_hours=job_data.incubation_hours,
            lighting_type=job_data.lighting_type,
            status=JobStatus.QUEUED,
        )
        self.db.add(job)
        await self.db.flush()

        image = PlateImage(
            job_id=job.id,
            original_filename=original_filename,
            storage_path=image_path,
            content_type=content_type,
            file_size_bytes=file_size,
            width=image_width,
            height=image_height,
        )
        self.db.add(image)
        await self.db.commit()
        
        # Re-fetch with eager loading to avoid lazy load issues in async context
        stmt = (
            select(PlateJob)
            .options(selectinload(PlateJob.image))
            .where(PlateJob.id == job.id)
        )
        result = await self.db.execute(stmt)
        job = result.scalar_one()

        logger.info("Created job", job_id=str(job.id), client_id=client_id)
        return job

    async def get_job(self, job_id: UUID, client_id: str) -> Optional[PlateJob]:
        stmt = (
            select(PlateJob)
            .options(selectinload(PlateJob.image), selectinload(PlateJob.result))
            .where(PlateJob.id == job_id, PlateJob.client_id == client_id)
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_job_by_id(self, job_id: UUID) -> Optional[PlateJob]:
        stmt = (
            select(PlateJob)
            .options(selectinload(PlateJob.image), selectinload(PlateJob.result))
            .where(PlateJob.id == job_id)
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_jobs(
        self,
        client_id: str,
        page: int = 1,
        page_size: int = 20,
        status: Optional[JobStatus] = None,
    ) -> tuple[list[PlateJob], int]:
        base_query = select(PlateJob).where(PlateJob.client_id == client_id)

        if status:
            base_query = base_query.where(PlateJob.status == status)

        count_stmt = select(func.count()).select_from(base_query.subquery())
        count_result = await self.db.execute(count_stmt)
        total = count_result.scalar() or 0

        stmt = (
            base_query
            .order_by(PlateJob.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        result = await self.db.execute(stmt)
        jobs = list(result.scalars().all())

        return jobs, total

    async def update_job_status(
        self,
        job_id: UUID,
        status: JobStatus,
        progress: float = 0.0,
        error_message: Optional[str] = None,
    ) -> None:
        stmt = select(PlateJob).where(PlateJob.id == job_id)
        result = await self.db.execute(stmt)
        job = result.scalar_one_or_none()

        if job:
            job.status = status
            job.progress = progress
            if error_message:
                job.error_message = error_message
            if status in (JobStatus.SUCCEEDED, JobStatus.FAILED):
                job.completed_at = datetime.utcnow()
            await self.db.commit()
            logger.info("Updated job status", job_id=str(job_id), status=status.value)

    async def save_result(
        self,
        job_id: UUID,
        cfu_count_total: Optional[int],
        detections: list,
        quality: dict,
        confidence: dict,
        artifacts: dict,
        model_metadata: dict,
    ) -> PlateResult:
        result = PlateResult(
            job_id=job_id,
            cfu_count_total=cfu_count_total,
            detections=detections,
            quality=quality,
            confidence=confidence,
            artifacts=artifacts,
            model_metadata=model_metadata,
        )
        self.db.add(result)
        await self.db.commit()
        await self.db.refresh(result)

        logger.info("Saved job result", job_id=str(job_id), cfu_count=cfu_count_total)
        return result
