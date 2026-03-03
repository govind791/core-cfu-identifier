"""
Celery Tasks
============
process_plate_image is the single task that:
  1. Updates job status → RUNNING
  2. Downloads original image from MinIO
  3. Runs CFUPipeline
  4. Uploads annotated image (if detections found)
  5. Saves results to DB
  6. Updates job status → SUCCEEDED / FAILED
"""

from __future__ import annotations

import traceback
from uuid import UUID

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.core.logging import get_logger
from app.models import JobStatus, PlateJob
from app.services.storage import StorageService
from app.worker.celery_app import celery_app
from app.worker.pipeline import get_pipeline

logger = get_logger(__name__)

# ── Synchronous DB session for Celery (not async) ─────────────────────────
_sync_engine = None
_sync_session_factory = None


def _get_sync_session() -> Session:
    global _sync_engine, _sync_session_factory
    if _sync_session_factory is None:
        _sync_engine = create_engine(
            settings.sync_database_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
        _sync_session_factory = sessionmaker(
            bind=_sync_engine, autocommit=False, autoflush=False
        )
    return _sync_session_factory()


# ── Helper: update job status ─────────────────────────────────────────────
def _update_status(
    session: Session,
    job_id: UUID,
    status: JobStatus,
    progress: float = 0.0,
    error_message: str | None = None,
) -> None:
    from datetime import datetime
    from sqlalchemy import select

    job = session.execute(
        select(PlateJob).where(PlateJob.id == job_id)
    ).scalar_one_or_none()

    if job is None:
        logger.error("Job not found for status update", job_id=str(job_id))
        return

    job.status = status
    job.progress = progress
    if error_message:
        job.error_message = error_message[:1000]
    if status in (JobStatus.SUCCEEDED, JobStatus.FAILED):
        job.completed_at = datetime.utcnow()
    session.commit()


@celery_app.task(
    bind=True,
    name="app.worker.tasks.process_plate_image",
    max_retries=2,
    default_retry_delay=30,
    acks_late=True,
    reject_on_worker_lost=True,
)
def process_plate_image(self, job_id_str: str) -> dict:
    """
    Main Celery task for CFU detection.

    Args:
        job_id_str: UUID string of the PlateJob row.

    Returns:
        dict summary of the result (stored in Celery result backend).
    """
    job_id = UUID(job_id_str)
    session = _get_sync_session()

    try:
        # ── Step 1: Fetch job metadata ─────────────────────────────
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload
        from app.models import PlateImage, PlateResult, AuditLog

        job = session.execute(
            select(PlateJob)
            .options(selectinload(PlateJob.image))
            .where(PlateJob.id == job_id)
        ).scalar_one_or_none()

        if job is None:
            raise RuntimeError(f"Job {job_id} not found")

        if job.status == JobStatus.FAILED:
            logger.warning("Job already failed — skipping", job_id=job_id_str)
            return {"status": "skipped"}

        # ── Step 2: Mark RUNNING ───────────────────────────────────
        _update_status(session, job_id, JobStatus.RUNNING, progress=0.05)
        _audit(session, job_id, "job_started", "worker")

        # ── Step 3: Download image ─────────────────────────────────
        storage = StorageService()
        image_bytes = storage.download_image(job.image.storage_path)
        _update_status(session, job_id, JobStatus.RUNNING, progress=0.20)
        logger.info("Downloaded image", job_id=job_id_str, bytes=len(image_bytes))

       # ── Step 4: Run detection pipeline ────────────────────────
        pipeline = get_pipeline()
        _update_status(session, job_id, JobStatus.RUNNING, progress=0.40)

        result = pipeline.run(image_bytes)
        _update_status(session, job_id, JobStatus.RUNNING, progress=0.80)

        # ── Normalize result (important fix) ──────────────────────
        # Support both object-style and dict-style pipeline outputs
        if isinstance(result, dict):
            cfu_count_total = result.get("cfu_count", 0)
            detections_json = []
            quality_json = {}
            confidence_json = {}
            artifacts_json = {}
            model_metadata_json = {}
            annotated_url = None
            needs_review = False
            processing_time_ms = 0
            overall_confidence = 1.0
        else:
            annotated_url = None

            if getattr(result, "annotated_image_bytes", None):
                storage = StorageService()
                annotated_path = storage.upload_annotated_image(
                    client_id=job.client_id,
                    job_id=job_id,
                    image_bytes=result.annotated_image_bytes,
                )
                annotated_url = storage.get_signed_url(annotated_path)

            detections_json = [
                {
                    "x": d.x,
                    "y": d.y,
                    "radius_px": d.radius_px,
                    "score": d.score,
                }
                for d in getattr(result, "detections", [])
            ]

            quality_json = getattr(result, "quality", {})
            confidence_json = getattr(result, "confidence", {})
            artifacts_json = {"annotated_image_url": annotated_url}
            model_metadata_json = getattr(result, "model_metadata", {})
            cfu_count_total = getattr(result, "cfu_count_total", 0)
            needs_review = getattr(result, "needs_review", False)
            processing_time_ms = getattr(result, "processing_time_ms", 0)
            overall_confidence = getattr(result, "overall_confidence", 1.0)

        # ── Step 5: Upload annotated image ────────────────────────
        annotated_url: str | None = None
        if result.annotated_image_bytes:
            annotated_path = storage.upload_annotated_image(
                client_id=job.client_id,
                job_id=job_id,
                image_bytes=result.annotated_image_bytes,
            )
            annotated_url = storage.get_signed_url(annotated_path)
            _audit(session, job_id, "annotated_image_created", "worker")

        # ── Step 6: Serialize detections ──────────────────────────
        detections_json = [
            {
                "x": d.x,
                "y": d.y,
                "radius_px": d.radius_px,
                "score": d.score,
            }
            for d in result.detections
        ]

        quality_json = {
            "plate_found": result.quality.plate_found,
            "focus_score": result.quality.focus_score,
            "glare_score": result.quality.glare_score,
            "overgrowth_detected": result.quality.overgrowth_detected,
        }

        confidence_json = {
            "overall_score": result.overall_confidence,
            "needs_review": result.needs_review,
            "reason_codes": result.reason_codes,
        }

        artifacts_json = {"annotated_image_url": annotated_url}

        model_metadata_json = {
            "model_name": result.model_name,
            "model_version": result.model_version,
            "pipeline_hash": result.pipeline_hash,
            "processing_time_ms": result.processing_time_ms,
        }

           # ── Step 7: Save result to DB ──────────────────────────────
        from app.models import PlateResult

        plate_result = PlateResult(
            job_id=job_id,
            cfu_count_total=cfu_count_total,
            detections=detections_json,
            quality=quality_json,
            confidence=confidence_json,
            artifacts=artifacts_json,
            model_metadata=model_metadata_json,
        )

        session.add(plate_result)
        session.commit()

        # ── Step 8: Mark SUCCEEDED ─────────────────────────────────
        _update_status(session, job_id, JobStatus.SUCCEEDED, progress=1.0)

        return {
            "status": "succeeded",
            "job_id": job_id_str,
            "cfu_count": cfu_count_total,
            "needs_review": needs_review,
        }

        # ── Step 8: Mark SUCCEEDED ─────────────────────────────────
        _update_status(session, job_id, JobStatus.SUCCEEDED, progress=1.0)
        _audit(
            session, job_id, "job_completed", "worker",
            details={
                "cfu_count": result.cfu_count_total,
                "needs_review": result.needs_review,
                "processing_ms": result.processing_time_ms,
            },
        )

        logger.info(
            "Job completed successfully",
            job_id=job_id_str,
            cfu_count=result.cfu_count_total,
            confidence=result.overall_confidence,
        )

        return {
            "status": "succeeded",
            "job_id": job_id_str,
            "cfu_count": result.cfu_count_total,
            "needs_review": result.needs_review,
        }

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("Job failed", job_id=job_id_str, error=str(exc), traceback=tb)

        try:
            _update_status(
                session, job_id, JobStatus.FAILED,
                error_message=f"{type(exc).__name__}: {str(exc)}"[:1000],
            )
            _audit(session, job_id, "job_failed", "worker", details={"error": str(exc)})
        except Exception as inner:
            logger.error("Failed to update failure status", error=str(inner))

        # Retry for transient errors (network, I/O), not logic errors
        if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
            raise self.retry(exc=exc)

        return {"status": "failed", "job_id": job_id_str, "error": str(exc)}

    finally:
        session.close()


def _audit(
    session: Session,
    job_id: UUID,
    action: str,
    actor: str,
    details: dict | None = None,
) -> None:
    """Write a synchronous audit log row."""
    from app.models.audit import AuditLog
    log = AuditLog(
        job_id=job_id,
        action=action,
        actor=actor,
        actor_type="system",
        details=details,
    )
    session.add(log)
    session.commit()