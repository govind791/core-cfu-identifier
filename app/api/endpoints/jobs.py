import io
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import get_current_client
from app.models import JobStatus
from app.schemas.job import (
    CaptureMethod,
    JobCreate,
    JobListResponse,
    JobResponse,
    JobStatusResponse,
    PlateType,
)
from app.schemas.result import (
    Artifacts,
    ConfidenceMetrics,
    Detection,
    JobResultResponse,
    ModelMetadata,
    QualityMetrics,
)
from app.services.job_service import JobService
from app.services.audit_service import AuditService, AuditAction, ActorType
from app.services.storage import StorageService
from app.worker.tasks import process_plate_image
from app.core.config import settings
from app.core.logging import get_logger
from datetime import datetime

logger = get_logger(__name__)

router = APIRouter()

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
MAX_FILE_SIZE = settings.max_image_size_mb * 1024 * 1024


def get_storage_service() -> StorageService:
    return StorageService()


@router.post("", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
async def create_job(
    image: UploadFile = File(...),
    sample_id: str = Form(...),
    plate_type: PlateType = Form(...),
    capture_method: CaptureMethod = Form(...),
    captured_at: datetime = Form(...),
    operator_id: Optional[str] = Form(None),
    facility_id: Optional[str] = Form(None),
    dilution: Optional[str] = Form(None),
    incubation_hours: Optional[float] = Form(None),
    lighting_type: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
    client: dict = Depends(get_current_client),
    storage: StorageService = Depends(get_storage_service),
):
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_CONTENT_TYPES)}",
        )

    file_content = await image.read()
    file_size = len(file_content)

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Maximum size: {settings.max_image_size_mb}MB",
        )

    try:
        pil_image = Image.open(io.BytesIO(file_content))
        image_width, image_height = pil_image.size

        if min(image_width, image_height) < settings.min_image_resolution:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image resolution too low. Minimum: {settings.min_image_resolution}px",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image file: {str(e)}",
        )

    job_data = JobCreate(
        sample_id=sample_id,
        plate_type=plate_type,
        capture_method=capture_method,
        captured_at=captured_at,
        operator_id=operator_id,
        facility_id=facility_id,
        dilution=dilution,
        incubation_hours=incubation_hours,
        lighting_type=lighting_type,
    )

    job_service = JobService(db)

    temp_job = await job_service.create_job(
        client_id=client["client_id"],
        job_data=job_data,
        image_path="pending",
        original_filename=image.filename or "image.jpg",
        content_type=image.content_type,
        file_size=file_size,
        image_width=image_width,
        image_height=image_height,
    )

    image_path = storage.upload_image(
        client_id=client["client_id"],
        job_id=temp_job.id,
        file_data=io.BytesIO(file_content),
        filename=image.filename or "original.jpg",
        content_type=image.content_type,
        file_size=file_size,
    )

    temp_job.image.storage_path = image_path
    await db.commit()

    audit_service = AuditService(db)
    await audit_service.log(
        job_id=temp_job.id,
        action=AuditAction.JOB_CREATED,
        actor=client["client_id"],
        actor_type=ActorType.CLIENT,
        details={"sample_id": sample_id, "plate_type": plate_type.value},
    )

    process_plate_image.delay(str(temp_job.id))

    await audit_service.log(
        job_id=temp_job.id,
        action=AuditAction.JOB_QUEUED,
        actor="api",
        actor_type=ActorType.SYSTEM,
    )

    logger.info("Job created and queued", job_id=str(temp_job.id), client_id=client["client_id"])

    return JobResponse(job_id=temp_job.id, status=JobStatus.QUEUED)


@router.get("", response_model=JobListResponse)
async def list_jobs(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status_filter: Optional[str] = Query(None, alias="status"),
    db: AsyncSession = Depends(get_db),
    client: dict = Depends(get_current_client),
):
    job_status = None
    if status_filter:
        try:
            job_status = JobStatus(status_filter)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status. Allowed: {[s.value for s in JobStatus]}",
            )

    job_service = JobService(db)
    jobs, total = await job_service.list_jobs(
        client_id=client["client_id"],
        page=page,
        page_size=page_size,
        status=job_status,
    )

    return JobListResponse(
        jobs=[
            JobStatusResponse(
                job_id=job.id,
                status=job.status,
                progress=job.progress,
                error_message=job.error_message,
                created_at=job.created_at,
                completed_at=job.completed_at,
            )
            for job in jobs
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    client: dict = Depends(get_current_client),
):
    job_service = JobService(db)
    job = await job_service.get_job(job_id, client["client_id"])

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )

    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        progress=job.progress,
        error_message=job.error_message,
        created_at=job.created_at,
        completed_at=job.completed_at,
    )


@router.get("/{job_id}/result", response_model=JobResultResponse)
async def get_job_result(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    client: dict = Depends(get_current_client),
    storage: StorageService = Depends(get_storage_service),
):
    job_service = JobService(db)
    job = await job_service.get_job(job_id, client["client_id"])

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )

    if job.status == JobStatus.QUEUED or job.status == JobStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Job is still processing",
        )

    if job.status == JobStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Job failed: {job.error_message}",
        )

    if not job.result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Result not found",
        )

    result = job.result

    annotated_url = None
    if result.artifacts and result.artifacts.get("annotated_image_url"):
        pass

    detections = [Detection(**d) for d in (result.detections or [])]

    quality = result.quality or {}
    confidence = result.confidence or {}
    model_meta = result.model_metadata or {}

    return JobResultResponse(
        job_id=job.id,
        cfu_count_total=result.cfu_count_total,
        detections=detections,
        quality=QualityMetrics(
            plate_found=quality.get("plate_found", False),
            focus_score=quality.get("focus_score", 0.0),
            glare_score=quality.get("glare_score", 0.0),
            overgrowth_detected=quality.get("overgrowth_detected", False),
        ),
        confidence=ConfidenceMetrics(
            overall_score=confidence.get("overall_score", 0.0),
            needs_review=confidence.get("needs_review", True),
            reason_codes=confidence.get("reason_codes", []),
        ),
        artifacts=Artifacts(
            annotated_image_url=result.artifacts.get("annotated_image_url") if result.artifacts else None,
        ),
        model_metadata=ModelMetadata(
            model_name=model_meta.get("model_name", "unknown"),
            model_version=model_meta.get("model_version", "unknown"),
            pipeline_hash=model_meta.get("pipeline_hash", "unknown"),
        ),
    )


@router.post("/batch", response_model=dict, status_code=status.HTTP_201_CREATED)
async def create_batch_jobs(
    images: list[UploadFile] = File(...),
    metadata_json: str = Form(...),
    db: AsyncSession = Depends(get_db),
    client: dict = Depends(get_current_client),
    storage: StorageService = Depends(get_storage_service),
):
    import json

    try:
        metadata_list = json.loads(metadata_json)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid metadata JSON",
        )

    if len(images) != len(metadata_list):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Number of images must match number of metadata entries",
        )

    job_ids = []
    job_service = JobService(db)
    audit_service = AuditService(db)

    for image, meta in zip(images, metadata_list):
        if image.content_type not in ALLOWED_CONTENT_TYPES:
            continue

        file_content = await image.read()
        file_size = len(file_content)

        if file_size > MAX_FILE_SIZE:
            continue

        try:
            pil_image = Image.open(io.BytesIO(file_content))
            image_width, image_height = pil_image.size
        except Exception:
            continue

        job_data = JobCreate(
            sample_id=meta.get("sample_id", "unknown"),
            plate_type=PlateType(meta.get("plate_type", "TFA_90MM")),
            capture_method=CaptureMethod(meta.get("capture_method", "PHONE")),
            captured_at=datetime.fromisoformat(meta.get("captured_at", datetime.utcnow().isoformat())),
            operator_id=meta.get("operator_id"),
            facility_id=meta.get("facility_id"),
            dilution=meta.get("dilution"),
            incubation_hours=meta.get("incubation_hours"),
            lighting_type=meta.get("lighting_type"),
        )

        job = await job_service.create_job(
            client_id=client["client_id"],
            job_data=job_data,
            image_path="pending",
            original_filename=image.filename or "image.jpg",
            content_type=image.content_type,
            file_size=file_size,
            image_width=image_width,
            image_height=image_height,
        )

        image_path = storage.upload_image(
            client_id=client["client_id"],
            job_id=job.id,
            file_data=io.BytesIO(file_content),
            filename=image.filename or "original.jpg",
            content_type=image.content_type,
            file_size=file_size,
        )

        job.image.storage_path = image_path
        await db.commit()

        await audit_service.log(
            job_id=job.id,
            action=AuditAction.JOB_CREATED,
            actor=client["client_id"],
            actor_type=ActorType.CLIENT,
            details={"batch": True},
        )

        process_plate_image.delay(str(job.id))
        job_ids.append(str(job.id))

    return {"job_ids": job_ids, "count": len(job_ids)}
