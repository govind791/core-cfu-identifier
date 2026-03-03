import io
from datetime import datetime, timedelta
from typing import BinaryIO, Optional
from uuid import UUID

from minio import Minio
from minio.error import S3Error

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class StorageService:
    def __init__(self):
        self.client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_use_ssl,
        )
        self.bucket = settings.minio_bucket
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self) -> None:
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                logger.info("Created bucket", bucket=self.bucket)
        except S3Error as e:
            logger.error("Failed to ensure bucket exists", error=str(e))
            raise

    def _build_path(
        self, client_id: str, job_id: UUID, filename: str, date: Optional[datetime] = None
    ) -> str:
        if date is None:
            date = datetime.utcnow()
        return f"{client_id}/{date.year}/{date.month:02d}/{date.day:02d}/{job_id}/{filename}"

    def upload_image(
        self,
        client_id: str,
        job_id: UUID,
        file_data: BinaryIO,
        filename: str,
        content_type: str,
        file_size: int,
    ) -> str:
        path = self._build_path(client_id, job_id, filename)

        try:
            self.client.put_object(
                self.bucket,
                path,
                file_data,
                file_size,
                content_type=content_type,
            )
            logger.info("Uploaded image", path=path, size=file_size)
            return path
        except S3Error as e:
            logger.error("Failed to upload image", path=path, error=str(e))
            raise

    def upload_annotated_image(
        self,
        client_id: str,
        job_id: UUID,
        image_bytes: bytes,
    ) -> str:
        path = self._build_path(client_id, job_id, "annotated.png")

        try:
            self.client.put_object(
                self.bucket,
                path,
                io.BytesIO(image_bytes),
                len(image_bytes),
                content_type="image/png",
            )
            logger.info("Uploaded annotated image", path=path)
            return path
        except S3Error as e:
            logger.error("Failed to upload annotated image", path=path, error=str(e))
            raise

    def get_signed_url(self, path: str, expires: Optional[int] = None) -> str:
        """
        Generate a URL for accessing the object.
        For annotated images, returns a direct public URL.
        For other objects, returns a presigned URL.
        """
        # Use direct public URL for annotated images (already processed, non-sensitive)
        if "annotated" in path:
            endpoint = settings.minio_external_endpoint or settings.minio_endpoint
            protocol = "https" if settings.minio_use_ssl else "http"
            url = f"{protocol}://{endpoint}/{self.bucket}/{path}"
            return url
        
        # For other objects, use presigned URLs
        if expires is None:
            expires = settings.signed_url_expiry_seconds

        try:
            url = self.client.presigned_get_object(
                self.bucket,
                path,
                expires=timedelta(seconds=expires),
            )
            
            # Replace internal endpoint with external for browser access
            if settings.minio_external_endpoint and settings.minio_external_endpoint != settings.minio_endpoint:
                url = url.replace(
                    f"://{settings.minio_endpoint}/",
                    f"://{settings.minio_external_endpoint}/"
                )
            
            return url
        except S3Error as e:
            logger.error("Failed to generate signed URL", path=path, error=str(e))
            raise

    def download_image(self, path: str) -> bytes:
        try:
            response = self.client.get_object(self.bucket, path)
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except S3Error as e:
            logger.error("Failed to download image", path=path, error=str(e))
            raise

    def delete_object(self, path: str) -> None:
        try:
            self.client.remove_object(self.bucket, path)
            logger.info("Deleted object", path=path)
        except S3Error as e:
            logger.error("Failed to delete object", path=path, error=str(e))
            raise


storage_service = StorageService()
