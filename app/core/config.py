from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    app_name: str = "CFU Detection & Counting Service"
    app_version: str = "1.0.0"
    debug: bool = False

    # Database
    database_url: str = "postgresql+asyncpg://cfu_user:cfu_password@localhost:5432/cfu_db"
    sync_database_url: str = "postgresql://cfu_user:cfu_password@localhost:5432/cfu_db"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Celery
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"

    # MinIO/S3
    minio_endpoint: str = "localhost:9000"
    minio_external_endpoint: Optional[str] = None  # External endpoint for browser-accessible URLs
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "cfu-images"
    minio_use_ssl: bool = False

    # Auth
    secret_key: str = "your-super-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # Processing
    min_image_resolution: int = 640
    max_image_size_mb: int = 50
    focus_score_threshold: float = 0.5
    glare_score_threshold: float = 0.5
    max_cfu_before_review: int = 300
    signed_url_expiry_seconds: int = 3600

    # Model
    model_name: str = "cfu-detector"
    model_version: str = "v1.0.0"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
