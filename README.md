# CFU Detection & Counting Service

A robust, auditable FastAPI backend for Colony Forming Unit (CFU) detection and counting on TFA agar plate images.

## Features

- **Async Image Processing** - Queue-based colony detection via Celery workers
- **CFU Detection** - Automated colony counting with quality metrics
- **Annotated Output** - Visual output with colony markers when detected
- **Multi-tenant** - Client-scoped data isolation with JWT authentication
- **Audit Trail** - Full traceability of all operations
- **S3-Compatible Storage** - MinIO for local dev, AWS S3 for production

## Tech Stack

- **API**: FastAPI + Uvicorn
- **Database**: PostgreSQL with SQLAlchemy (async)
- **Queue**: Redis + Celery
- **Storage**: MinIO (S3-compatible)
- **Containerization**: Docker + Docker Compose

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+ (for local development)

### Run with Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api worker

# Stop services
docker-compose down
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| API | 8000 | FastAPI application |
| Flower | 5555 | Celery monitoring |
| MinIO Console | 9001 | Object storage UI |
| PostgreSQL | 5432 | Database |
| Redis | 6379 | Message broker |

## API Documentation

Once running, access:
- **Swagger UI**: http://localhost:8000/v1/docs
- **ReDoc**: http://localhost:8000/v1/redoc
- **OpenAPI JSON**: http://localhost:8000/v1/openapi.json

## API Usage

### 1. Get Access Token

```bash
curl -X POST http://localhost:8000/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"client_id": "demo_client", "client_secret": "demo_secret"}'
```

### 2. Submit a Plate Image

```bash
curl -X POST http://localhost:8000/v1/plates/jobs \
  -H "Authorization: Bearer <token>" \
  -F "image=@plate.jpg" \
  -F "sample_id=SAMPLE-001" \
  -F "plate_type=TFA_90MM" \
  -F "capture_method=PHONE" \
  -F "captured_at=2024-01-15T10:30:00Z"
```

### 3. Check Job Status

```bash
curl http://localhost:8000/v1/plates/jobs/{job_id} \
  -H "Authorization: Bearer <token>"
```

### 4. Get Results

```bash
curl http://localhost:8000/v1/plates/jobs/{job_id}/result \
  -H "Authorization: Bearer <token>"
```

## Result Contract

```json
{
  "cfu_count_total": 42,
  "detections": [
    {"x": 0.45, "y": 0.51, "radius_px": 8, "score": 0.87}
  ],
  "quality": {
    "plate_found": true,
    "focus_score": 0.81,
    "glare_score": 0.12,
    "overgrowth_detected": false
  },
  "confidence": {
    "overall_score": 0.78,
    "needs_review": false,
    "reason_codes": []
  },
  "artifacts": {
    "annotated_image_url": "signed-url-or-null"
  },
  "model_metadata": {
    "model_name": "cfu-detector",
    "model_version": "v1.0.0",
    "pipeline_hash": "sha256..."
  }
}
```

## Local Development

### Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start infrastructure (DB, Redis, MinIO)
docker-compose up -d postgres redis minio minio-init

# Run migrations
alembic upgrade head

# Start API
uvicorn app.main:app --reload

# Start worker (in another terminal)
celery -A app.worker.celery_app worker --loglevel=info
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| DATABASE_URL | postgresql+asyncpg://... | Async database URL |
| REDIS_URL | redis://localhost:6379/0 | Redis connection |
| MINIO_ENDPOINT | localhost:9000 | MinIO endpoint |
| MINIO_ACCESS_KEY | minioadmin | MinIO access key |
| MINIO_SECRET_KEY | minioadmin | MinIO secret key |
| SECRET_KEY | your-super-secret-key | JWT signing key |

## Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=app --cov-report=html
```

## Project Structure

```
├── app/
│   ├── api/
│   │   ├── endpoints/
│   │   │   ├── auth.py      # Authentication
│   │   │   ├── health.py    # Health checks
│   │   │   └── jobs.py      # Job CRUD
│   │   └── router.py
│   ├── core/
│   │   ├── config.py        # Settings
│   │   ├── database.py      # DB setup
│   │   ├── logging.py       # Structured logs
│   │   └── security.py      # JWT auth
│   ├── models/
│   │   ├── audit.py         # Audit logs
│   │   ├── image.py         # Plate images
│   │   ├── job.py           # Plate jobs
│   │   └── result.py        # Processing results
│   ├── schemas/
│   │   ├── auth.py          # Auth schemas
│   │   ├── job.py           # Job schemas
│   │   └── result.py        # Result schemas
│   ├── services/
│   │   ├── audit_service.py # Audit logging
│   │   ├── job_service.py   # Job operations
│   │   └── storage.py       # S3/MinIO
│   ├── worker/
│   │   ├── celery_app.py    # Celery config
│   │   ├── pipeline.py      # Image processing
│   │   └── tasks.py         # Celery tasks
│   └── main.py              # FastAPI app
├── alembic/                  # DB migrations
├── tests/                    # Test suite
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.worker
└── requirements.txt
```

## Key Design Decisions

1. **No inline ML in API threads** - All processing via Celery workers
2. **Annotated images only when detections exist** - Per TRD requirement
3. **Every result is reproducible** - Model metadata and pipeline hash stored
4. **Failures are valid outcomes** - Jobs can fail gracefully with error messages

## Demo Clients

For development/testing:

| Client ID | Secret | Scopes |
|-----------|--------|--------|
| demo_client | demo_secret | jobs:read, jobs:write |
| lab_alpha | alpha_secret_123 | jobs:read, jobs:write |

## License

Proprietary - Core QC Labs
