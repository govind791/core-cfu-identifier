from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import api_router
from app.core.config import settings
from app.core.database import engine, Base
from app.core.logging import setup_logging, get_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.debug)
    logger = get_logger(__name__)
    logger.info("Starting CFU Detection Service", version=settings.app_version)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created/verified")

    yield

    logger.info("Shutting down CFU Detection Service")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
## CFU Detection & Counting Service

A robust, auditable backend for TFA agar plate image analysis.

### Features
- **Image Upload**: Accept TFA agar plate images (JPG/PNG)
- **Async Processing**: Queue-based colony detection via Celery workers
- **CFU Counting**: Automated colony forming unit detection and counting
- **Quality Metrics**: Focus, glare, and overgrowth detection
- **Annotated Output**: Visual output with colony markers when detected
- **Multi-tenant**: Client-scoped data isolation
- **Audit Trail**: Full traceability of all operations

### API Workflow
1. Authenticate via `/v1/auth/token`
2. Submit plate image via `POST /v1/plates/jobs`
3. Check status via `GET /v1/plates/jobs/{job_id}`
4. Retrieve results via `GET /v1/plates/jobs/{job_id}/result`
    """,
    openapi_url="/v1/openapi.json",
    docs_url="/v1/docs",
    redoc_url="/v1/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/v1")


@app.get("/")
async def root():
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/v1/docs",
        "health": "/v1/health",
    }
