from fastapi import APIRouter

from app.api.endpoints import jobs, auth, health

api_router = APIRouter()

api_router.include_router(health.router, tags=["health"])
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(jobs.router, prefix="/plates/jobs", tags=["jobs"])
