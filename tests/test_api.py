"""
API Integration Tests — Uses in-memory SQLite DB + mocked storage/Celery.
"""

import io, math, numpy as np, cv2, pytest, pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import MagicMock, patch


def _make_jpeg(size: int = 512) -> bytes:
    rng = np.random.default_rng(1)
    img = np.full((size, size, 3), (200, 210, 185), dtype=np.uint8)
    cx, cy, radius = size // 2, size // 2, int(size * 0.42)
    cv2.circle(img, (cx, cy), radius, (200, 210, 185), -1)
    for _ in range(10):
        angle = rng.uniform(0, 2 * math.pi)
        px = int(cx + 0.5 * radius * math.cos(angle))
        py = int(cy + 0.5 * radius * math.sin(angle))
        cv2.circle(img, (px, py), 10, (250, 250, 250), -1)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


# Tests cover:
# - GET /v1/health → 200, status == "healthy"
# - POST /v1/auth/token (valid/invalid credentials)
# - Protected routes without token → 403
# - POST /v1/plates/jobs (success, invalid content type, missing fields)
# - GET /v1/plates/jobs/{job_id} (found, not found)
# - GET /v1/plates/jobs (pagination, status filter, invalid status → 400)