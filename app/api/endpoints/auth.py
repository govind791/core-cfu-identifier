from datetime import timedelta
from functools import lru_cache

from fastapi import APIRouter, HTTPException, status

from app.core.config import settings
from app.core.security import create_access_token, pwd_context, verify_password
from app.schemas.auth import Token, ClientAuth

router = APIRouter()


@lru_cache()
def get_demo_clients() -> dict:
    # Hash ONCE safely at startup using pwd_context directly
    demo_hash = pwd_context.hash("demo_secret")
    alpha_hash = pwd_context.hash("alpha_secret_123")

    return {
        "demo_client": {
            "hashed_secret": demo_hash,
            "name": "Demo Client",
            "scopes": ["jobs:read", "jobs:write"],
        },
        "lab_alpha": {
            "hashed_secret": alpha_hash,
            "name": "Lab Alpha",
            "scopes": ["jobs:read", "jobs:write"],
        },
    }


@router.post("/token", response_model=Token)
async def get_access_token(client_auth: ClientAuth):
    client = get_demo_clients().get(client_auth.client_id)

    if not client:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid client credentials",
        )

    if not verify_password(client_auth.client_secret, client["hashed_secret"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid client credentials",
        )

    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)

    access_token = create_access_token(
        data={
            "client_id": client_auth.client_id,
            "name": client["name"],
            "scopes": client["scopes"],
        },
        expires_delta=access_token_expires,
    )

    return Token(access_token=access_token, token_type="bearer")