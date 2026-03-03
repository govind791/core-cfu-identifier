from typing import Optional

from pydantic import BaseModel


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    client_id: Optional[str] = None
    scopes: list[str] = []


class ClientCreate(BaseModel):
    client_id: str
    client_secret: str
    name: str
    scopes: list[str] = ["jobs:read", "jobs:write"]


class ClientAuth(BaseModel):
    client_id: str
    client_secret: str
