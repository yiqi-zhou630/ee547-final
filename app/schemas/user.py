# app/schemas/user.py
from pydantic import BaseModel, EmailStr
from datetime import datetime


class UserBase(BaseModel):
    email: EmailStr
    name: str | None = None
    role: str  # "teacher" / "student"


class UserCreate(UserBase):
    password: str


class UserUpdate(BaseModel):
    name: str | None = None
    password: str | None = None


class UserPublic(UserBase):
    id: int
    created_at: datetime

    model_config = {"from_attributes": True}