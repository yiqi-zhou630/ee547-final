# app/schemas/auth.py
from pydantic import BaseModel, EmailStr, ConfigDict


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    email: EmailStr | None = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: str  # 必填，可以重复
    role: str  # "teacher" / "student"


class UserPublic(BaseModel):

    id: int
    email: EmailStr
    name: str
    role: str

    model_config = ConfigDict(from_attributes=True)
