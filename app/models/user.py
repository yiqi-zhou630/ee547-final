# app/models/user.py
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from app.db.base import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=True)
    name = Column(String(100), nullable=True)
    role = Column(String(20), nullable=False)  # 'teacher' / 'student'
    created_at = Column(DateTime(timezone=True), server_default=func.now())
