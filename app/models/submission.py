# app/models/submission.py
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Numeric,
    ForeignKey,
)
from sqlalchemy.sql import func
from app.db.base import Base

class Submission(Base):
    __tablename__ = "submissions"

    id = Column(Integer, primary_key=True, index=True)

    question_id = Column(Integer, ForeignKey("questions.id"), nullable=False, index=True)
    student_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    answer_text = Column(Text, nullable=False)

    # 状态：pending_ml / ml_scored / graded
    status = Column(String(20), nullable=False, default="pending_ml", index=True)

    # ML 评分
    ml_label = Column(String(50), nullable=True)
    ml_score = Column(Numeric(3, 1), nullable=True)
    ml_confidence = Column(Numeric(4, 3), nullable=True)
    model_version = Column(String(50), nullable=True)
    ml_explanation = Column(Text, nullable=True)
    ml_scored_at = Column(DateTime(timezone=True), nullable=True)

    # 老师评分
    teacher_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    final_score = Column(Numeric(3, 1), nullable=True)
    final_label = Column(String(50), nullable=True)
    teacher_comment = Column(Text, nullable=True)
    graded_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
