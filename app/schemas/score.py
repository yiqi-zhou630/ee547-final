# app/schemas/score.py
from pydantic import BaseModel
from decimal import Decimal


class ScoreUpdate(BaseModel):
    """教师更新评分"""
    final_score: Decimal
    teacher_comment: str | None = None


class ScorePublic(BaseModel):
    """评分响应"""
    submission_id: int
    question_id: int
    student_id: int

    # ML scoring
    ml_score: Decimal | None = None
    ml_label: str | None = None
    ml_confidence: Decimal | None = None

    # Teacher grading
    final_score: Decimal | None = None
    teacher_comment: str | None = None

    status: str

    model_config = {"from_attributes": True}