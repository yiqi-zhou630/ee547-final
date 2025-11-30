# app/schemas/submission.py
from pydantic import BaseModel
from datetime import datetime
from decimal import Decimal


class SubmissionBase(BaseModel):
    question_id: int
    answer_text: str


class SubmissionCreate(SubmissionBase):
    pass


class SubmissionUpdate(BaseModel):
    answer_text: str | None = None


class SubmissionPublic(SubmissionBase):
    """学生可见的提交信息（隐藏 ML 评分细节）"""
    id: int
    student_id: int
    status: str  # pending_ml / ml_scored / graded

    # Teacher grading (学生只能看到最终结果)
    final_score: Decimal | None = None
    teacher_comment: str | None = None
    graded_at: datetime | None = None

    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class SubmissionDetail(SubmissionPublic):
    """教师可见的完整提交信息（包含 ML 评分）"""
    # ML scoring (只有教师能看到)
    ml_label: str | None = None
    ml_score: Decimal | None = None
    ml_confidence: Decimal | None = None
    model_version: str | None = None
    ml_explanation: str | None = None
    ml_scored_at: datetime | None = None
    teacher_id: int | None = None