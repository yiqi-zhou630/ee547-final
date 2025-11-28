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
    id: int
    student_id: int
    status: str  # pending_ml / ml_scored / graded

    # ML scoring
    ml_label: str | None = None
    ml_score: Decimal | None = None
    ml_confidence: Decimal | None = None
    model_version: str | None = None
    ml_explanation: str | None = None
    ml_scored_at: datetime | None = None

    # Teacher grading
    teacher_id: int | None = None
    final_score: Decimal | None = None
    teacher_comment: str | None = None
    graded_at: datetime | None = None

    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}