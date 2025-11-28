# app/schemas/question.py
from pydantic import BaseModel
from datetime import datetime


class QuestionBase(BaseModel):
    title: str | None = None
    question_text: str
    reference_answer: str
    max_score: int = 5
    topic: str | None = None


class QuestionCreate(QuestionBase):
    pass


class QuestionUpdate(BaseModel):
    title: str | None = None
    question_text: str | None = None
    reference_answer: str | None = None
    max_score: int | None = None
    topic: str | None = None


class QuestionPublic(QuestionBase):
    id: int
    teacher_id: int | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}