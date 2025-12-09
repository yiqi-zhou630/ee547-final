# app/services/scoring_service.py
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional, List

from sqlalchemy.orm import Session

from app.models.submission import Submission
from app.models.question import Question
from app.models.user import User

from app.schemas.score import ScoreUpdate
from app.services.ml_client import score_answer


class ScoringError(Exception):
    pass


def _get_submission_and_question(
    db: Session,
    submission_id: int,
) -> tuple[Submission, Question]:
    submission: Optional[Submission] = db.query(Submission).get(submission_id)
    if submission is None:
        raise ScoringError(f"submission {submission_id} not found")

    question: Optional[Question] = db.query(Question).get(submission.question_id)
    if question is None:
        raise ScoringError(
            f"question {submission.question_id} for submission {submission_id} not found"
        )

    return submission, question


def run_ml_scoring_for_submission(
    db: Session,
    submission_id: int,
    *,
    model_version: str | None = None,
) -> Submission:
    """
    worker 调用：对单个 submission 跑 ML 打分。

    - 调用 ml_client.score_answer
    - 写回 Submission.ml_* 字段
    - status: 'pending_ml' -> 'ml_scored'
    """
    submission, question = _get_submission_and_question(db, submission_id)

    # 如果已经 graded，可以根据业务决定要不要覆盖；这里简单允许重复打分覆盖 ML 字段
    question_text = question.question_text
    ref_answer = question.reference_answer
    student_answer = submission.answer_text

    # 在 ml_client 中定义：
    # def score_answer(...) -> tuple[str, Decimal | float, Decimal | float, str]:
    ml_label, ml_score, ml_confidence, ml_explanation = score_answer(
        question_text=question_text,
        ref_answer=ref_answer,
        student_answer=student_answer,
    )

    ml_score_dec = Decimal(str(ml_score)) if ml_score is not None else None
    ml_conf_dec = (
        Decimal(str(ml_confidence)) if ml_confidence is not None else None
    )

    submission.ml_label = ml_label
    submission.ml_score = ml_score_dec
    submission.ml_confidence = ml_conf_dec
    submission.ml_explanation = ml_explanation
    submission.model_version = model_version
    submission.ml_scored_at = datetime.now(timezone.utc)

    if submission.status == "pending_ml":
        submission.status = "ml_scored"

    db.add(submission)
    db.commit()
    db.refresh(submission)
    return submission


def teacher_override_score(
    db: Session,
    *,
    submission: Submission,
    teacher: User,
    score_in: ScoreUpdate,
) -> Submission:

    submission.final_score = score_in.final_score
    submission.teacher_comment = score_in.teacher_comment
    submission.teacher_id = teacher.id
    submission.final_label = None
    submission.graded_at = datetime.now(timezone.utc)
    submission.status = "graded"

    db.add(submission)
    db.commit()
    db.refresh(submission)
    return submission


def list_need_grading_for_teacher(
    db: Session,
    *,
    teacher: User,
    skip: int = 0,
    limit: int = 100,
) -> List[Submission]:
    """
    老师查看“需要人工检查”的作答：
      - 该老师名下的题目
      - 且 submission.status = 'ml_scored'
    """
    # 先查出该老师题目的 id 列表
    q_ids_subquery = (
        db.query(Question.id)
        .filter(Question.teacher_id == teacher.id)
        .subquery()
    )

    return (
        db.query(Submission)
        .filter(
            Submission.question_id.in_(q_ids_subquery),
            Submission.status == "ml_scored",
        )
        .order_by(Submission.created_at.asc())
        .offset(skip)
        .limit(limit)
        .all()
    )
