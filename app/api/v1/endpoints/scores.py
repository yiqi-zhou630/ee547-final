# app/api/v1/endpoints/scores.py
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.submission import Submission
from app.models.question import Question
from app.models.user import User
from app.schemas.score import ScoreUpdate, ScorePublic
from app.services import scoring_service, submission_service
from app.core.security import get_current_teacher

router = APIRouter(tags=["scores"])


def _submission_to_score_public(sub: Submission) -> ScorePublic:
    return ScorePublic(
        submission_id=sub.id,
        question_id=sub.question_id,
        student_id=sub.student_id,
        ml_score=sub.ml_score,
        ml_label=sub.ml_label,
        ml_confidence=sub.ml_confidence,
        final_score=sub.final_score,
        teacher_comment=sub.teacher_comment,
        status=sub.status,
    )


@router.get("/pending", response_model=List[ScorePublic])
def list_pending_scores(
    db: Session = Depends(get_db),
    current_teacher: User = Depends(get_current_teacher),
    skip: int = 0,
    limit: int = 100,
):
    subs = scoring_service.list_need_grading_for_teacher(
        db, teacher=current_teacher, skip=skip, limit=limit
    )
    return [_submission_to_score_public(sub) for sub in subs]


@router.get("/{submission_id}", response_model=ScorePublic)
def get_score(
    submission_id: int,
    db: Session = Depends(get_db),
    current_teacher: User = Depends(get_current_teacher),
):
    sub = submission_service.get_submission(db, submission_id)
    if not sub:
        raise HTTPException(status_code=404, detail="Submission not found")

    # 权限检查：submission 对应的 question 必须属于当前老师
    question: Question = (
        db.query(Question).filter(Question.id == sub.question_id).first()
    )
    if question is None or question.teacher_id != current_teacher.id:
        raise HTTPException(status_code=403, detail="Not allowed to view this score")

    return _submission_to_score_public(sub)


@router.put("/{submission_id}", response_model=ScorePublic)
def update_score(
    submission_id: int,
    score_in: ScoreUpdate,
    db: Session = Depends(get_db),
    current_teacher: User = Depends(get_current_teacher),
):
    """
    老师更新/确认评分：
      - 写 final_score / teacher_comment
      - status -> 'graded'
    """
    sub = submission_service.get_submission(db, submission_id)
    if not sub:
        raise HTTPException(status_code=404, detail="Submission not found")

    # 权限：必须是自己的题目
    question: Question = (
        db.query(Question).filter(Question.id == sub.question_id).first()
    )
    if question is None or question.teacher_id != current_teacher.id:
        raise HTTPException(status_code=403, detail="Not allowed to grade this submission")

    updated = scoring_service.teacher_override_score(
        db, submission=sub, teacher=current_teacher, score_in=score_in
    )
    return _submission_to_score_public(updated)
