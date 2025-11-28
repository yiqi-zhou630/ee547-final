# app/api/v1/endpoints/scores.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime

from app.db.session import get_db
from app.models.submission import Submission
from app.models.user import User
from app.schemas.score import ScoreUpdate, ScorePublic
from app.core.security import get_current_teacher

router = APIRouter()


@router.get("/", response_model=list[ScorePublic])
def list_scores(
    skip: int = 0,
    limit: int = 100,
    question_id: int | None = None,
    student_id: int | None = None,
    status_filter: str | None = None,
    db: Session = Depends(get_db),
    current_teacher: User = Depends(get_current_teacher),
):
    """获取评分列表（仅教师）"""
    query = db.query(Submission)

    if question_id:
        query = query.filter(Submission.question_id == question_id)

    if student_id:
        query = query.filter(Submission.student_id == student_id)

    if status_filter:
        query = query.filter(Submission.status == status_filter)

    submissions = query.offset(skip).limit(limit).all()

    # 转换为 ScorePublic
    scores = [
        ScorePublic(
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
        for sub in submissions
    ]

    return scores


@router.get("/{submission_id}", response_model=ScorePublic)
def get_score(
    submission_id: int,
    db: Session = Depends(get_db),
    current_teacher: User = Depends(get_current_teacher),
):
    """获取单个评分详情（仅教师）"""
    submission = db.query(Submission).filter(Submission.id == submission_id).first()
    if not submission:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Submission not found",
        )

    return ScorePublic(
        submission_id=submission.id,
        question_id=submission.question_id,
        student_id=submission.student_id,
        ml_score=submission.ml_score,
        ml_label=submission.ml_label,
        ml_confidence=submission.ml_confidence,
        final_score=submission.final_score,
        teacher_comment=submission.teacher_comment,
        status=submission.status,
    )


@router.put("/{submission_id}", response_model=ScorePublic)
def update_score(
    submission_id: int,
    payload: ScoreUpdate,
    db: Session = Depends(get_db),
    current_teacher: User = Depends(get_current_teacher),
):
    """更新评分（教师人工评分）"""
    submission = db.query(Submission).filter(Submission.id == submission_id).first()
    if not submission:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Submission not found",
        )

    # 更新教师评分
    submission.final_score = payload.final_score
    submission.teacher_comment = payload.teacher_comment
    submission.teacher_id = current_teacher.id
    submission.graded_at = datetime.utcnow()
    submission.status = "graded"

    db.commit()
    db.refresh(submission)

    return ScorePublic(
        submission_id=submission.id,
        question_id=submission.question_id,
        student_id=submission.student_id,
        ml_score=submission.ml_score,
        ml_label=submission.ml_label,
        ml_confidence=submission.ml_confidence,
        final_score=submission.final_score,
        teacher_comment=submission.teacher_comment,
        status=submission.status,
    )


@router.post("/{submission_id}/confirm", response_model=ScorePublic)
def confirm_ml_score(
    submission_id: int,
    db: Session = Depends(get_db),
    current_teacher: User = Depends(get_current_teacher),
):
    """确认 ML 评分（将 ML 评分作为最终分数）"""
    submission = db.query(Submission).filter(Submission.id == submission_id).first()
    if not submission:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Submission not found",
        )

    if submission.ml_score is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No ML score available to confirm",
        )

    # 使用 ML 评分作为最终分数
    submission.final_score = submission.ml_score
    submission.teacher_id = current_teacher.id
    submission.graded_at = datetime.utcnow()
    submission.status = "graded"

    db.commit()
    db.refresh(submission)

    return ScorePublic(
        submission_id=submission.id,
        question_id=submission.question_id,
        student_id=submission.student_id,
        ml_score=submission.ml_score,
        ml_label=submission.ml_label,
        ml_confidence=submission.ml_confidence,
        final_score=submission.final_score,
        teacher_comment=submission.teacher_comment,
        status=submission.status,
    )


@router.get("/pending/count")
def get_pending_count(
    db: Session = Depends(get_db),
    current_teacher: User = Depends(get_current_teacher),
):
    """获取待处理的评分数量"""
    pending_count = db.query(Submission).filter(
        Submission.status == "ml_scored"
    ).count()

    return {"pending_count": pending_count}
