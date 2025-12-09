# app/services/submission_service.py
from typing import List, Optional

from sqlalchemy.orm import Session

from app.models.submission import Submission
from app.models.user import User
from app.schemas.submission import SubmissionCreate, SubmissionUpdate

from app.workers.queue import enqueue_scoring_task


def create_submission_and_enqueue_task(
    db: Session,
    *,
    student: User,
    obj_in: SubmissionCreate,
) -> Submission:
    """
    学生提交答案 + 创建 ML打分任务
    status 初始为 'pending_ml'
    """
    submission = Submission(
        question_id=obj_in.question_id,
        student_id=student.id,
        answer_text=obj_in.answer_text,
        status="pending_ml",
    )

    db.add(submission)
    db.commit()
    db.refresh(submission)

    # 入队，让 worker 去跑 ml scoring
    enqueue_scoring_task(submission.id)

    return submission


def update_submission_answer(
    db: Session,
    *,
    db_obj: Submission,
    obj_in: SubmissionUpdate,
) -> Submission:
    """
    如果允许学生在 ML 打分前修改答案
    （通常只允许 status == 'pending_ml' 时修改）
    """
    if db_obj.status != "pending_ml":
        return db_obj

    update_data = obj_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_obj, field, value)
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj


def get_submission(db: Session, submission_id: int) -> Optional[Submission]:
    return db.query(Submission).get(submission_id)


def list_submissions_for_student(
    db: Session,
    *,
    student: User,
    skip: int = 0,
    limit: int = 100,
) -> List[Submission]:
    """
    学生查看自己的所有提交
    """
    return (
        db.query(Submission)
        .filter(Submission.student_id == student.id)
        .order_by(Submission.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def list_submissions_for_question(
    db: Session,
    *,
    question_id: int,
    skip: int = 0,
    limit: int = 100,
) -> List[Submission]:
    """
    老师按题目查看所有学生的提交
    """
    return (
        db.query(Submission)
        .filter(Submission.question_id == question_id)
        .order_by(Submission.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def list_pending_ml_submissions(
    db: Session,
    *,
    skip: int = 0,
    limit: int = 100,
) -> List[Submission]:
    return (
        db.query(Submission)
        .filter(Submission.status == "pending_ml")
        .order_by(Submission.created_at.asc())
        .offset(skip)
        .limit(limit)
        .all()
    )
