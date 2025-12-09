# app/api/v1/endpoints/submissions.py
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.submission import Submission
from app.models.user import User
from app.schemas.submission import (
    SubmissionCreate,
    SubmissionUpdate,
    SubmissionPublic,
    SubmissionDetail,
)
from app.services import submission_service, scoring_service
from app.core.security import get_current_student, get_current_teacher, get_current_user

router = APIRouter(prefix="/submissions", tags=["submissions"])


@router.post("/", response_model=SubmissionPublic, status_code=status.HTTP_201_CREATED)
def create_submission(
    obj_in: SubmissionCreate,
    db: Session = Depends(get_db),
    current_student: User = Depends(get_current_student),
):
    """
    学生提交答案；会创建一条 submission 并入队 ML 打分任务。
    """
    sub = submission_service.create_submission_and_enqueue_task(
        db, student=current_student, obj_in=obj_in
    )
    return sub


@router.get("/me", response_model=List[SubmissionPublic])
def list_my_submissions(
    db: Session = Depends(get_db),
    current_student: User = Depends(get_current_student),
    skip: int = 0,
    limit: int = 100,
):
    """
    学生查看自己的所有提交。
    """
    subs = submission_service.list_submissions_for_student(
        db, student=current_student, skip=skip, limit=limit
    )
    return subs


@router.get("/{submission_id}", response_model=SubmissionPublic)
def get_submission_for_student(
    submission_id: int,
    db: Session = Depends(get_db),
    current_student: User = Depends(get_current_student),
):
    """
    学生查看自己的单条提交（不含 ML 细节）。
    """
    sub = submission_service.get_submission(db, submission_id)
    if not sub or sub.student_id != current_student.id:
        raise HTTPException(status_code=404, detail="Submission not found")
    return sub


@router.get("/teacher/{submission_id}", response_model=SubmissionDetail)
def get_submission_for_teacher(
    submission_id: int,
    db: Session = Depends(get_db),
    current_teacher: User = Depends(get_current_teacher),
):
    """
    老师查看某条提交的详细信息（包含 ML 结果）。
    """
    sub = submission_service.get_submission(db, submission_id)
    if not sub:
        raise HTTPException(status_code=404, detail="Submission not found")

    # 权限：这条 submission 对应的题目必须是当前老师的
    question = scoring_service._get_submission_and_question(db, submission_id)[1]
    if question.teacher_id != current_teacher.id:
        raise HTTPException(status_code=403, detail="Not allowed to view this submission")

    return sub


@router.put("/{submission_id}", response_model=SubmissionPublic)
def update_submission_answer(
    submission_id: int,
    obj_in: SubmissionUpdate,
    db: Session = Depends(get_db),
    current_student: User = Depends(get_current_student),
):
    sub = submission_service.get_submission(db, submission_id)
    if not sub or sub.student_id != current_student.id:
        raise HTTPException(status_code=404, detail="Submission not found")

    if sub.status != "pending_ml":
        raise HTTPException(
            status_code=400, detail="Cannot edit submission after ML scoring started"
        )

    sub = submission_service.update_submission_answer(db, db_obj=sub, obj_in=obj_in)
    return sub
