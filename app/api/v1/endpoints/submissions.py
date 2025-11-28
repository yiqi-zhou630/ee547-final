# app/api/v1/endpoints/submissions.py
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.submission import Submission
from app.models.question import Question
from app.models.user import User
from app.schemas.submission import SubmissionCreate, SubmissionUpdate, SubmissionPublic
from app.core.security import get_current_user, get_current_student, get_current_teacher

router = APIRouter()


@router.get("/", response_model=list[SubmissionPublic])
def list_submissions(
    skip: int = 0,
    limit: int = 100,
    question_id: int | None = None,
    student_id: int | None = None,
    status_filter: str | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """获取提交列表"""
    query = db.query(Submission)

    # 学生只能看自己的
    if current_user.role == "student":
        query = query.filter(Submission.student_id == current_user.id)
    else:
        # 教师可以筛选
        if student_id:
            query = query.filter(Submission.student_id == student_id)

    if question_id:
        query = query.filter(Submission.question_id == question_id)

    if status_filter:
        query = query.filter(Submission.status == status_filter)

    submissions = query.offset(skip).limit(limit).all()
    return submissions


@router.get("/{submission_id}", response_model=SubmissionPublic)
def get_submission(
    submission_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """获取单个提交"""
    submission = db.query(Submission).filter(Submission.id == submission_id).first()
    if not submission:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Submission not found",
        )

    # 学生只能看自己的
    if current_user.role == "student" and submission.student_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only view your own submissions",
        )

    return submission


@router.post("/", response_model=SubmissionPublic, status_code=status.HTTP_201_CREATED)
def create_submission(
    payload: SubmissionCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_student: User = Depends(get_current_student),
):
    """创建提交（仅学生）"""
    # 检查题目是否存在
    question = db.query(Question).filter(Question.id == payload.question_id).first()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Question not found",
        )

    submission = Submission(
        **payload.model_dump(),
        student_id=current_student.id,
        status="pending_ml",
    )

    db.add(submission)
    db.commit()
    db.refresh(submission)

    # TODO: 加入评分队列
    # background_tasks.add_task(enqueue_scoring_task, submission.id)

    return submission


@router.put("/{submission_id}", response_model=SubmissionPublic)
def update_submission(
    submission_id: int,
    payload: SubmissionUpdate,
    db: Session = Depends(get_db),
    current_student: User = Depends(get_current_student),
):
    """更新提交（仅学生，且只能更新 pending_ml 状态的）"""
    submission = db.query(Submission).filter(Submission.id == submission_id).first()
    if not submission:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Submission not found",
        )

    if submission.student_id != current_student.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only update your own submissions",
        )

    if submission.status != "pending_ml":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot update submission after ML scoring",
        )

    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(submission, key, value)

    db.commit()
    db.refresh(submission)
    return submission


@router.delete("/{submission_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_submission(
    submission_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """删除提交"""
    submission = db.query(Submission).filter(Submission.id == submission_id).first()
    if not submission:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Submission not found",
        )

    # 学生只能删除自己的
    if current_user.role == "student" and submission.student_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only delete your own submissions",
        )

    db.delete(submission)
    db.commit()
    return None
