# app/api/v1/endpoints/questions.py
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.question import Question
from app.models.user import User
from app.schemas.question import (
    QuestionCreate,
    QuestionUpdate,
    QuestionPublic,
)
from app.services import question_service
from app.core.security import get_current_teacher, get_current_user

router = APIRouter(prefix="/questions", tags=["questions"])


@router.post("/", response_model=QuestionPublic, status_code=status.HTTP_201_CREATED)
def create_question(
    obj_in: QuestionCreate,
    db: Session = Depends(get_db),
    current_teacher: User = Depends(get_current_teacher),
):
    """
    老师创建题目。
    """
    q = question_service.create_question(db, teacher=current_teacher, obj_in=obj_in)
    return q


@router.get("/", response_model=List[QuestionPublic])
def list_questions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),  # 学生和老师都能看
    skip: int = 0,
    limit: int = 100,
):
    """
    列出所有题目（学生/老师通用）。
    """
    qs = question_service.list_all_questions(db, skip=skip, limit=limit)
    return qs


@router.get("/mine", response_model=List[QuestionPublic])
def list_my_questions(
    db: Session = Depends(get_db),
    current_teacher: User = Depends(get_current_teacher),
    skip: int = 0,
    limit: int = 100,
):
    """
    老师查看自己创建的题目列表。
    """
    qs = question_service.list_questions_for_teacher(
        db, teacher=current_teacher, skip=skip, limit=limit
    )
    return qs


@router.get("/{question_id}", response_model=QuestionPublic)
def get_question(
    question_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),  # 任意登录用户可看
):
    q = question_service.get_question(db, question_id)
    if not q:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Question not found",
        )
    return q


@router.put("/{question_id}", response_model=QuestionPublic)
def update_question(
    question_id: int,
    obj_in: QuestionUpdate,
    db: Session = Depends(get_db),
    current_teacher: User = Depends(get_current_teacher),
):
    """
    老师更新自己创建的题目。
    """
    q = question_service.get_question(db, question_id)
    if not q:
        raise HTTPException(status_code=404, detail="Question not found")

    if q.teacher_id != current_teacher.id:
        raise HTTPException(status_code=403, detail="Not allowed to edit this question")

    q = question_service.update_question(db, db_obj=q, obj_in=obj_in)
    return q


@router.delete("/{question_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_question(
    question_id: int,
    db: Session = Depends(get_db),
    current_teacher: User = Depends(get_current_teacher),
):
    q = question_service.get_question(db, question_id)
    if not q:
        raise HTTPException(status_code=404, detail="Question not found")

    if q.teacher_id != current_teacher.id:
        raise HTTPException(status_code=403, detail="Not allowed to delete this question")

    question_service.delete_question(db, db_obj=q)
    return None
