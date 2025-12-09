# app/services/question_service.py
from typing import List, Optional

from sqlalchemy.orm import Session

from app.models.question import Question
from app.models.user import User
from app.schemas.question import QuestionCreate, QuestionUpdate


def create_question(
    db: Session,
    *,
    teacher: User,
    obj_in: QuestionCreate,
) -> Question:
    """
    teacher creat question
    """
    db_obj = Question(
        teacher_id=teacher.id,
        title=obj_in.title,
        question_text=obj_in.question_text,
        reference_answer=obj_in.reference_answer,
        max_score=obj_in.max_score,
        topic=obj_in.topic,
    )
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj


def get_question(db: Session, question_id: int) -> Optional[Question]:
    return db.query(Question).get(question_id)


def list_questions_for_teacher(
    db: Session,
    *,
    teacher: User,
    skip: int = 0,
    limit: int = 100,
) -> List[Question]:
    """
    question list created by some teacher
    """
    return (
        db.query(Question)
        .filter(Question.teacher_id == teacher.id)
        .order_by(Question.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def list_all_questions(
    db: Session,
    *,
    skip: int = 0,
    limit: int = 100,
) -> List[Question]:
    """
    student: list all question
    """
    return (
        db.query(Question)
        .order_by(Question.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def update_question(
    db: Session,
    *,
    db_obj: Question,
    obj_in: QuestionUpdate,
) -> Question:
    """
    teacher update question
    """
    update_data = obj_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_obj, field, value)
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj


def delete_question(db: Session, *, db_obj: Question) -> None:
    db.delete(db_obj)
    db.commit()
