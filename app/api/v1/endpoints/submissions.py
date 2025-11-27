from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.deps import get_db
from app.models.submission import Submission

router = APIRouter()

@router.post("/submissions")
def create_submission(
    question_id: int,
    student_id: int,
    answer_text: str,
    db: Session = Depends(get_db)
):
    submission = Submission(
        question_id=question_id,
        student_id=student_id,
        answer_text=answer_text,
        status="pending_ml"
    )

    db.add(submission)
    db.commit()
    db.refresh(submission)

    return submission
