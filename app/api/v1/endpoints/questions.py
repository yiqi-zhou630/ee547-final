from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.deps import get_db
from app.models.question import Question

router = APIRouter()

@router.get("/questions")
def list_questions(db: Session = Depends(get_db)):
    questions = db.query(Question).all()
    return questions
