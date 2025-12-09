"""
Scoring Service
Orchestrates ML model scoring and saves results to database
"""

from datetime import datetime, timezone
from decimal import Decimal
import logging
from typing import List
from sqlalchemy.orm import Session
from app.models.user import User
from app.models.submission import Submission
from app.services.ml_client import score_answer
from app.core.config import settings
from app.schemas.score import ScoreUpdate
from app.models.question import Question
logger = logging.getLogger(__name__)


class ScoringError(Exception):
    """Raised when scoring fails"""
    pass


def run_ml_scoring_for_submission(
    db: Session,
    submission_id: int,
    model_version: str = None
) -> Submission:
    """
    Run ML scoring for a submission and save results to database.
    
    This function:
    1. Retrieves the submission and associated question
    2. Calls the ML model to score the answer
    3. Saves scoring results to the database
    4. Updates submission status to 'ml_scored'
    
    Args:
        db: Database session
        submission_id: ID of submission to score
        model_version: Model version identifier (defaults to settings.ML_MODEL_VERSION)
    
    Returns:
        Updated Submission object with ML scoring results
    
    Raises:
        ScoringError: If submission not found or scoring fails
    """
    if model_version is None:
        model_version = settings.ML_MODEL_VERSION
    
    try:
        # Retrieve submission and related data
        submission = db.query(Submission).filter(
            Submission.id == submission_id
        ).first()
        
        if not submission:
            raise ScoringError(f"Submission {submission_id} not found")
        
        # Retrieve question with reference answer
        from app.models.question import Question
        question = db.query(Question).filter(
            Question.id == submission.question_id
        ).first()
        
        if not question:
            raise ScoringError(
                f"Question {submission.question_id} not found for submission {submission_id}"
            )
        
        logger.info(
            f"Starting ML scoring for submission {submission_id}: "
            f"question_id={submission.question_id}, student_id={submission.student_id}"
        )
        
        # Call ML model to score the answer
        label, score, confidence, explanation = score_answer(
            question_text=question.question_text,
            ref_answer=question.reference_answer,
            student_answer=submission.answer_text
        )
        
        # Update submission with ML scoring results
        submission.ml_label = label
        submission.ml_score = Decimal(str(round(score, 1)))
        submission.ml_confidence = Decimal(str(round(confidence, 3)))
        submission.ml_explanation = explanation
        submission.model_version = model_version
        submission.ml_scored_at = datetime.now(timezone.utc)
        submission.status = "ml_scored"
        
        # Save to database
        db.add(submission)
        db.commit()
        db.refresh(submission)
        
        logger.info(
            f"Completed ML scoring for submission {submission_id}: "
            f"label={label}, score={score}, confidence={confidence:.3f}"
        )
        
        return submission
        
    except ScoringError:
        raise
    except Exception as e:
        logger.error(
            f"Error during ML scoring for submission {submission_id}: {e}",
            exc_info=True
        )
        raise ScoringError(f"ML scoring failed: {str(e)}")

def teacher_override_score(
    db: Session,
    *,
    submission: Submission,
    teacher: User,
    score_in: ScoreUpdate,
) -> Submission:
    """
    老师给最终分 / 修改分数。

    对应 schemas.score.ScoreUpdate:
      - final_score: Decimal
      - teacher_comment: str | None
    """
    submission.final_score = score_in.final_score
    submission.teacher_comment = score_in.teacher_comment
    submission.teacher_id = teacher.id
    submission.final_label = None  # 如果以后需要 label，可以在 ScoreUpdate 里加
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