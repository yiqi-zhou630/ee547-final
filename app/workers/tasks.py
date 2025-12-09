"""
Scoring Tasks for Worker
These tasks are executed by RQ workers to score submissions asynchronously
"""

import logging
from app.db.session import SessionLocal
from app.services.scoring_service import run_ml_scoring_for_submission, ScoringError
from app.core.config import settings

logger = logging.getLogger(__name__)


def scoring_task(submission_id: int) -> dict:
    """
    Worker task to score a submission using ML model.
    
    This task:
    1. Creates a database session
    2. Calls scoring_service to run ML model
    3. Saves results to database
    4. Returns result summary
    
    Args:
        submission_id: ID of submission to score
    
    Returns:
        Dictionary with scoring results
    
    Note:
        This function is called by RQ workers.
        It will be enqueued by enqueue_scoring_task() in queue.py
    """
    db = SessionLocal()
    try:
        logger.info(f"Starting scoring task for submission {submission_id}")
        
        # Run ML scoring service
        submission = run_ml_scoring_for_submission(
            db=db,
            submission_id=submission_id,
            model_version=settings.ML_MODEL_VERSION
        )
        
        # Prepare result
        result = {
            "status": "success",
            "submission_id": submission.id,
            "ml_label": submission.ml_label,
            "ml_score": float(submission.ml_score) if submission.ml_score else None,
            "ml_confidence": float(submission.ml_confidence) if submission.ml_confidence else None,
            "ml_explanation": submission.ml_explanation,
            "model_version": submission.model_version,
            "message": f"Successfully scored submission {submission_id}"
        }
        
        logger.info(
            f"Completed scoring task for submission {submission_id}: "
            f"label={submission.ml_label}, score={submission.ml_score}"
        )
        
        return result
        
    except ScoringError as e:
        logger.error(f"Scoring failed for submission {submission_id}: {e}")
        return {
            "status": "error",
            "submission_id": submission_id,
            "error": str(e),
            "message": f"Scoring failed for submission {submission_id}"
        }
    
    except Exception as e:
        logger.error(
            f"Unexpected error during scoring task for submission {submission_id}: {e}",
            exc_info=True
        )
        return {
            "status": "error",
            "submission_id": submission_id,
            "error": str(e),
            "message": f"Unexpected error during scoring"
        }
    
    finally:
        db.close()
