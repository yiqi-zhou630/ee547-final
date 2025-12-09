# app/workers/tasks.py

from app.db.session import SessionLocal
from app.services.scoring_service import (
    run_ml_scoring_for_submission,
    ScoringError,
)


def scoring_task(submission_id: int) -> None:
    db = SessionLocal()
    try:
        print(f"[worker] scoring_task running for submission {submission_id}")
        run_ml_scoring_for_submission(db, submission_id, model_version="demo-v1")
        print(f"[worker] scoring_task finished for submission {submission_id}")
    except ScoringError as e:
        print(f"[worker] scoring_task FAILED for submission {submission_id}: {e}")
    finally:
        db.close()
