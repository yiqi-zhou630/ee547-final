# app/api/v1/endpoints/health.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db

router = APIRouter(tags=["health"])


@router.get("/live")
def liveness_probe():
    return {"status": "ok"}


@router.get("/db")
def db_health(db: Session = Depends(get_db)):
    db.execute("SELECT 1")
    return {"status": "ok"}
