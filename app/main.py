# app/main.py
from fastapi import FastAPI

from app.core.config import settings
from app.db.base import Base
from app.db.session import engine
from app.api.v1.endpoints import auth, users, questions, submissions, scores, health
from app import models  # noqa

app = FastAPI(title=settings.PROJECT_NAME)


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)


app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
# 其它 router 也类似：
# app.include_router(users.router, prefix="/api/v1/users", tags=["users"])
