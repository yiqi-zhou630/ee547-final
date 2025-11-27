# app/db/base.py
from sqlalchemy.orm import declarative_base

Base = declarative_base()

from app.models.user import User      # noqa
from app.models.question import Question  # noqa
from app.models.submission import Submission  # noqa
