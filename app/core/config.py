# app/core/config.py
from datetime import timedelta
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Short Answer Grading Support System"

    DATABASE_URL: str = (
        "postgresql+psycopg2://ee547_user:20020630Qq@localhost:5432/ee547_db"
    )

    SECRET_KEY: str = "change-me-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
