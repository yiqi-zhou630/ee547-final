# app/core/config.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Short Answer Grading Support System"

    # Database
    # 使用 SQLite 进行本地开发（无需 PostgreSQL）
    DATABASE_URL: str = "sqlite:///./ee547.db"

    # JWT Authentication
    SECRET_KEY: str = "change-me-in-production-use-random-string"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    # AWS S3
    AWS_ACCESS_KEY_ID: str | None = None
    AWS_SECRET_ACCESS_KEY: str | None = None
    AWS_REGION: str = "us-west-2"
    S3_BUCKET_NAME: str | None = None

    # AWS SQS
    SQS_QUEUE_URL: str | None = None

    # Redis (for local queue)
    REDIS_URL: str = "redis://localhost:6379/0"

    # ML Model
    ML_MODEL_PATH: str = "./model_training/outputs/final_model"
    ML_USE_FEATURES: bool = False
    ML_MODEL_VERSION: str = "1.0.0"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
