# app/core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://ee547_user:ee547_pass@localhost:5432/ee547_db"

    class Config:
        env_file = ".env"

settings = Settings()
