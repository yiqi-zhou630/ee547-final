# app/db/deps.py
from typing import Generator
from app.db.session import SessionLocal

def get_db() -> Generator:
    db = SessionLocal() # 创建一个新的数据库会话
    try:
        yield db    # 提供数据库会话“交给”请求使用
    finally:
        db.close()  # 请求结束时自动关闭数据库会话
