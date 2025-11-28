# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.db.base import Base
from app.db.session import engine
from app.api.v1.endpoints import auth, users, questions, submissions, scores, health

app = FastAPI(title=settings.PROJECT_NAME)

# CORS 配置（允许前端跨域请求）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该指定具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    """应用启动时创建数据库表"""
    Base.metadata.create_all(bind=engine)


# 注册所有路由
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(users.router, prefix="/api/v1/users", tags=["Users"])
app.include_router(questions.router, prefix="/api/v1/questions", tags=["Questions"])
app.include_router(submissions.router, prefix="/api/v1/submissions", tags=["Submissions"])
app.include_router(scores.router, prefix="/api/v1/scores", tags=["Scores"])
app.include_router(health.router, prefix="/api/v1/health", tags=["Health"])


@app.get("/")
def root():
    """根路径"""
    return {
        "message": "Welcome to EE547 Grading System API",
        "docs": "/docs",
        "redoc": "/redoc",
    }

