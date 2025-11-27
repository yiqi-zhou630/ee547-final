from pathlib import Path
import os

# 以当前目录为项目根目录：即 ee547-final/
BASE_DIR = Path(".")

# 需要创建的目录
DIRS = [
    BASE_DIR / "app" / "api" / "v1" / "endpoints",
    BASE_DIR / "app" / "core",
    BASE_DIR / "app" / "db",
    BASE_DIR / "app" / "models",
    BASE_DIR / "app" / "schemas",
    BASE_DIR / "app" / "services",
    BASE_DIR / "app" / "workers",
    BASE_DIR / "alembic" / "versions",
    BASE_DIR / "tests",
]

# 需要创建的文件
FILES = [
    # __init__.py
    BASE_DIR / "app" / "__init__.py",
    BASE_DIR / "app" / "api" / "__init__.py",
    BASE_DIR / "app" / "api" / "v1" / "__init__.py",
    BASE_DIR / "app" / "core" / "__init__.py",
    BASE_DIR / "app" / "db" / "__init__.py",
    BASE_DIR / "app" / "models" / "__init__.py",
    BASE_DIR / "app" / "schemas" / "__init__.py",
    BASE_DIR / "app" / "services" / "__init__.py",
    BASE_DIR / "app" / "workers" / "__init__.py",
    BASE_DIR / "tests" / "__init__.py",

    # endpoints
    BASE_DIR / "app" / "api" / "v1" / "endpoints" / "auth.py",
    BASE_DIR / "app" / "api" / "v1" / "endpoints" / "users.py",
    BASE_DIR / "app" / "api" / "v1" / "endpoints" / "questions.py",
    BASE_DIR / "app" / "api" / "v1" / "endpoints" / "submissions.py",
    BASE_DIR / "app" / "api" / "v1" / "endpoints" / "scores.py",
    BASE_DIR / "app" / "api" / "v1" / "endpoints" / "health.py",

    # core
    BASE_DIR / "app" / "core" / "config.py",
    BASE_DIR / "app" / "core" / "security.py",
    BASE_DIR / "app" / "core" / "logging_config.py",

    # db
    BASE_DIR / "app" / "db" / "base.py",
    BASE_DIR / "app" / "db" / "session.py",
    BASE_DIR / "app" / "db" / "init_db.py",

    # models
    BASE_DIR / "app" / "models" / "user.py",
    BASE_DIR / "app" / "models" / "question.py",
    BASE_DIR / "app" / "models" / "submission.py",
    BASE_DIR / "app" / "models" / "score.py",

    # schemas
    BASE_DIR / "app" / "schemas" / "user.py",
    BASE_DIR / "app" / "schemas" / "auth.py",
    BASE_DIR / "app" / "schemas" / "question.py",
    BASE_DIR / "app" / "schemas" / "submission.py",
    BASE_DIR / "app" / "schemas" / "score.py",

    # services
    BASE_DIR / "app" / "services" / "question_service.py",
    BASE_DIR / "app" / "services" / "submission_service.py",
    BASE_DIR / "app" / "services" / "scoring_service.py",
    BASE_DIR / "app" / "services" / "ml_client.py",
    BASE_DIR / "app" / "services" / "s3_client.py",

    # workers
    BASE_DIR / "app" / "workers" / "queue.py",
    BASE_DIR / "app" / "workers" / "tasks.py",
    BASE_DIR / "app" / "workers" / "worker_main.py",

    # main
    BASE_DIR / "app" / "main.py",

    # alembic
    BASE_DIR / "alembic" / "env.py",
    BASE_DIR / "alembic" / "script.py.mako",

    # tests
    BASE_DIR / "tests" / "test_questions.py",
    BASE_DIR / "tests" / "test_submissions.py",
    BASE_DIR / "tests" / "test_scoring.py",

    # 根目录文件
    BASE_DIR / "requirements.txt",
    BASE_DIR / ".env.example",
    BASE_DIR / "docker-compose.yml",
    BASE_DIR / "Dockerfile",
    BASE_DIR / "README.md",
]


def main():
    # 创建目录
    for d in DIRS:
        os.makedirs(d, exist_ok=True)

    # 创建文件
    for f in FILES:
        f.parent.mkdir(parents=True, exist_ok=True)
        if not f.exists():
            with open(f, "w", encoding="utf-8") as fp:
                if f.name == "__init__.py":
                    fp.write("# Package marker\n")
                else:
                    fp.write("")
    print("FastAPI 目录结构已创建在当前目录：", BASE_DIR.resolve())


if __name__ == "__main__":
    main()
