```text
ee547-final/
├── app/
│   ├── api/
│   │   ├── v1/
│   │   │   ├── endpoints/
│   │   │   │   ├── auth.py          # 登录/注册（如果需要）
│   │   │   │   ├── users.py         # 教师/学生的基本信息
│   │   │   │   ├── questions.py     # 题目相关接口（老师上传题目/参考答案）
│   │   │   │   ├── submissions.py   # 学生提交答案、查看结果
│   │   │   │   ├── scores.py        # 老师查看/确认/修改评分
│   │   │   │   └── health.py        # 健康检查
│   │   │   └── __init__.py
│   │   └── __init__.py
│   │
│   ├── core/
│   │   ├── config.py        # 读取环境变量、全局配置（DB、S3、队列等）
│   │   ├── security.py      # 鉴权相关（JWT / OAuth2）
│   │   └── logging_config.py# 日志配置（可选）
│   │
│   ├── db/
│   │   ├── base.py          # Base = declarative_base()
│   │   ├── session.py       # SessionLocal / async session
│   │   └── init_db.py       # 初始化数据库（创建表/初始用户等，可选）
│   │
│   ├── models/              # SQLAlchemy ORM 模型
│   │   ├── user.py          # users 表
│   │   ├── question.py      # questions 表
│   │   ├── submission.py    # submissions 表
│   │   └── score.py         # scores 表
│   │
│   ├── schemas/             # Pydantic 模型（请求/响应）
│   │   ├── user.py
│   │   ├── auth.py
│   │   ├── question.py
│   │   ├── submission.py
│   │   └── score.py
│   │
│   ├── services/            # 业务逻辑/领域服务
│   │   ├── question_service.py   # 题目相关业务
│   │   ├── submission_service.py # 提交相关业务（创建提交、状态变更等）
│   │   ├── scoring_service.py    # 封装“创建打分任务”“查询打分结果”等
│   │   ├── ml_client.py          # 调用 ML 模型的封装（直接加载模型 or 调用另一个服务）
│   │   └── s3_client.py          # 与 S3 交互（加载模型/存文件）
│   │
│   ├── workers/              # 异步 worker 相关
│   │   ├── queue.py          # 队列封装（RQ / SQS / Celery）
│   │   ├── tasks.py          # 具体任务：从 submission 取数据，跑模型，写 scores
│   │   └── worker_main.py    # worker 入口：rq worker / celery worker 在这里启动
│   │
│   ├── main.py               # FastAPI 应用入口
│   └── __init__.py
│
├── alembic/                  # 数据库迁移（如果用 Alembic）
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
│
├── tests/
│   ├── __init__.py
│   ├── test_questions.py
│   ├── test_submissions.py
│   └── test_scoring.py
│
├── requirements.txt          # 或 pyproject.toml
├── .env.example              # 环境变量示例（DB_URI、S3、QUEUE_URL 等）
├── docker-compose.yml        # 本地开发时起 DB + Redis/SQS mock + API + worker
├── Dockerfile                # 部署到 ECS/EC2 用
└── README.md
```

```text
1. Download PostgreSQL and set a password during installation.

2. Install dependencies in the root directory:

pip install fastapi "uvicorn[standard]" SQLAlchemy pydantic pydantic-settings

pip install "passlib[bcrypt]" "python-jose[cryptography]"

pip install psycopg2-binary

pip install python-multipart

3. Start the database:

"D:\PostgreSQL\16\bin\psql.exe" -U postgres

After entering the password:

CREATE USER ee547_user WITH PASSWORD 'ee547_pass';

CREATE DATABASE ee547_db OWNER ee547_user;

Type \q to exit the database.

4. Start Fastapi:

uvicorn app.main:app --reload

5. Start Frontend
npm start

6. Start using

Notice: If you encounter a version incompatibility error 
during registration, install an older version of bcrypt.

pip install "bcrypt==4.0.1"  

Open your browser and access: http://127.0.0.1:3000