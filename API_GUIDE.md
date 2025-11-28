# EE547 Grading System API 使用指南

## 🚀 快速开始

### 前置要求

- Python 3.10+
- PostgreSQL 数据库（或 Docker）

---

## 方式一：手动安装 PostgreSQL（推荐用于学习）

### 1. 安装 PostgreSQL

下载并安装 PostgreSQL：https://www.postgresql.org/download/

安装时记住设置的 postgres 用户密码。

### 2. 创建数据库和用户

```bash
# 启动 PostgreSQL 命令行（Windows）
"D:\PostgreSQL\16\bin\psql.exe" -U postgres

# 或者（Mac/Linux）
psql -U postgres
```

在 psql 中执行以下命令：

```sql
-- 创建用户
CREATE USER ee547_user WITH PASSWORD 'password';

-- 创建数据库
CREATE DATABASE ee547_db OWNER ee547_user;

-- 授权（重要！）
GRANT ALL PRIVILEGES ON DATABASE ee547_db TO ee547_user;

-- 连接到新数据库并授权 schema
\c ee547_db
GRANT ALL ON SCHEMA public TO ee547_user;

-- 退出
\q
```

### 3. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量（可选）

如果数据库密码不是 `password`，需要创建 `.env` 文件：

```bash
# 复制示例文件
cp .env.example .env

# 修改 .env 中的数据库连接
DATABASE_URL=postgresql+psycopg2://ee547_user:你的密码@localhost:5432/ee547_db
```

### 5. 启动应用

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**应用会在启动时自动创建所有数据库表！**

### 6. 访问 API 文档

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## 方式二：使用 Docker Compose（推荐用于快速开发）

这种方式**不需要手动安装 PostgreSQL**，Docker 会自动帮你搞定！

### 1. 安装 Docker

确保已安装 Docker Desktop

### 2. 启动数据库服务

```bash
# 启动 PostgreSQL 和 Redis
docker-compose up -d

# 查看运行状态
docker-compose ps
```

这会启动：
- PostgreSQL 数据库（端口 5432）
- Redis（端口 6379）

### 3. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 4. 启动应用

```bash
uvicorn app.main:app --reload
```

### 5. 停止服务

```bash
# 停止但保留数据
docker-compose stop

# 停止并删除容器（数据会保留在 volume 中）
docker-compose down

# 停止并删除所有数据
docker-compose down -v
```

---

## 📁 项目结构

```
app/
├── api/v1/endpoints/     # API 端点
│   ├── auth.py          # 认证（注册/登录）
│   ├── users.py         # 用户管理
│   ├── questions.py     # 题目管理
│   ├── submissions.py   # 答案提交
│   └── scores.py        # 评分管理
├── core/                # 核心功能
│   ├── config.py        # 配置
│   └── security.py      # 安全认证
├── db/                  # 数据库
├── models/              # 数据模型
├── schemas/             # Pydantic schemas
└── main.py              # 应用入口
```

## 🔑 API 端点

### Authentication (认证)

- `POST /api/v1/auth/register` - 用户注册
- `POST /api/v1/auth/login` - 登录（JSON）
- `POST /api/v1/auth/token` - 登录（OAuth2 表单）

### Users (用户)

- `GET /api/v1/users/me` - 获取当前用户信息
- `PUT /api/v1/users/me` - 更新当前用户信息
- `GET /api/v1/users/` - 获取用户列表（仅教师）
- `GET /api/v1/users/{user_id}` - 获取指定用户（仅教师）
- `POST /api/v1/users/` - 创建用户（仅教师）
- `DELETE /api/v1/users/{user_id}` - 删除用户（仅教师）

### Questions (题目)

- `GET /api/v1/questions/` - 获取题目列表
- `GET /api/v1/questions/{question_id}` - 获取单个题目
- `POST /api/v1/questions/` - 创建题目（仅教师）
- `PUT /api/v1/questions/{question_id}` - 更新题目（仅教师）
- `DELETE /api/v1/questions/{question_id}` - 删除题目（仅教师）

### Submissions (提交)

- `GET /api/v1/submissions/` - 获取提交列表
- `GET /api/v1/submissions/{submission_id}` - 获取单个提交
- `POST /api/v1/submissions/` - 创建提交（仅学生）
- `PUT /api/v1/submissions/{submission_id}` - 更新提交（仅学生）
- `DELETE /api/v1/submissions/{submission_id}` - 删除提交

### Scores (评分)

- `GET /api/v1/scores/` - 获取评分列表（仅教师）
- `GET /api/v1/scores/{submission_id}` - 获取评分详情（仅教师）
- `PUT /api/v1/scores/{submission_id}` - 更新评分（仅教师）
- `POST /api/v1/scores/{submission_id}/confirm` - 确认 ML 评分（仅教师）
- `GET /api/v1/scores/pending/count` - 获取待处理评分数量（仅教师）

## 🔐 认证流程

1. **注册用户**：
```bash
POST /api/v1/auth/register
{
  "email": "student@example.com",
  "password": "password123",
  "name": "张三",
  "role": "student"  # 或 "teacher"
}
```

2. **登录获取 Token**：
```bash
POST /api/v1/auth/login
{
  "email": "student@example.com",
  "password": "password123"
}

# 返回
{
  "access_token": "eyJhbGc...",
  "token_type": "bearer"
}
```

3. **使用 Token 访问 API**：
```bash
Authorization: Bearer eyJhbGc...
```

## 👥 角色权限

### 学生 (student)
- ✅ 查看所有题目
- ✅ 提交答案
- ✅ 查看自己的提交和评分
- ❌ 不能管理题目
- ❌ 不能查看其他学生的提交

### 教师 (teacher)
- ✅ 创建/修改/删除题目
- ✅ 查看所有学生的提交
- ✅ 评分和修改评分
- ✅ 管理用户
- ✅ 查看统计数据

## 📊 数据流程

1. **教师**创建题目
2. **学生**提交答案
3. **ML 模型**自动评分（状态：`pending_ml` → `ml_scored`）
4. **教师**审核并确认或修改评分（状态：`ml_scored` → `graded`）

## 🛠️ 开发提示

### 使用 Swagger UI 测试 API

1. 访问 http://localhost:8000/docs
2. 点击右上角 "Authorize" 按钮
3. 输入 token: `Bearer <your_token>`
4. 现在可以测试所有需要认证的 API

### 数据库迁移

如果使用 Alembic 进行数据库迁移：

```bash
# 初始化
alembic init alembic

# 创建迁移
alembic revision --autogenerate -m "Initial migration"

# 应用迁移
alembic upgrade head
```

## 🐛 常见问题

### 1. 导入错误
如果看到 "无法解析导入" 错误，请安装依赖：
```bash
pip install -r requirements.txt
```

### 2. 数据库连接错误

**错误信息**：`FATAL: password authentication failed for user "ee547_user"`

**解决方法**：
1. 确认 PostgreSQL 已启动
2. 确认用户和数据库已创建
3. 检查密码是否正确
4. 在 psql 中重新授权：
```sql
\c ee547_db
GRANT ALL ON SCHEMA public TO ee547_user;
```

### 3. 数据库表不存在

**错误信息**：`relation "users" does not exist`

**原因**：应用启动时未自动创建表

**解决方法**：
- 确保 `app/main.py` 中有 `Base.metadata.create_all(bind=engine)`
- 重启应用即可自动创建表

### 4. Token 过期
Token 默认 7 天有效期，过期后需要重新登录。

### 5. 端口被占用

**错误信息**：`Address already in use`

**解决方法**：
```bash
# 更改端口
uvicorn app.main:app --reload --port 8001

# 或者找到占用进程并关闭
netstat -ano | findstr :8000  # Windows
lsof -i :8000                  # Mac/Linux
```

### 6. Docker 数据库连接问题

如果使用 Docker，应用连接数据库时使用：
- 在 Docker 内部：`db:5432`（容器名）
- 在 Docker 外部：`localhost:5432`（主机）

---

## 🔍 验证数据库设置

启动应用后，检查数据库表是否创建成功：

```bash
# 连接到数据库
psql -U ee547_user -d ee547_db

# 查看所有表
\dt

# 应该看到：
# users
# questions
# submissions

# 查看 users 表结构
\d users

# 退出
\q
```

---

## 🧪 快速测试

启动应用后，可以快速测试 API：

### 1. 检查健康状态
```bash
curl http://localhost:8000/api/v1/health
```

### 2. 注册用户
```bash
curl -X POST "http://localhost:8000/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "teacher@example.com",
    "password": "password123",
    "name": "李老师",
    "role": "teacher"
  }'
```

### 3. 登录
```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "teacher@example.com",
    "password": "password123"
  }'
```

会返回类似：
```json
{
  "access_token": "eyJhbGc...",
  "token_type": "bearer"
}
```

### 4. 使用 Token 访问 API
```bash
curl -X GET "http://localhost:8000/api/v1/users/me" \
  -H "Authorization: Bearer <你的token>"
```

---

## 📝 下一步

- [ ] 实现 ML 评分服务（`app/services/ml_client.py`）
- [ ] 实现异步任务队列（`app/workers/`）
- [ ] 集成 AWS S3 存储
- [ ] 添加单元测试
- [ ] 部署到云端
