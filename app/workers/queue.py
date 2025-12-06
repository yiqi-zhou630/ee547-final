# app/workers/queue.py

from typing import Any, Callable

from redis import Redis
from rq import Queue

from app.core.config import settings

# 可根据你的 .env / config 调整这里的属性名
# 比如 .env 里写的是 QUEUE_URL=redis://redis:6379/0
# 那 config.py 里通常会有 settings.QUEUE_URL
_DEFAULT_QUEUE_NAME = "default"
_SCORING_QUEUE_NAME = "scoring"

_redis_conn: Redis | None = None


def get_redis_connection() -> Redis:
    global _redis_conn
    if _redis_conn is None:
        # 如果你用的是 REDIS_URL，把这里改成 settings.REDIS_URL 即可
        redis_url = settings.QUEUE_URL
        _redis_conn = Redis.from_url(redis_url)
    return _redis_conn


def get_queue(name: str = _DEFAULT_QUEUE_NAME) -> Queue:
    return Queue(name, connection=get_redis_connection())


def enqueue_job(
    func: Callable[..., Any],
    *args: Any,
    queue_name: str = _DEFAULT_QUEUE_NAME,
    **kwargs: Any,
) -> str:
    """
    封装一个通用的“入队”函数，方便以后有别的异步任务。

    Parameters
    ----------
    func : Callable
        需要在 worker 中执行的函数（必须可被 import）
    *args :
        传给 func 的位置参数
    queue_name : str, optional
        队列名，默认 "default"
    **kwargs :
        传给 func 的关键字参数

    Returns
    -------
    str
        RQ Job 的 ID（可以存数据库或日志，方便排查）
    """
    q = get_queue(queue_name)
    job = q.enqueue(func, *args, **kwargs)
    return job.id


def enqueue_scoring_task(submission_id: int) -> str:
    from app.workers.tasks import scoring_task

    q = get_queue(_SCORING_QUEUE_NAME)
    job = q.enqueue(scoring_task, submission_id)
    return job.id
