# app/workers/queue.py

from typing import Any, Callable

from redis import Redis
from rq import Queue

from app.core.config import settings

_DEFAULT_QUEUE_NAME = "default"
_SCORING_QUEUE_NAME = "scoring"

_redis_conn: Redis | None = None


def get_redis_connection() -> Redis:
    global _redis_conn
    if _redis_conn is None:
        redis_url = settings.REDIS_URL
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

    q = get_queue(queue_name)
    job = q.enqueue(func, *args, **kwargs)
    return job.id


def enqueue_scoring_task(submission_id: int) -> str:
    from app.workers.tasks import scoring_task

    q = get_queue(_SCORING_QUEUE_NAME)
    job = q.enqueue(scoring_task, submission_id)
    return job.id
