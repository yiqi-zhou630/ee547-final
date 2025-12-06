# app/workers/worker_main.py
from rq import Worker, Queue
from app.workers.queue import get_redis_connection
from app.workers.queue import _SCORING_QUEUE_NAME  # 也可以自己手写 "scoring"

QUEUE_NAMES = ["scoring"]


def main():
    redis_conn = get_redis_connection()

    queues = [Queue(name, connection=redis_conn) for name in QUEUE_NAMES]

    worker = Worker(queues, connection=redis_conn)

    worker.work()


if __name__ == "__main__":
    main()
