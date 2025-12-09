# app/workers/worker_main.py

from rq import Queue, SimpleWorker
from app.workers.queue import get_redis_connection


QUEUE_NAMES = ["scoring"]


def main():
    redis_conn = get_redis_connection()

    queues = [Queue(name, connection=redis_conn) for name in QUEUE_NAMES]


    worker = SimpleWorker(queues, connection=redis_conn)


    worker.work()


if __name__ == "__main__":
    main()
