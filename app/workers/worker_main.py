# app/workers/worker_main.py
from rq import Worker, Connection
from app.workers.queue import get_redis_connection

# 要监听的队列名，要跟 queue.py 里的一致
QUEUE_NAMES = ["scoring"]

def main():
    redis_conn = get_redis_connection()
    with Connection(redis_conn):
        worker = Worker(QUEUE_NAMES)
        worker.work()

if __name__ == "__main__":
    main()
